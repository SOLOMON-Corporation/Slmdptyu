import os
import json
import torch
import pandas as pd
from tqdm import tqdm

from slmdptyu.solomon.ultralytics.nn.autobackend import AutoBackend
from slmdptyu.solomon.ultralytics.yolo.data.utils import check_det_dataset
from slmdptyu.solomon.ultralytics.yolo.utils import LOGGER, TQDM_BAR_FORMAT, colorstr, emojis 
from slmdptyu.ultralytics.ultralytics.yolo.utils import callbacks
from slmdptyu.ultralytics.ultralytics.yolo.utils.checks import check_imgsz
from slmdptyu.ultralytics.ultralytics.yolo.utils.ops import Profile
from slmdptyu.ultralytics.ultralytics.yolo.utils.torch_utils import de_parallel, select_device, smart_inference_mode 
from slmdptyu.ultralytics.ultralytics.yolo.engine.validator import BaseValidator


class BaseSolValidator(BaseValidator):

    @smart_inference_mode()
    def __call__(self, trainer=None, model=None):
        """
        Supports validation of a pre-trained model if passed or a model being trained
        if trainer is passed (trainer gets priority).
        """
        self.training = trainer is not None
        if self.training:
            self.device = trainer.device
            self.data = trainer.data
            model = trainer.ema.ema or trainer.model
            self.args.half = self.device.type != 'cpu'  # force FP16 val during training
            model = model.half() if self.args.half else model.float()
            self.model = model
            self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)
            model.eval()
        else:
            callbacks.add_integration_callbacks(self)
            # self.run_callbacks('on_val_start')
            assert model is not None, 'Either trainer or model is needed for validation'
            self.device = select_device(self.args.device, self.args.batch)
            self.args.half &= self.device.type != 'cpu'
            model = AutoBackend(model, device=self.device, dnn=self.args.dnn, data=self.args.data, fp16=self.args.half)
            self.model = model
            stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
            imgsz = check_imgsz(self.args.imgsz, stride=stride)
            if engine:
                self.args.batch = model.batch_size
            else:
                self.device = model.device
                if not pt and not jit:
                    self.args.batch = 1  # export.py models default to batch-size 1
                    LOGGER.info(f'Forcing batch=1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models')

            if isinstance(self.args.data, str) and self.args.data.endswith('.yaml'):
                self.data = check_det_dataset(self.args.data)
            else:
                raise FileNotFoundError(emojis(f"Dataset '{self.args.data}' for task={self.args.task} not found ❌"))

            if self.device.type == 'cpu':
                self.args.workers = 0  # faster CPU val as time dominated by inference, not dataloading
            if not pt:
                self.args.rect = False
            self.dataloader = self.dataloader or self.get_dataloader(self.data.get(self.args.split), self.args.batch)

            model.eval()
            model.warmup(imgsz=(1 if pt else self.args.batch, 3, imgsz, imgsz))  # warmup

        dt = Profile(), Profile(), Profile(), Profile()
        n_batches = len(self.dataloader)
        desc = self.get_desc()
        LOGGER.info('\n')
        # NOTE: keeping `not self.training` in tqdm will eliminate pbar after segmentation evaluation during training,
        # which may affect classification task since this arg is in yolov5/classify/val.py.
        # bar = tqdm(self.dataloader, desc, n_batches, not self.training, bar_format=TQDM_BAR_FORMAT)
        # bar = tqdm(self.dataloader, desc, n_batches, bar_format=TQDM_BAR_FORMAT)
        LOGGER.info(desc)
        self.init_metrics(de_parallel(model))
        self.jdict = []  # empty before each val
        for batch_i, batch in enumerate(self.dataloader):
            # self.run_callbacks('on_val_batch_start')
            self.batch_i = batch_i
            # preprocess
            with dt[0]:
                batch = self.preprocess(batch)

            # inference
            with dt[1]:
                preds = model(batch['img'])

            # loss
            with dt[2]:
                if self.training:
                    self.loss += trainer.criterion(preds, batch)[1]

            # postprocess
            with dt[3]:
                preds = self.postprocess(preds)

            self.update_metrics(preds, batch)
            if self.args.plots and batch_i < 3:
                self.plot_val_samples(batch, batch_i)
                self.plot_predictions(batch, preds, batch_i)

            # self.run_callbacks('on_val_batch_end')
        stats = self.get_stats()
        self.check_stats(stats)
        self.print_results()
        self.speed = dict(zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1E3 for x in dt)))
        self.finalize_metrics()
        # self.run_callbacks('on_val_end')
        if self.training:
            model.float()
            metric = {f'{k}/{k_}': v_ for k, v in stats.items() for k_, v_ in v.items()}
            results = {**metric, **trainer.label_loss_items(self.loss.cpu() / len(self.dataloader), prefix='val')}
            stats = {k: round(float(v), 5) for k, v in results.items()}  # return results as 5 decimal place floats
        else:
            LOGGER.info('Speed: %.1fms preprocess, %.1fms inference, %.1fms loss, %.1fms postprocess per image' %
                        tuple(self.speed.values()))
            metric = {f'{k}/{k_}': v_ for k, v in stats.items() for k_, v_ in v.items()}
            if self.args.save_json and self.jdict:
                with open(str(self.save_dir / 'predictions.json'), 'w') as f:
                    LOGGER.info(f'Saving {f.name}...')
                    json.dump(self.jdict, f)  # flatten and save
                stats = self.eval_json(stats)  # update stats
                results = {**metric, **trainer.label_loss_items(self.loss.cpu() / len(self.dataloader), prefix='val')}
                stats = {k: round(float(v), 5) for k, v in results.items()}  # return results as 5 decimal place floats
            if self.args.plots or self.args.save_json:
                LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")
        
        if trainer != None:
            metric.update({'epoch':trainer.epoch+1})
        else:
            metric.update({'epoch':-1})
        compare_pd = pd.DataFrame(metric, index=[0])
        if os.path.exists(self.save_dir / 'solmetric.csv'):
            compare_pre = pd.read_csv(self.save_dir / 'solmetric.csv')
            compare_pre = compare_pre.append(compare_pd)
            compare_pre.to_csv(
                self.save_dir / 'solmetric.csv', index=False, float_format='%.3f')
        else:
            compare_pd.to_csv(
                self.save_dir / 'solmetric.csv', index=False, float_format='%.3f')

        return stats