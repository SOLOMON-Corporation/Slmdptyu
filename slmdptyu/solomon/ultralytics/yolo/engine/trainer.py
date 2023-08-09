import time
import torch
import torch.distributed as dist
import numpy as np
from copy import deepcopy
from pathlib import Path
from torch.cuda import amp
from torch.optim import lr_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from datetime import datetime
from slmdptyu.solomon.ultralytics.yolo.utils import DEFAULT_CFG
from slmdptyu.ultralytics.ultralytics.yolo.engine.trainer import BaseTrainer
from slmdptyu.solomon.ultralytics.nn.tasks import attempt_load_one_weight , attempt_load_weights
from slmdptyu.ultralytics.ultralytics.yolo.cfg import get_cfg
from slmdptyu.solomon.ultralytics.yolo.data.utils import check_det_dataset
from slmdptyu.solomon.ultralytics.yolo.utils import (LOGGER, RANK, SETTINGS, 
    __version__, clean_url, colorstr, emojis, yaml_save)
from slmdptyu.ultralytics.ultralytics.yolo.utils import callbacks
from slmdptyu.ultralytics.ultralytics.yolo.utils.autobatch import check_train_batch_size
from slmdptyu.ultralytics.ultralytics.yolo.utils.checks import check_imgsz, print_args
from slmdptyu.ultralytics.ultralytics.yolo.utils.files import increment_path, get_latest_run
from slmdptyu.ultralytics.ultralytics.yolo.utils.torch_utils import (
    de_parallel, EarlyStopping, ModelEMA, init_seeds, one_cycle, select_device, strip_optimizer)


class BaseSolTrainer(BaseTrainer):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initializes the BaseTrainer class.

        Args:
            cfg (str, optional): Path to a configuration file. Defaults to DEFAULT_CFG.
            overrides (dict, optional): Configuration overrides. Defaults to None.
        """
        self.args = get_cfg(cfg, overrides)
        self.device = select_device(self.args.device, self.args.batch)
        self.check_resume()
        self.validator = None
        self.model = None
        self.metrics = None
        init_seeds(self.args.seed + 1 + RANK, deterministic=self.args.deterministic)

        # Dirs
        project = self.args.project or Path(SETTINGS['runs_dir']) / self.args.task
        name = self.args.name or f'{self.args.mode}'
        if hasattr(self.args, 'save_dir'):
            self.save_dir = Path(self.args.save_dir)
        else:
            self.save_dir = Path(
                increment_path(Path(project) / name, exist_ok=self.args.exist_ok if RANK in (-1, 0) else True))
        # self.wdir = self.save_dir / 'weights'  # weights dir
        if RANK in (-1, 0):
            # self.wdir.mkdir(parents=True, exist_ok=True)  # make dir
            self.args.save_dir = str(self.save_dir)
            yaml_save(self.save_dir / 'args.yaml', vars(self.args))  # save run args
        if self.args.val:
            self.best = self.save_dir / "model_best.pth"  # checkpoint paths
        else:
            self.best = self.save_dir / "model_final.pth"  # checkpoint paths
        self.save_period = self.args.save_period

        self.batch_size = self.args.batch
        self.epochs = self.args.epochs
        self.start_epoch = 0
        if RANK == -1:
            print_args(vars(self.args))

        # Device
        if self.device.type == 'cpu':
            self.args.workers = 0  # faster CPU training as time dominated by inference, not dataloading

        # Model and Dataloaders.
        self.model = self.args.model
        try:
            if self.args.data.endswith('.yaml') or self.args.task in ('detect', 'segment', 'pose'):
                self.data = check_det_dataset(self.args.data)
                if 'yaml_file' in self.data:
                    self.args.data = self.data['yaml_file']  # for validating 'yolo train data=url.zip' usage
        except Exception as e:
            raise RuntimeError(emojis(f"Dataset '{clean_url(self.args.data)}' error ❌ {e}")) from e

        self.trainset, self.testset = self.get_dataset(self.data)
        self.ema = None

        # Optimization utils init
        self.lf = None
        self.scheduler = None

        # Epoch level metrics
        self.best_fitness = None
        self.fitness = None
        self.loss = None
        self.tloss = None
        self.loss_names = ['Loss']
        self.csv = self.save_dir / 'results.csv'
        self.plot_idx = [0, 1, 2]

        # Callbacks
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        if RANK in (-1, 0):
            callbacks.add_integration_callbacks(self)

        # Connect to C#
        try:
            from slmutils.general import Init
            self.tfControl = Init()
            print(colorstr("yellow", 'Run in SolVision'))
            self.tfControl.getLossName('Total_Loss')
        except:
            from slmutils.general import Init2
            self.tfControl = Init2()
            LOGGER.info(colorstr("yellow", 'Run in Python'))
            self.tfControl.getLossName('Total_Loss')

    def run_callbacks(self, event: str):
        if self.args.save_callbacks:
            for callback in self.callbacks.get(event, []):
                callback(self)

    def train(self):
        world_size = 1
        self._do_train(world_size)

    def _setup_train(self, world_size):
        """
        Builds dataloaders and optimizer on correct rank process.
        """
        # Model
        self.run_callbacks('on_pretrain_routine_start')
        ckpt = self.setup_model()
        self.model = self.model.to(self.device)
        self.set_model_attributes()
        # Check AMP
        self.amp = torch.tensor(self.args.amp).to(self.device)  # True or False
        if self.amp and RANK in (-1, 0):  # Single-GPU and DDP
            callbacks_backup = callbacks.default_callbacks.copy()  # backup callbacks as check_amp() resets them
            self.amp = torch.tensor(True, device=self.device)
            callbacks.default_callbacks = callbacks_backup  # restore callbacks
        if RANK > -1:  # DDP
            dist.broadcast(self.amp, src=0)  # broadcast the tensor from rank 0 to all other ranks (returns None)
        self.amp = bool(self.amp)  # as boolean
        self.scaler = amp.GradScaler(enabled=self.amp)
        if world_size > 1:
            self.model = DDP(self.model, device_ids=[RANK])
        # Check imgsz
        gs = max(int(self.model.stride.max() if hasattr(self.model, 'stride') else 32), 32)  # grid size (max stride)
        self.args.imgsz = check_imgsz(self.args.imgsz, stride=gs, floor=gs, max_dim=1)
        # Batch size
        if self.batch_size == -1:
            if RANK == -1:  # single-GPU only, estimate best batch size
                self.batch_size = check_train_batch_size(self.model, self.args.imgsz, self.amp)
            else:
                SyntaxError('batch=-1 to use AutoBatch is only available in Single-GPU training. '
                            'Please pass a valid batch size value for Multi-GPU DDP training, i.e. batch=16')

        # Optimizer
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)  # accumulate loss before optimizing
        weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs  # scale weight_decay
        self.optimizer = self.build_optimizer(model=self.model,
                                              name=self.args.optimizer,
                                              lr=self.args.lr0,
                                              momentum=self.args.momentum,
                                              decay=weight_decay)
        # Scheduler
        if self.args.cos_lr:
            self.lf = one_cycle(1, self.args.lrf, self.epochs)  # cosine 1->hyp['lrf']
        else:
            self.lf = lambda x: (1 - x / self.epochs) * (1.0 - self.args.lrf) + self.args.lrf  # linear
        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)
        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False

        # dataloaders
        batch_size = self.batch_size // world_size if world_size > 1 else self.batch_size
        self.train_loader = self.get_dataloader(self.trainset, batch_size=batch_size, rank=RANK, mode='train')
        if RANK in (-1, 0):
            self.test_loader = self.get_dataloader(self.testset, batch_size=batch_size * 2, rank=-1, mode='val')
            self.validator = self.get_validator()
            metric_keys = self.validator.metrics.keys
            for name in self.data['names'].values():
                metric_keys.insert(-1, name)
            metric_keys = metric_keys +  self.label_loss_items(prefix='val')
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))
            self.ema = ModelEMA(self.model)
            if self.args.plots and not self.args.v5loader:
                self.plot_training_labels()
        self.resume_training(ckpt)
        self.scheduler.last_epoch = self.start_epoch - 1  # do not move
        # self.curr_iters = np.arange(1, len(self.train_loader)+1)
        self.run_callbacks('on_pretrain_routine_end')

    def _do_train(self, world_size=1):
        if world_size > 1:
            self._setup_ddp(world_size)

        self._setup_train(world_size)

        stop_without_saving = False

        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        nb = len(self.train_loader)  # number of batches
        nw = max(round(self.args.warmup_epochs * nb), 100)  # number of warmup iterations
        last_opt_step = -1
        self.run_callbacks('on_train_start')
        LOGGER.info(f'Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n'
                    f'Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n'
                    f"Logging results to {colorstr('bold', self.save_dir)}\n"
                    f'Starting training for {self.epochs} epochs...')
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch
            self.run_callbacks('on_train_epoch_start')
            self.model.train()
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(self.train_loader)
            # Update dataloader attributes (optional)
            if epoch == (self.epochs - self.args.close_mosaic):
                LOGGER.info('Closing dataloader mosaic')
                if hasattr(self.train_loader.dataset, 'mosaic'):
                    self.train_loader.dataset.mosaic = False
                if hasattr(self.train_loader.dataset, 'close_mosaic'):
                    self.train_loader.dataset.close_mosaic(hyp=self.args)

            if RANK in (-1, 0):
                LOGGER.info(self.progress_string())
                # pbar = tqdm(enumerate(self.train_loader), total=nb, bar_format=TQDM_BAR_FORMAT)
                pbar = enumerate(self.train_loader)
            self.tloss = None
            self.optimizer.zero_grad()
            for i, batch in pbar:
                self.run_callbacks('on_train_batch_start')
                # Warmup
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    self.accumulate = max(1, np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round())
                    for j, x in enumerate(self.optimizer.param_groups):
                        # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x['lr'] = np.interp(
                            ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x['initial_lr'] * self.lf(epoch)])
                        if 'momentum' in x:
                            x['momentum'] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

                # Forward
                with torch.cuda.amp.autocast(self.amp):
                    batch = self.preprocess_batch(batch)
                    preds = self.model(batch['img'])
                    self.loss, self.loss_items = self.criterion(preds, batch)
                    if RANK != -1:
                        self.loss *= world_size
                    self.tloss = (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None \
                        else self.loss_items

                # Backward
                self.scaler.scale(self.loss).backward()

                # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

                # Log
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                loss_len = self.tloss.shape[0] if len(self.tloss.size()) else 1
                losses = self.tloss if loss_len > 1 else torch.unsqueeze(self.tloss, 0)
                if RANK in (-1, 0):
                    # pbar.set_description(
                    #     ('%11s' * 2 + '%11.4g' * (2 + loss_len)) %
                    #     (f'{epoch + 1}/{self.epochs}', mem, *losses, batch['cls'].shape[0], batch['img'].shape[-1]))
                    progress = f"{epoch + 1}/{self.epochs}"
                    losses_str = ''.join([f"{l:11.4g}" for l in losses])
                    Instances = batch["cls"].shape[0]
                    Size = batch["img"].shape[-1]
                    # tfControl
                    tfC_epoch = int((epoch+1)/self.epochs*100)
                    self.tfControl.getEpoch(tfC_epoch)
                    tfC_loss = float(losses.mean().item()*100)
                    self.tfControl.getLoss(tfC_loss)
                    
                    self.run_callbacks('on_batch_end')
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)

                self.run_callbacks('on_train_batch_end')
            LOGGER.info(f'{progress:>11s}{mem:>11s}{losses_str}{Instances:11.4g}{Size:11.4g}')

            self.lr = {f'lr/pg{ir}': x['lr'] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers

            self.scheduler.step()
            # tfControl
            if self.tfControl.NeedToSaveModel():
                LOGGER.info("NeedToSaveModel")
                self.stop = True
            elif self.tfControl.NeedToStopTrain():
                LOGGER.info("NeedToStopTrain")
                stop_without_saving = True
                self.stop = True

            if self.stop:
                break  # must break all DDP ranks
            
            self.run_callbacks('on_train_epoch_end')

            if RANK in (-1, 0):

                # Validation
                self.ema.update_attr(self.model, include=['yaml', 'nc', 'args', 'names', 'stride', 'class_weights'])
                final_epoch = (epoch + 1 == self.epochs) or self.stopper.possible_stop

                if self.args.val:
                    # if 0 in (self.curr_iters % self.args.eval_step) or final_epoch:
                    if ((epoch+1) % self.args.eval_step) == 0 or final_epoch:
                        self.metrics, self.fitness = self.validate()
                    
                        # Save model
                        if self.args.save or (epoch + 1 == self.epochs):
                            self.save_model()
                            self.run_callbacks('on_model_save')

                    self.stop = self.stopper(epoch + 1, self.fitness)
                    # self.curr_iters += nb

            tnow = time.time()
            self.epoch_time = tnow - self.epoch_time_start
            self.epoch_time_start = tnow
            self.run_callbacks('on_fit_epoch_end')
            torch.cuda.empty_cache()  # clears GPU vRAM at end of epoch, can help with out of memory errors

            # Early Stopping
            if RANK != -1:  # if DDP training
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                if RANK != 0:
                    self.stop = broadcast_list[0]
            if self.stop:
                break  # must break all DDP ranks

        if RANK in (-1, 0):
            # Save model
            if not stop_without_saving:
                if not self.args.val:
                    self.save_model()
                    self.run_callbacks('on_model_save')

                if self.args.val and not self.best.exists():
                    self.save_model()
                    self.run_callbacks('on_model_save')

                # Do final val with model_final.pt
                LOGGER.info(f'\n{epoch - self.start_epoch + 1} epochs completed in '
                            f'{(time.time() - self.train_time_start) / 3600:.3f} hours.')
            
            self.final_eval()

            if not stop_without_saving:
                if self.args.val:
                    target_file = self.best.parent / 'model_final.pth'
                    if target_file.exists():
                        target_file.unlink()
                    self.best.rename(self.best.parent / 'model_final.pth')   

            self.run_callbacks('on_train_end')
        torch.cuda.empty_cache()
        self.run_callbacks('teardown')

    def validate(self):
        metrics = self.validator(self)
        f_score = metrics[f'all/f{self.args.eval_fscore}']
        fitness = 0 if np.isnan(f_score) else f_score
        if not self.best_fitness or self.best_fitness <= fitness:
            self.best_fitness = fitness
        return metrics, fitness
    
    def save_model(self):
        ckpt = {
            'epoch': self.epoch,
            'best_fitness': self.best_fitness,
            'model': deepcopy(de_parallel(self.model)).half(),
            'ema': deepcopy(self.ema.ema).half(),
            'updates': self.ema.updates,
            'optimizer': self.optimizer.state_dict(),
            'train_args': vars(self.args),  # save as dict
            'date': datetime.now().isoformat(),
            'version': __version__}

        # Save last, best and delete
        if self.best_fitness == self.fitness:
            torch.save(ckpt, self.best)
        del ckpt

    def setup_model(self):
        """
        load/create/download model for any task.
        """
        if isinstance(self.model, torch.nn.Module):  # if model is loaded beforehand. No setup needed
            return

        model, weights = self.model, None
        ckpt = None
        if str(model).endswith('.pt') or str(model).endswith('.pth'):
            weights, ckpt = attempt_load_one_weight(model)
            cfg = ckpt['model'].yaml
        else:
            cfg = model
        self.model = self.get_model(cfg=cfg, weights=weights, verbose=RANK == -1)  # calls Model(cfg, weights)
        return ckpt
    
    def final_eval(self):
        if self.best.exists():
            strip_optimizer(self.best)  # strip optimizers
            LOGGER.info(f'\nValidating {self.best}...')
            self.metrics = self.validator(model=self.best)
            self.metrics = {f'{k}/{k_}': v_ for k, v in self.metrics.items() for k_, v_ in v.items()}
            self.run_callbacks('on_fit_epoch_end')

    def check_resume(self):
        resume = self.args.resume
        if resume:
            try:
                last = Path(
                    resume if isinstance(resume, (str,
                                                              Path)) and Path(resume).exists() else get_latest_run())
                self.args = get_cfg(attempt_load_weights(last).args)
                self.args.model, resume = str(last), True  # reinstate
            except Exception as e:
                raise FileNotFoundError('Resume checkpoint not found. Please pass a valid checkpoint to resume from, '
                                        "i.e. 'yolo train resume model=path/to/last.pt'") from e
        self.resume = resume