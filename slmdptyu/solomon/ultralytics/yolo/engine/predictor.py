import cv2
import numpy as np
from pathlib import Path
from slmdptyu.solomon.ultralytics.nn.autobackend import AutoBackend
from slmdptyu.ultralytics.ultralytics.yolo.cfg import get_cfg
from slmdptyu.solomon.ultralytics.yolo.utils import LOGGER, colorstr, ops
from slmdptyu.ultralytics.ultralytics.yolo.utils import callbacks
from slmdptyu.ultralytics.ultralytics.yolo.utils.checks import check_imshow
from slmdptyu.ultralytics.ultralytics.yolo.utils.files import increment_path
from slmdptyu.ultralytics.ultralytics.yolo.utils.torch_utils import select_device, smart_inference_mode
from slmdptyu.solomon.ultralytics.yolo.utils import DEFAULT_CFG
from slmdptyu.ultralytics.ultralytics.yolo.engine.predictor import BasePredictor
from slmdptyu.solomon.ultralytics.yolo.utils.plotting import plot

class BaseSolPredictor(BasePredictor):

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initializes the BasePredictor class.

        Args:
            cfg (str, optional): Path to a configuration file. Defaults to DEFAULT_CFG.
            overrides (dict, optional): Configuration overrides. Defaults to None.
        """
        self.args = get_cfg(cfg, overrides)
        self.save_dir = Path(self.args.save_dir) / "pred_results"
        if self.args.conf is None:
            self.args.conf = 0.25  # default conf=0.25
        self.done_warmup = False
        if self.args.show:
            self.args.show = check_imshow(warn=True)

        # Usable if setup is done
        self.model = None
        self.data = self.args.data  # data_dict
        self.imgsz = None
        self.device = None
        self.dataset = None
        self.vid_path, self.vid_writer = None, None
        self.plotted_img = None
        self.annotator = None
        self.data_path = None
        self.source_type = None
        self.batch = None
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        callbacks.add_integration_callbacks(self)

    def write_results(self, idx, results, batch):
        p, im, _ = batch
        log_string = ''
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        if self.source_type.webcam or self.source_type.from_img:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)
        self.data_path = p
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # print string
        result = results[idx]
        log_string += result.verbose()

        if self.args.save or self.args.show:  # Add bbox to image
            plot_args = dict(line_width=self.args.line_thickness, boxes=self.args.boxes)
            if not self.args.retina_masks:
                plot_args['im_gpu'] = im[idx]
            boxes = result.boxes.data.cpu().numpy()
            result_json = {
                'rois': boxes[:, :4],
                'class_ids': boxes[:, 5],
                'scores': boxes[:, 4]
            }
            if self.args.train_mask:
                masks = [
                    mask[box[1]:box[3], box[0]:box[2]]
                    for box, mask in zip(
                        boxes.round().astype(int)[:, :4], 
                        [] if result.masks is None else result.masks.data.cpu().numpy()
                    )
                ]
                result_json['masks'] = masks
            if self.args.train_keypoints:
                if result.keypoints is None:
                    result_json['keypoints'] = np.zeros((0, 0, 2))
                else:
                    result_json['keypoints'] = result.keypoints[..., :2].cpu().numpy()
            self.plotted_img = plot(
                image=result.orig_img, 
                results=result_json, 
                class_names=list(result.names.values()), 
                plot_cfg={
                    'show_bboxes': True,
                    'show_masks':  self.args.train_mask,
                    'show_kpts':   self.args.train_keypoints,
                    'show_scores': True,
                    'show_label':  False
                }
            )
        # write
        if self.args.save_txt:
            result.save_txt(f'{self.txt_path}.txt', save_conf=self.args.save_conf)
        if self.args.save_crop:
            result.save_crop(save_dir=self.save_dir / 'crops', file_name=self.data_path.stem)

        return log_string

    @smart_inference_mode()
    def stream_inference(self, source=None, model=None):
        if self.args.verbose:
            LOGGER.info('')

        # setup model
        if not self.model:
            self.setup_model(model)
        # setup source every time predict is called
        self.setup_source(source if source is not None else self.args.source)

        # check if save_dir/ label file exists
        if self.args.save or self.args.save_txt:
            (self.save_dir / 'labels' if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)
        # warmup model
        if not self.done_warmup:
            if not self.model.onnx and not self.model.tflite:
                self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs, 3, *self.imgsz))
            self.done_warmup = True

        self.seen, self.windows, self.dt, self.batch = 0, [], (ops.Profile(), ops.Profile(), ops.Profile()), None
        self.run_callbacks('on_predict_start')
        for batch in self.dataset:
            self.run_callbacks('on_predict_batch_start')
            self.batch = batch
            path, im, im0s, vid_cap, s = batch
            visualize = increment_path(self.save_dir / Path(path).stem, mkdir=True) if self.args.visualize else False

            # preprocess
            with self.dt[0]:
                im = self.preprocess(im)
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # inference
            with self.dt[1]:
                preds = self.model(
                    im, 
                    augment=self.args.augment, 
                    visualize=visualize, 
                    topk=self.args.max_det, 
                    iou_thr=self.args.iou, 
                    conf_thr=self.args.conf
                )

            # postprocess
            with self.dt[2]:
                if self.model.onnx or self.model.xml:
                    if self.args.format == 'onnx':
                        self.results = self.postprocess(preds, im, im0s)
                    elif self.args.format == 'onnx_e2e' or self.args.format == 'openvino':
                        self.results = self.assign_results_onnx(preds, im, im0s)
                elif self.model.tflite:
                    self.results = self.assign_results_tflite(preds, im, im0s)
                elif self.model.engine:
                    self.results = self.assign_results_trt(preds, im, im0s, engine = self.model.engine)
                else:
                    self.results = self.postprocess(preds, im, im0s)
            self.run_callbacks('on_predict_postprocess_end')

            # visualize, save, write results
            n = len(im)
            for i in range(n):
                self.results[i].speed = {
                    'preprocess': self.dt[0].dt * 1E3 / n,
                    'inference': self.dt[1].dt * 1E3 / n,
                    'postprocess': self.dt[2].dt * 1E3 / n}
                if self.source_type.tensor:  # skip write, show and plot operations if input is raw tensor
                    continue
                p, im0 = (path[i], im0s[i].copy()) if self.source_type.webcam or self.source_type.from_img \
                    else (path, im0s.copy())
                p = Path(p)

                if self.args.verbose or self.args.save or self.args.save_txt or self.args.show:
                    s += self.write_results(i, self.results, (p, im, im0))

                if self.args.save and self.plotted_img is not None:
                    self.save_preds(vid_cap, i, str(self.save_dir / p.name))
            self.run_callbacks('on_predict_batch_end')
            yield from self.results

            # Print time (inference-only)
            if self.args.verbose:
                LOGGER.info(f'{s}{self.dt[0].dt * 1E3:.1f}ms preprocess, {self.dt[1].dt * 1E3:.1f}ms inference, {self.dt[2].dt * 1E3:.1f}ms postprocess')

        # Release assets
        if isinstance(self.vid_writer[-1], cv2.VideoWriter):
            self.vid_writer[-1].release()  # release final video writer

        # Print results
        if self.args.verbose and self.seen:
            t = tuple(x.t / self.seen * 1E3 for x in self.dt)  # speeds per image
            LOGGER.info(f'Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess per image at shape '
                        f'{(1, 3, *self.imgsz)}' % t)
        if self.args.save or self.args.save_txt or self.args.save_crop:
            nl = len(list(self.save_dir.glob('labels/*.txt')))  # number of labels
            s = f"\n{nl} label{'s' * (nl > 1)} saved to {self.save_dir / 'labels'}" if self.args.save_txt else ''
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}{s}")

        self.run_callbacks('on_predict_end')

    def setup_model(self, model, verbose=True):
        device = select_device(self.args.device, verbose=verbose)
        model = model or self.args.model
        self.args.half &= device.type != 'cpu'  # half precision only supported on CUDA
        self.model = AutoBackend(model,
                                 device=device,
                                 dnn=self.args.dnn,
                                 data=self.args.data,
                                 fp16=self.args.half,
                                 fuse=True,
                                 verbose=verbose)
        self.device = device
        self.model.eval()
