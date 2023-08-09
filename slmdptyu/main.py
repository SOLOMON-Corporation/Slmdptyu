from pycocotools.coco import COCO
from pathlib import Path
import logging
import numpy as np
import torch
import shutil
import yaml
import sys
import os

SLMDPTYU = Path(__file__).resolve().parent
AIKIT = SLMDPTYU.parent
sys.path.insert(0, str(SLMDPTYU))
sys.path.insert(0, str(SLMDPTYU / "ultralytics"))
sys.path.insert(0, str(AIKIT))
sys.path.insert(0, str(AIKIT / "slmutils/packages/albumentations"))
sys.path.insert(0, str(AIKIT / "slmutils/packages/qudida"))
try:
    from slmutils.general import cvtCS2PyImg, print_traceback, colorstr, LOGGER, exec_environment
    from slmutils.voc_config import ObjectConfig
    from slmutils.application_tools import get_word_process
    import version
except Exception:
    raise ImportError


class ObjectDetectionV2:
    @print_traceback
    def __init__(
        self,
        arg_mode: str,
        arg_config_path: str,
        root_model: str = '',
        gpu_device=0,
    ):
        fileHandler = LOGGER.handlers[1]
        LOGGER.handlers[1] = logging.FileHandler(
            str(Path(arg_config_path).parent / "error.log")
        )
        LOGGER.handlers[1].formatter = fileHandler.formatter
        LOGGER.handlers[1].level = fileHandler.level

        self.detect_platform = None
        self.model = None

        torch.set_num_threads(1)
        LOGGER.info(colorstr("green", "bold", "=" * 20))
        LOGGER.info(colorstr("green", "bold",
                    f"number of threads:{torch.get_num_threads()}"))
        LOGGER.info(colorstr("green", "bold", "=" * 20))

        # VOC config
        self.config = ObjectConfig(arg_config_path)
        # load config from voc_config.json
        self.config.load_config_from_file()

        self.modes = [mode.lower() for mode in arg_mode.split(',')]
        #  If arg_mode is invalid, use train.json attributes to set modes.
        if all(mode not in self.modes for mode in ['box', 'mask', 'keypoint', 'angle']):
            self.modes = []
            if self.config.vars["has_box"]:
                self.modes.append('box')
            if self.config.vars["has_mask"]:
                self.modes.append('mask')
            if self.config.vars["has_keypoint"]:
                self.modes.append('keypoint')
            if self.config.vars["has_angle"]:
                self.modes.append('angle')

        # get global vars
        model_size = self.config.vars["modelsiz"].lower()
        if model_size == 'ultrapro':
            model_size = 'x'
        elif model_size == 'pro':
            model_size = 'l'
        elif model_size == 'default':
            model_size = 'm'
        elif model_size == 'light':
            model_size = 's'
        elif model_size == 'ultralight':
            model_size = 'n'

        if root_model == "":
            model_path = AIKIT / 'data/Models'
            if 'keypoint' in self.modes:
                model_path /= 'InstanceKeypoint5_meta'
            elif 'mask' in self.modes:
                model_path /= 'InstanceSegment5_meta'
            elif 'gaugev2' in self.modes:
                model_path /= 'InstanceKeypoint5_meta'
            else:
                model_path /= 'FeatureDetect5_meta'

            if model_size == 'x':
                model_path /= 'net_ultraPro'
            if model_size == 'l':
                model_path /= 'net_pro'
            if model_size == 'm':
                model_path /= 'net1'
            if model_size == 's':
                model_path /= 'net_light'
            if model_size == 'n':
                model_path /= 'net_ultraLight'

            model_path /= 'weights_predict.caffemodel.pth'
        else:
            model_path = root_model
        # add additional vars
        additional_vars = {
            "aikit_version": version.__version__,
            "Mode": self.modes,
            "model_size": model_size,
            "gpu_device": gpu_device,
            "root_model": str(model_path).replace("\\", "/"),
        }
        self.config.setup_vars(**additional_vars)
        # display it
        self.config.display_pretty()
        LOGGER.info(colorstr("green", "bold",
                    "Successfully initialize VOC config"))

        # In AI Server or run in python, it need to create a path to Model
        if self.config.vars.get("selected_model", False) and exec_environment == 0:
            Model_folder = os.path.join(
                self.config.vars["root_path"], "Models")
            if not os.path.exists(Model_folder):
                os.mkdir(Model_folder)
            Work_dir = os.path.join(
                Model_folder, f"{self.config.vars['selected_model']}")
            if not os.path.exists(Work_dir):
                os.mkdir(Work_dir)

        # Model
        model_file = os.path.join(
            self.config.vars['model_path'], "model_final.pth")
        if os.path.exists(model_file) and os.path.getsize(model_file) > 512:
            self.weights = model_file
        elif os.path.exists(Path(model_file).with_suffix('.pt')):  # .pt compatible
            self.weights = model_file.replace('.pth', '.pt')
        else:
            self.weights = self.config.vars['root_model']

        # YOLO Config
        self.initail_yolocfg()
        LOGGER.info(colorstr("green", "bold",
                    "Successfully initialize YOLO config"))

        # Overwrite yolocfg
        yaml_name = "yolov8" + model_size
        self.yolocfg['model_size'] = model_size

        if 'keypoint' in self.modes and 'mask' in self.modes:
            yaml_name += "-segpose"
        elif 'keypoint' in self.modes:
            yaml_name += "-pose"
        elif 'mask' in self.modes:
            yaml_name += "-seg"
        elif 'gaugev2' in self.modes:
            yaml_name += "-gaugeV2"

        self.yaml_path = SLMDPTYU / \
            f"solomon/ultralytics/models/v8/{yaml_name}.yaml"

    def initail_yolocfg(self):
        # Mode
        train_mask = True if 'mask' in self.modes else False
        train_keypoints = True if 'keypoint' in self.modes or 'gaugev2' in self.modes else False
        train_angle = True if 'angle' in self.modes else False \
            or any([eval(i) for i in self.config.vars["is_use_angle"]])

        # Model
        lr0 = self.config.vars["base_lr"]
        imgsz = max(
            self.config.vars['min_dimension'],
            self.config.vars['max_dimension']
        )
        device = self.config.vars["gpu_device"]
        batch_size = self.config.vars.get("batch_size", 1)
        # multi_scale = bool(self.config.vars.get('multiscale'))
        # assert multi_scale is False, "multi_scale is not support yet"
        max_detections = self.config.vars['max_detections']
        assert max_detections <= 500, "max_detections should below 500"
        val = bool(self.config.vars["auto_optimization"])
        epochs = self.config.vars['max_iter']
        eval_fscore = self.config.vars["eval_fscore"]
        eval_iou = self.config.vars["eval_iou"]
        eval_step = self.config.vars["eval_step"]
        eval_step = max(epochs // 10 if eval_step == 0 else eval_step, 1)
        save_dir = self.config.vars["model_path"]
        plots = False if exec_environment else True

        # Dataset
        doflip = self.config.vars.get('doflip', "False,,0.5")
        dozoom = self.config.vars.get('dozoom', "False,,,0.5")
        doshift = self.config.vars.get('doshift', "False,,,0.5")
        donoise = self.config.vars.get("donoise", "False,,,0.5")
        dorotate = self.config.vars.get('dorotate', "False,,0.5")
        docopypaste = self.config.vars.get('docopypaste', "False,0.5")
        domultiview = self.config.vars.get("domultiview", "False,0.5")
        doaddshadow = self.config.vars.get("doaddshadow", "False,,0.5")
        dohistogram = self.config.vars.get("dohistogram", "False,,0.5")
        doresizelow = self.config.vars.get("doresizelow", "False,,0.5")
        docolorjitter = self.config.vars.get("docolorjitter", "False,,,,,0.5")

        FLIP_ON, flip_type, flip_prob = doflip.split(',')
        flipud, fliplr = 0.0, 0.0
        if FLIP_ON == 'True':
            if flip_type == 'both':
                flipud, fliplr = float(flip_prob) / 2, float(flip_prob) / 2
            elif flip_type == 'horizontal':
                flipud, fliplr = 0.0, float(flip_prob)
            elif flip_type == 'vertical':
                flipud, fliplr = float(flip_prob), 0.0

        ROTATE_ON, rot_degree, p = dorotate.split(',')
        degrees, rotate_prob = [0.0, 0.0], float(p)
        if ROTATE_ON == 'True':
            degrees = [float(d) for d in rot_degree.strip('()').split(' ')]

        SHIFT_ON, shift_type, shift, p = doshift.split(',')
        translate_x, translate_y = [0.5, 0.5], [0.5, 0.5]
        translate_prob = float(p)
        if SHIFT_ON == 'True':
            shift = [0.5 + float(t) for t in shift.strip('()').split(' ')]
            if shift_type == "horizontal":
                translate_x = shift
            elif shift_type == "vertical":
                translate_y = shift
            else:
                translate_x = translate_y = shift

        ZOOM_ON, zoom_type, s, p = dozoom.split(',')
        scale, scale_prob = [1, 1], float(p)
        if ZOOM_ON == 'True':
            zoom = float("(0 0.1)".strip('()').split(' ')[1])
            if zoom_type == "zoomout":
                scale[0] = 1 - zoom
            elif zoom_type == "zoomin":
                scale[1] = 1 + zoom
            else:
                scale = [1 - zoom, 1 + zoom]

        COPYPASTE_ON, p = docopypaste.split(',')
        copypaste_prob = float(p) if COPYPASTE_ON == 'True' else 0.0

        ADDSHADOW_ON, _, p = doaddshadow.split(',')
        shadow_prob = float(p) if ADDSHADOW_ON == 'True' else 0.0

        HISTOGRAM_ON, mode, p = dohistogram.split(',')
        hist_prob, CLAHE_prob = 0.0, 0.0
        if HISTOGRAM_ON == 'True':
            CLAHE_prob = float(p) if mode == "clahe" else 0.0
            hist_prob = float(p) if mode != "clahe" else 0.0

        RESIZELOW_ON, sizelow_range, p = doresizelow.split(',')
        sizelow_prob = float(p) if eval(RESIZELOW_ON) else 0.0
        blur_limit = []
        for i in sizelow_range.strip("()").split()[::-1]:
            i = round((1 - float(i)) * 33)
            if i % 2 != 0:
                blur_limit.append(i)
            else:
                blur_limit.append(i + 1)

        COLORJITTER_ON, b, c, s, h, p = docolorjitter.split(',')
        brightness = contrast = saturation = [1.0, 1.0]
        hue = [0.0, 0.0]
        cj_prob = 0.0
        if COLORJITTER_ON == 'True':
            brightness = [float(i) for i in b.strip("()").split()]
            contrast = [float(i) for i in c.strip("()").split()]
            saturation = [float(i) for i in s.strip("()").split()]
            hue = [float(i) for i in h.strip("()").split()]
            cj_prob = float(p)

        NOISE_ON, G, SP, S, p = donoise.split(',')
        noise_prob, noise_limit = 0.0, [0.0, 0.0]
        if NOISE_ON == 'True':
            noise_prob = float(p)
            noise_limit = [65025 * float(i)**2 for i in G.split()[1:]]

        MULTIVIEW_ON, p = domultiview.split(',')
        multiview_prob = float(p) if MULTIVIEW_ON == 'True' else 0.0

        paste_pointer = self.config.vars.get('gauge_mode', None) in [
            "Circular", "Sector"]

        train_coco = COCO(os.path.join(
            self.config.vars['dataset_folder'], 'train.json'))

        data = {
            "path": self.config.vars['root_path'],
            "train": self.config.vars['train_ann_file'],
            "val": self.config.vars['val_ann_file'],
            "names": dict(enumerate(self.config.vars['class_name'])),
            "with_angle": train_angle,
            "reflect_lines": [
                int(i) for i in self.config.vars.get('reflect_lines', '0').split(',')
            ],
            "kpts_names": None,
            "kpts_coordinates": None,
            'max_gauge_value': None,
            "img": os.path.join(self.config.vars['dataset_folder'], train_coco.imgs[0]['file_name'])
        }

        if 'keypoint' in self.modes or 'gaugev2' in self.modes or \
                self.config.vars.get('gauge_mode', None) in ["Circular", "Sector"]:
            with open(
                os.path.join(
                    self.config.vars['dataset_folder'],
                    'Annotation/class_name_keypoint.txt'
                )
            ) as f:
                kpts_names = f.readlines()
            kpts_names = [''.join(name.strip().split(' ')[:-3])
                          for name in kpts_names]
            self.config.vars['keypoint_names'] = kpts_names
            data["kpts_names"] = kpts_names
            nkeypoints = self.config.vars.get("num_keypoints", 0)
            data["kpt_shape"] = [nkeypoints, 3]
            data["flip_idx"] = list(range(nkeypoints))

        if 'gaugev2' in self.modes:
            pointer_names, scaler_names, pointer_idx, npointer, nscaler = [], [], [], 0, 0
            for name in kpts_names:
                if 'pointer' in name.lower():
                    pointer_names.append(name)
                    npointer += 1
                    pointer_idx.append(True)
                else:
                    scaler_names.append(name)
                    nscaler += 1
                    pointer_idx.append(False)
            data['pointer_shape'] = [npointer, 3]
            data['scaler_shape'] = [nscaler, 3]
            data["pointer_names"] = pointer_names
            data["scaler_names"] = pointer_names
            data["pointer_idx"] = pointer_idx

        if 'gauge_mode' in self.config.vars.keys():
            gauge_cat_ids = np.where(
                ['gauge' in cat['name'].lower() for i, cat in train_coco.cats.items()])[0]
            annotation_ids = train_coco.getAnnIds(train_coco.imgs[0]['id'])
            cat_ids = np.array(
                [train_coco.anns[anno_id]["category_id"] - 1 for anno_id in annotation_ids])
            gauge_ind = np.where(
                [cat_id in gauge_cat_ids for cat_id in cat_ids])[0]
            boxes = np.array([train_coco.anns[anno_id]["bbox"]
                              for anno_id in annotation_ids])
            gauge_box = boxes[gauge_ind[0]]
            box_centers = np.concatenate([
                boxes[:, [0]] + boxes[:, [2]] / 2,
                boxes[:, [1]] + boxes[:, [3]] / 2
            ], 1)
            in_gauge = \
                (gauge_box[0] < box_centers[:, 0]) & \
                (box_centers[:, 0] < (gauge_box[0] + gauge_box[2])) & \
                (gauge_box[1] < box_centers[:, 1]) & \
                (box_centers[:, 1] < (gauge_box[1] + gauge_box[3]))
            self.kpts_coordinates = np.zeros(
                (len(self.config.vars['class_name']), 2))
            self.kpts_coordinates[cat_ids[in_gauge]] = box_centers[in_gauge]
            data['kpts_coordinates'] = self.kpts_coordinates.round().astype(
                np.int64).flatten().tolist()

            if self.config.vars.get('gauge_mode', None) == "Circular":
                data['max_gauge_value'] = self.config.vars.get(
                    "max_gauge_value", None)

        data_file = str(
            SLMDPTYU / "ultralytics/ultralytics/datasets/custom.yaml")
        with open(data_file, 'w') as f:
            yaml.safe_dump(data, f)

        if 'keypoint' in self.modes and 'mask' in self.modes:
            self.task = 'segpose'
        elif 'keypoint' in self.modes:
            self.task = 'pose'
        elif 'mask' in self.modes:
            self.task = 'segment'
        elif 'gaugev2' in self.modes:
            self.task = 'gaugev2'
        else:
            self.task = 'detect'

        self.yolocfg = {
            'task': self.task,
            "model": self.weights,       # path to model file, i.e. yolov8n.pt, yolov8n.yaml
            "data": data_file,           # path to data file, i.e. i.e. coco128.yaml
            "epochs": epochs,            # number of epochs to train for
            # epochs to wait for no observable improvement for early stopping of training
            "patience": 0,
            # number of images per batch (-1 for AutoBatch)
            "batch": batch_size,
            "imgsz": imgsz,              # size of input images as integer or w,h
            "save": True,                # save train checkpoints and predict results
            # Save checkpoint every x epochs (disabled if < 1)
            "save_period": -1,
            "cache": False,              # FIXME True/ram, disk or False. Use cache for data loading
            # device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu
            "device": device,
            # number of worker threads for data loading (per RANK if DDP)
            "workers": 0,
            "project": None,             # project name
            "name": None,                # experiment name
            "save_dir": save_dir,
            "exist_ok": False,           # whether to overwrite existing experiment
            "pretrained": False,         # whether to use a pretrained model
            # optimizer to use, choices=['SGD', 'Adam', 'AdamW', 'RMSProp']
            "optimizer": "SGD",
            "verbose": True,            # whether to print verbose output
            "seed": 0,                   # random seed for reproducibility
            "deterministic": True,       # whether to enable deterministic mode
            "single_cls": False,         # train multi-class data as single-class
            "image_weights": False,      # use weighted image selection for training
            "rect": False,               # support rectangular training
            "cos_lr": False,             # use cosine learning rate scheduler
            "close_mosaic": 0,           # disable mosaic augmentation for final 10 epochs
            "resume": False,             # resume training from last checkpoint
            "amp": True,
            # Segmentation
            # masks should overlap during training (segment train only)
            "overlap_mask": False,
            # mask downsample ratio (segment train only)
            "mask_ratio": 4,
            # Classification
            # use dropout regularization (classify train only)
            "dropout": False,
            "eval_fscore": eval_fscore,
            "eval_iou": eval_iou,
            "eval_step": eval_step,
            "train_mask": train_mask,
            "train_keypoints": train_keypoints,
            "train_angle": train_angle,
            "save_metrics": False,
            "save_callbacks": False,
            "model_size": None,

            # Val/Test settings
            "val": val,                  # validate/test during training
            # dataset split to use for validation, i.e. 'val', 'test' or 'train'
            "split": "val",
            "save_json": True,           # save results to JSON file
            # save hybrid version of labels (labels + additional predictions)
            "save_hybrid": False,
            # object confidence threshold for detection (default 0.25 predict, 0.001 val)
            "conf": 0.05,
            # intersection over union (IoU) threshold for NMS
            "iou": 0.7,
            "max_det": max_detections,   # maximum number of detections per image
            "half": False,               # use half precision (FP16)
            "dnn": False,                # use OpenCV DNN for ONNX inference
            "plots": plots,              # save plots during train/val

            # Prediction settings
            "source": None,              # source directory for images or videos
            "show": False,               # show results if possible
            "save_txt": False,           # save results as .txt file
            "save_conf": False,          # save results with confidence scores
            "save_crop": False,          # save cropped images with results
            "show_labels": True,         # show object labels in plots
            "show_conf": True,           # show object confidence scores in plots
            "vid_stride": False,         # video frame-rate stride
            "line_thickness": 3,         # bounding box thickness (pixels)
            "visualize": False,          # visualize model features
            "augment": False,            # apply image augmentation to prediction sources
            "agnostic_nms": False,       # class-agnostic NMS
            # filter results by class, i.e. class=0, or class=[0,2,3]
            "classes": None,
            "retina_masks": True,        # use high-resolution segmentation masks
            "boxes": True,               # Show boxes in segmentation predictions

            # Export settings
            "format": "torchscript",     # format to export to
            "keras": False,              # use Keras
            "optimize": False,           # TorchScript: optimize for mobile
            "int8": False,               # CoreML/TF INT8 quantization
            "dynamic": False,            # ONNX/TF/TensorRT: dynamic axes
            "simplify": True,            # ONNX: simplify model
            "opset": None,               # ONNX: opset version (optional)
            "workspace": 4,              # TensorRT: workspace size (GB)
            "nms": False,                # CoreML: add NMS

            # Hyperparameters
            # initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
            "lr0": lr0,
            "lrf": 0.01,                 # final learning rate (lr0 * lrf)
            "momentum": 0.937,           # SGD momentum/Adam beta1
            "weight_decay": 0.001,       # optimizer weight decay 5e-4
            "warmup_epochs": 3.0,        # warmup epochs (fractions ok)
            "warmup_momentum": 0.8,      # warmup initial momentum
            "warmup_bias_lr": 0.1,       # warmup initial bias lr
            "box": 7.5,                  # box loss gain
            "cls": 0.5,                  # cls loss gain (scale with pixels)
            "dfl": 1.5,                  # dfl loss gain
            "pose": 12.0,                # pose loss gain
            "kobj": 1.0,                 # keypoint obj loss gain
            "ang": 0.1,                  # angle loss gain
            "label_smoothing": 0.0,      # label smoothing (fraction)
            "nbs": 64,                   # nominal batch size

            # Solvision pipeline hyp
            'shadow': shadow_prob,
            'equalize': hist_prob,
            'CLAHE': CLAHE_prob,
            'blur': sizelow_prob,
            'blur_limit': blur_limit,
            'color_jitter': cj_prob,
            'brightness': brightness,
            'contrast': contrast,
            'saturation': saturation,
            'hue': hue,
            'noise': noise_prob,
            'noise_limit': noise_limit,
            'rotate_p': rotate_prob,
            'degrees_sol': degrees,
            'translate_p': translate_prob,
            'translate_x': translate_x,
            'translate_y': translate_y,
            'scale_p': scale_prob,
            'scale_sol': scale,
            'flipud': flipud,
            'fliplr': fliplr,
            'copy_paste': copypaste_prob,
            'multiview': multiview_prob,
            'paste_pointer': paste_pointer,

            # Special hyp
            'gauge_mode': self.config.vars.get('gauge_mode', '')

        }

    def init_model(self, model):
        from slmdptyu.solomon.ultralytics.yolo.engine.model import YOLOSOL
        self.model = YOLOSOL(model=model, task=self.task)
        LOGGER.info(colorstr("green", "bold", "Successfully build model"))

    @print_traceback
    def train(self, use_retrain=False):
        self.init_model(model=self.yaml_path)
        self.model.train(**self.yolocfg)
        LOGGER.info(colorstr("green", "bold", "Successfully train model"))
        self.weights = os.path.join(
            self.config.vars['model_path'], "model_final.pth")
        if exec_environment == 1:
            if self.model.trainer.tfControl.NeedToSaveModel() or \
                (not self.model.trainer.tfControl.NeedToSaveModel() and
                    not self.model.trainer.tfControl.NeedToStopTrain()):
                target_model = os.path.join(
                    self.config.vars['model_path'], "model_final.pth")
                if os.path.exists(target_model) and os.path.getsize(target_model) > 512:
                    if self.config.vars["model_size"] in ['m', 'n', 's']:
                        self.export_tflite(imgsz=(288, 512))
                    # self.export_end2end(
                    #     imgsz=(self.config.vars['min_dimension'],
                    #            self.config.vars['max_dimension']),
                    #     format='onnx_e2e'
                    # )
                    self.export_end2end(
                        imgsz=(self.config.vars['min_dimension'],
                               self.config.vars['max_dimension']),
                        # format='onnx_e2e',
                        format='openvino'
                    )

                    onnx_f = self.export_onnx(
                        imgsz=(self.config.vars['min_dimension'],
                               self.config.vars['max_dimension'])
                    )
                    onnx_f = Path(onnx_f)
                    try:
                        os.remove(onnx_f.parent / "model_final.onnx")
                    except Exception:
                        pass
                    os.rename(
                        onnx_f.parent / "model.onnx",
                        onnx_f.parent / "model_final.onnx"
                    )

        torch.cuda.empty_cache()

    def val(self):
        self.init_model(model=self.weights)
        metrics = self.model.val(**self.yolocfg)
        LOGGER.info(metrics)

    @print_traceback
    def detect(
        self,
        arg_h=None,
        arg_w=None,
        arg_channel=None,
        arg_bytearray=None,
        platform=None
    ):
        yolocfg_copy = self.yolocfg.copy()
        yolocfg_copy["format"] = platform
        if self.detect_platform != platform or self.model is None:
            self.detect_platform = platform
            model = self.weights

            if platform == 'onnx':
                model = os.path.join(os.path.dirname(
                    self.weights), 'model.onnx')
            elif platform == 'onnx_e2e':
                model = Path(self.weights).with_suffix('.onnx')
            elif platform == 'tflite':
                model = Path(self.weights).with_suffix('.tflite')
                # tensorflow 2.12 not support cpu on windows
                yolocfg_copy["device"] = 'cpu'
            elif platform == 'openvino':
                model = Path(self.weights).with_suffix('.xml')
            elif platform == 'engine':
                model = Path(self.weights).with_suffix('.engine')

            self.init_model(model=model)

        source = arg_bytearray if exec_environment == 0 else cvtCS2PyImg(
            arg_h, arg_w, arg_channel, arg_bytearray
        )[:, :, ::-1]
        if exec_environment:
            yolocfg_copy['save'] = False
        yolocfg_copy['imgsz'] = (
            self.config.vars['min_dimension'], self.config.vars['max_dimension'])
        yolocfg_copy['agnostic_nms'] = True
        # Input source. Accepts image, folder, video, url
        yolocfg_copy["source"] = source
        yolocfg_copy["max_det"] = self.config.vars["max_detections"]
        yolocfg_copy["conf"] = self.config.vars["test_score_thresh"]
        yolocfg_copy["iou"] = self.config.vars["eval_iou"]
        # Prediction
        result_ = self.model.predict(**yolocfg_copy)[0]
        result = {
            "rois": result_.boxes.xyxy.cpu().numpy(),
            "class_ids": result_.boxes.cls.cpu().numpy().astype(int),
            "scores": result_.boxes.conf.cpu().numpy()
        }
        if result_.masks is not None:
            result["masks"] = result_.masks.croped()
        if result_.keypoints is not None:
            result["keypoints"] = result_.keypoints.cpu().numpy()
        if result_.angles is not None:
            result["angles"] = result_.angles.cpu().numpy()
        if 'gauge_mode' in self.config.vars.keys():
            result.pop('angles', None)
            result_copy = result.copy()
            scenario = "gauge"
            if self.config.vars['gauge_mode'] in ["Circular", "Sector"]:
                scenario = 'gauge'
            elif self.config.vars['gauge_mode'] == "Linear":
                scenario = 'line-meter'
            cfg = {
                "kpts_coordinates": self.kpts_coordinates,
                "max_value": self.config.vars.get("max_gauge_value", None)
            }
            if len(result['rois']) == 0:
                result['word'] = []
            else:
                result = get_word_process(
                    scenario=scenario,
                    class_names=np.array(self.config.vars['class_name']),
                    keypoint_names=[],
                    cfg=cfg
                )(result)
                # add other objs back
                result.pop('keypoints', None)
                gauge_ids = np.where([True if 'gauge' in name.lower(
                ) else False for name in self.config.vars['class_name']])[0]
                filter_ = np.array(
                    [i not in gauge_ids for i in result_copy['class_ids']])
                if len(result["rois"]) > 0:
                    for k in result:
                        if k == 'word':
                            result['word'] = [
                                ''] * len(result_copy['rois'][filter_]) + result['word']
                        else:
                            result[k] = np.concatenate(
                                [result_copy[k][filter_], result[k]])
                else:
                    result = result_copy

        LOGGER.info(colorstr("green", "bold", "Successfully detect image"))
        torch.cuda.empty_cache()
        return result

    @print_traceback
    def detect_onnx(
        self,
        arg_h=None,
        arg_w=None,
        arg_channel=None,
        arg_bytearray=None,
        **kwargs
    ):
        # mode = kwargs.get("mode", 1) #0:yolo-base+pose, 1:e2e
        pass

    @print_traceback
    def detect_openvino(
        self,
        arg_h=None,
        arg_w=None,
        arg_channel=None,
        arg_bytearray=None,
        **kwargs
    ):
        pass

    @print_traceback
    def detect_tensorrt(
        self,
        arg_h=None,
        arg_w=None,
        arg_channel=None,
        arg_bytearray=None,
        **kwargs
    ):
        pass

    @print_traceback
    def detect_tflite(
        self,
        arg_h=None,
        arg_w=None,
        arg_channel=None,
        arg_bytearray=None,
        **kwargs
    ):
        pass

    def export_onnx(self, imgsz=None, half=False):
        self.init_model(model=self.weights)
        self.yolocfg["format"] = 'onnx'
        if imgsz is not None:
            self.yolocfg['imgsz'] = imgsz
        self.yolocfg['agnostic_nms'] = True
        self.yolocfg['half'] = half
        self.yolocfg['dynamic'] = False
        f = self.model.export(**self.yolocfg)
        if f:
            LOGGER.info(colorstr("green", "bold",
                                 f"Successfully export model to onnx, model={f}"))
        else:
            LOGGER.info(
                colorstr("red", "bold", "Fail to export model to onnx"))
        return f

    def export_tflite(self, imgsz=(288, 512)):
        self.init_model(model=self.weights)
        self.yolocfg["format"] = 'tflite'
        if imgsz is not None:
            self.yolocfg['imgsz'] = imgsz
        self.yolocfg['agnostic_nms'] = True  # not support non agnostic yet
        self.yolocfg['dynamic'] = False
        self.yolocfg["conf"] = self.config.vars["test_score_thresh"]
        f = self.model.export(**self.yolocfg)
        shutil.rmtree(os.path.join(
            self.config.vars['model_path'], "model_final_saved_model"))
        os.remove(os.path.join(
            self.config.vars['model_path'], "model.onnx"))
        if f:
            LOGGER.info(colorstr("green", "bold",
                                 f"Successfully export model to tflite, model={f}"))
        else:
            LOGGER.info(
                colorstr("red", "bold", "Fail to export model to tflite"))
        return f

    def export_end2end(self, imgsz=640, format='engine'):
        # engine, openvino, onnx_e2e
        self.init_model(model=self.weights)
        self.yolocfg["format"] = format
        self.yolocfg['imgsz'] = imgsz
        self.yolocfg['agnostic_nms'] = True
        # self.yolocfg['half'] = False # half=True not compatible with dynamic=True
        self.yolocfg['dynamic'] = False
        self.yolocfg["conf"] = self.config.vars["test_score_thresh"]
        self.yolocfg["iou"] = self.config.vars["eval_iou"]
        f = self.model.export(**self.yolocfg)
        if f:
            LOGGER.info(colorstr("green", "bold",
                                 f"Successfully export model to onnx, model={f}"))
        else:
            LOGGER.info(
                colorstr("red", "bold", "Fail to export model to onnx"))
        return f

    def del_self(self):
        del self
        torch.cuda.empty_cache()
        LOGGER.info(colorstr("green", "bold", "Successfully del self"))


ObjectGaugeV2 = ObjectDetectionV2

if __name__ == "__main__":
    # %% test
    model_path = Path("data/Models")
    default_model = 'net1/weights_predict.caffemodel.pth'
    box_model = str(model_path / 'FeatureDetect5_meta' / default_model)
    mask_model = str(model_path / 'InstanceSegment5_meta' / default_model)
    keypoint_model = str(model_path / 'InstanceKeypoint5_meta' / default_model)

    mode = ''
    # mode = "mask"
    # mode = "keypoint"
    # mode = "mask,keypoint"

    project_folder = Path(
        'data/data_slmdptyu/test-yolov8/Feature Detection5 meta Tool1'
    )
    config_path = project_folder / "voc_config.json"

    # # Train
    model = ObjectDetectionV2(mode, config_path, box_model)
    # model.train()

    # # Export
    # model.export_tflite(imgsz=(288, 512))
    # model.export_end2end(
    #     imgsz=(
    #         model.config.vars['min_dimension'],
    #         model.config.vars['max_dimension']
    #         # ), format='onnx_e2e'
    #     ), format='openvino'
    # )
    # model.export_onnx(
    #     imgsz=(
    #         model.config.vars['min_dimension'],
    #         model.config.vars['max_dimension']
    #     )
    # )

    # Detect
    # image_path = project_folder / "Images"
    # for img in os.listdir(image_path):
    #     if img.endswith(('.jpg', '.png', '.bmp')):
    #         img = r'9_Image.png'
    #         model.detect(None, None, None, image_path / img)
    #         model.detect(None, None, None, image_path / img, platform='onnx')
    #         model.detect(None, None, None, image_path / img, platform='onnx_e2e')
    #         model.detect(None, None, None, image_path /
    #                      img, platform='openvino')
    #         break
    model.detect(None, None, None, np.zeros((640, 640, 3), dtype=np.uint8))
