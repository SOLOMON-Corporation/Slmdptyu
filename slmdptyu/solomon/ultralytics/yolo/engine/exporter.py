import os
import time
import warnings
from copy import deepcopy
from pathlib import Path
import numpy as np
import shutil
import torch
import base64
from io import BytesIO
from PIL import Image
from slmdptyu.ultralytics.ultralytics.nn.autobackend import check_class_names
from slmdptyu.ultralytics.ultralytics.nn.modules import C2f
from slmdptyu.solomon.ultralytics.nn.tasks import (DetectionSolModel, SegmentationSolModel,
                                                   PoseSolModel, SegPoseSolModel, Gaugev2SolModel)
from slmdptyu.ultralytics.ultralytics.yolo.cfg import get_cfg
from slmdptyu.solomon.ultralytics.yolo.utils import LINUX, LOGGER, colorstr, yaml_save
from slmdptyu.ultralytics.ultralytics.yolo.utils import callbacks
from slmdptyu.ultralytics.ultralytics.yolo.utils.checks import check_imgsz
from slmdptyu.ultralytics.ultralytics.yolo.utils.files import file_size
from slmdptyu.ultralytics.ultralytics.yolo.utils.torch_utils import get_latest_opset, select_device, smart_inference_mode

from slmdptyu.solomon.ultralytics.yolo.utils import DEFAULT_CFG
from slmdptyu.ultralytics.ultralytics.yolo.engine.exporter import try_export, Exporter
from slmdptyu.solomon.ultralytics.nn.modules import DetectSol, SegmentSol, PoseSol, SegPoseSol


def export_formats():
    # YOLOv8 export formats
    import pandas
    x = [
        ['PyTorch', '-', '.pt', True, True],
        ['TorchScript', 'torchscript', '.torchscript', True, True],
        ['ONNX', 'onnx', '.onnx', True, True],
        ['OpenVINO', 'openvino', '.xml', True, False],
        ['TensorRT', 'engine', '.engine', False, True],
        ['CoreML', 'coreml', '.mlmodel', True, False],
        ['TensorFlow SavedModel', 'saved_model', '_saved_model', True, True],
        ['TensorFlow GraphDef', 'pb', '.pb', True, True],
        ['TensorFlow Lite', 'tflite', '.tflite', True, False],
        ['TensorFlow Edge TPU', 'edgetpu', '_edgetpu.tflite', True, False],
        ['TensorFlow.js', 'tfjs', '_web_model', True, False],
        ['PaddlePaddle', 'paddle', '_paddle_model', True, True],
        ['ONNX_e2e', 'onnx_e2e', '.onnx', True, True],
    ]
    return pandas.DataFrame(x, columns=['Format', 'Argument', 'Suffix', 'CPU', 'GPU'])


class ExporterSol(Exporter):
    """
    Exporter

    A class for exporting a model.

    Attributes:
        args (SimpleNamespace): Configuration for the exporter.
        save_dir (Path): Directory to save results.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initializes the Exporter class.

        Args:
            cfg (str, optional): Path to a configuration file. Defaults to DEFAULT_CFG.
            overrides (dict, optional): Configuration overrides. Defaults to None.
        """
        self.args = get_cfg(cfg, overrides)
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        callbacks.add_integration_callbacks(self)

    @smart_inference_mode()
    def __call__(self, model=None, im=None):
        # self.run_callbacks('on_export_start')
        t = time.time()
        format = self.args.format.lower()  # to lowercase
        if format in ('tensorrt', 'trt'):  # engine aliases
            format = 'engine'
        fmts = tuple(export_formats()['Argument'][1:])  # available export formats
        flags = [x == format for x in fmts]
        if sum(flags) != 1:
            raise ValueError(f"Invalid export format='{format}'. Valid formats are {fmts}")
        jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle, onnx_e2e = flags  # export booleans

        # Load PyTorch model
        self.device = select_device('cpu' if self.args.device is None else self.args.device)
        if self.args.half and onnx and self.device.type == 'cpu':
            LOGGER.warning('WARNING ⚠️ half=True only compatible with GPU export, i.e. use device=0')
            self.args.half = False
            assert not self.args.dynamic, 'half=True not compatible with dynamic=True, i.e. use only one.'

        # Checks
        model.names = check_class_names(model.names)
        self.imgsz = check_imgsz(self.args.imgsz, stride=model.stride, min_dim=2)  # check image size
        if self.args.optimize:
            assert self.device.type == 'cpu', '--optimize not compatible with cuda devices, i.e. use --device cpu'
        if edgetpu and not LINUX:
            raise SystemError('Edge TPU export only supported on Linux. See https://coral.ai/docs/edgetpu/compiler/')

        # Input
        im = torch.zeros(self.args.batch, 3, *self.imgsz).to(self.device)

        file = Path(self.args.save_dir) / 'model_final.pth'
        if file.suffix == '.yaml':
            file = Path(file.name)

        # Update model
        model = deepcopy(model).to(self.device)
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        model.float()
        model = model.fuse()
        for k, m in model.named_modules():
            if isinstance(m, (DetectSol, SegmentSol, PoseSol, SegPoseSol)):
                m.dynamic = self.args.dynamic
                m.export = True
                m.format = self.args.format
            elif isinstance(m, C2f) and not not any((saved_model, pb, tflite, edgetpu, tfjs)):
                # EdgeTPU does not support FlexSplitV while split provides cleaner ONNX graph
                m.forward = m.forward_split

        y = None
        for _ in range(2):
            y = model(im)  # dry runs
        # if self.args.half and (engine or onnx or xml or onnx_e2e) and self.device.type != 'cpu':
        #     im, model = im.half(), model.half()  # to FP16

        # Warnings
        warnings.filterwarnings('ignore', category=torch.jit.TracerWarning)  # suppress TracerWarning
        warnings.filterwarnings('ignore', category=UserWarning)  # suppress shape prim::Constant missing ONNX warning
        warnings.filterwarnings('ignore', category=DeprecationWarning)  # suppress CoreML np.bool deprecation warning

        # Assign
        self.im = im
        self.model = model
        self.file = file
        self.output_shape = tuple(y.shape) if isinstance(y, torch.Tensor) else tuple(tuple(x.shape) for x in y)
        self.pretty_name = Path(self.model.yaml.get('yaml_file', self.file)).stem.replace('yolo', 'YOLO')
        # trained_on = f'trained on {Path(self.args.data).name}' if self.args.data else '(untrained)'
        description = f'InstanceSegV8_{"float16" if self.args.half else "float32"}_{self.args.model_size}'
        task = "box,"
        if self.args.train_mask:
            task += 'mask,'
        if self.args.train_keypoints:
            task += 'keypoint,'
        if self.args.train_angle:
            task += 'angle,'

        buffered = BytesIO()
        img = Image.open(file.parent.parent.parent / f"Images/{Path(model.img).name}").convert('RGB')
        img.thumbnail((480, 360), Image.ANTIALIAS)
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

        import version
        if self.args.gauge_mode in ["Circular", "Sector", "Linear"]:
            kit_name = 'ObjectGaugeV2'
        else:
            kit_name = 'ObjectDetectionV2'

        self.metadata = {
            'author': 'Solomon-AI',
            'kit_name': kit_name,
            'version': version.__version__,
            'task': task,
            'batch': self.args.batch,
            'img_h': self.imgsz[0],
            'img_w': self.imgsz[1],
            'class_names': list(model.names.values()),
            'kpts_names': model.kpts_names,
            'dtype': "float16" if self.args.half else "float32",
            'model_size': self.args.model_size,
            'kpts_coordinates': model.kpts_coordinates,
            'max_gauge_value': model.max_gauge_value if hasattr(model, "max_gauge_value") else None,
            'img': img_str,
            'name': description,
            'tool': 'InstanceSegV8'
        }  # model metadata
        if self.args.gauge_mode in ["Circular", "Sector", "Linear"]:
            self.metadata['gauge_mode'] = self.args.gauge_mode

        if 'pose' in task:
            self.metadata['kpt_shape'] = model.kpt_shape

        LOGGER.info(f"\n{colorstr('PyTorch:')} starting from {file} with input shape {tuple(im.shape)} BCHW and "
                    f'output shape(s) {self.output_shape} ({file_size(file):.1f} MB)')

        # Exports
        f = [''] * len(fmts)  # exported filenames
        if onnx or engine or xml or onnx_e2e:  # OpenVINO requires ONNX
            f[2], _ = self._export_onnx()
        if tflite:  # TensorFlow formats
            f[5], s_model = self._export_saved_model()
            f[7], _ = self._export_tflite(s_model, nms=False, agnostic_nms=self.args.agnostic_nms)
        # Finish
        f = [str(x) for x in f if x]  # filter out '' and None
        if any(f):
            f = str(Path(f[-1]))
            square = self.imgsz[0] == self.imgsz[1]
            s = '' if square else f"WARNING ⚠️ non-PyTorch val requires square images, 'imgsz={self.imgsz}' will not " \
                                  f"work. Use export 'imgsz={max(self.imgsz)}' if val is required."
            imgsz = self.imgsz[0] if square else str(self.imgsz)[1:-1].replace(' ', '')
            data = ''
            LOGGER.info(
                f'\nExport complete ({time.time() - t:.1f}s)'
                f"\nResults saved to {colorstr('bold', file.parent.resolve())}"
                f'\nPredict:         yolo predict task={task} model={f} imgsz={imgsz} {data}'
                f'\nValidate:        yolo val task={task} model={f} imgsz={imgsz} data={self.args.data} {s}'
                f'\nVisualize:       https://netron.app')

        # self.run_callbacks('on_export_end')
        return f  # return list of exported files/dirs

    @try_export
    def _export_onnx(self, prefix=colorstr('ONNX:')):
        import onnx  # noqa

        opset_version = self.args.opset or get_latest_opset()
        LOGGER.info(f'\n{prefix} starting export with onnx {onnx.__version__} opset {opset_version}...')
        f = str(self.file.with_suffix('.onnx'))
        if self.args.format == 'tflite' or self.args.format == 'onnx':
            f = os.path.join(os.path.dirname(f), 'model.onnx')

        if isinstance(self.model, Gaugev2SolModel):
            if self.args.format in ['engine', 'openvino', 'onnx_e2e']:
                output_names = ['valid', 'rois', 'scores', 'class_ids', 'keypoints']
            else:
                output_names = ['output0']
        if isinstance(self.model, SegPoseSolModel):
            if self.args.format in ['engine', 'openvino', 'onnx_e2e']:
                output_names = ['valid', 'rois', 'scores', 'class_ids', 'masks', 'keypoints']
            else:
                output_names = ['output0', 'output1']
        elif isinstance(self.model, PoseSolModel):
            if self.args.format in ['engine', 'openvino', 'onnx_e2e']:
                output_names = ['valid', 'rois', 'scores', 'class_ids', 'keypoints']
            else:
                output_names = ['output0']
        elif isinstance(self.model, SegmentationSolModel):
            if self.args.format in ['engine', 'openvino', 'onnx_e2e']:
                output_names = ['valid', 'rois', 'scores', 'class_ids', 'masks']
            else:
                output_names = ['output0', 'output1']
        else:  # DetectionSolModel
            if self.args.format in ['engine', 'openvino', 'onnx_e2e']:
                output_names = ['valid', 'rois', 'scores', 'class_ids']
            else:
                output_names = ['output0']

        if self.args.train_angle and (self.args.format in ['engine', 'openvino', 'onnx_e2e']):
            output_names.append('angles_class')

        if self.args.format in ['engine', 'openvino', 'onnx_e2e']:
            self.model.model[-1].w_onnx_end2end2_single = True
        dynamic = self.args.dynamic
        if dynamic:
            dynamic = {'images': {0: 'batch', 2: 'height', 3: 'width'}}  # shape(1,3,640,640)
            if isinstance(self.model, SegPoseSolModel):
                if self.args.format == 'onnx':
                    # output format: class id [N], confidence score [N], bbox [N,4], and keypoints [N,nkpts,2]
                    dynamic['valid'] = {0: 'batch'}
                    dynamic['class_ids'] = {0: 'batch'}
                    dynamic['scores'] = {0: 'batch'}
                    dynamic['rois'] = {0: 'batch'}
                    dynamic['keypoints'] = {0: 'batch'}
                    dynamic['masks'] = {0: 'batch'}
                    if self.args.train_angle:
                        dynamic['angles_class'] = {0: 'batch'}
                    dynamic.pop('images')
                else:
                    dynamic['output0'] = {0: 'batch', 1: 'anchors'}  # shape(1,25200,85)
            elif isinstance(self.model, PoseSolModel):
                if self.args.format == 'onnx':
                    # output format: class id [N], confidence score [N], bbox [N,4], and keypoints [N,nkpts,2]
                    dynamic['valid'] = {0: 'batch'}
                    dynamic['class_ids'] = {0: 'batch'}
                    dynamic['scores'] = {0: 'batch'}
                    dynamic['rois'] = {0: 'batch'}
                    dynamic['keypoints'] = {0: 'batch'}
                    if self.args.train_angle:
                        dynamic['angles_class'] = {0: 'batch'}
                    dynamic.pop('images')
                else:
                    dynamic['output0'] = {0: 'batch', 1: 'anchors'}  # shape(1,25200,85)
            elif isinstance(self.model, SegmentationSolModel):
                if self.args.format == 'onnx':
                    dynamic['valid'] = {0: 'batch'}
                    dynamic['class_ids'] = {0: 'batch'}
                    dynamic['scores'] = {0: 'batch'}
                    dynamic['rois'] = {0: 'batch'}
                    dynamic['masks'] = {0: 'batch'}
                    if self.args.train_angle:
                        dynamic['angles_class'] = {0: 'batch'}
                    dynamic.pop('images')
                else:
                    dynamic['output0'] = {0: 'batch', 1: 'anchors'}  # shape(1,25200,85)
                    dynamic['output1'] = {0: 'batch', 2: 'mask_height', 3: 'mask_width'}  # shape(1,32,160,160)
            elif isinstance(self.model, DetectionSolModel):
                if self.args.format == 'onnx':
                    dynamic['valid'] = {0: 'batch'}
                    dynamic['class_ids'] = {0: 'batch'}
                    dynamic['scores'] = {0: 'batch'}
                    dynamic['rois'] = {0: 'batch'}
                    if self.args.train_angle:
                        dynamic['angles_class'] = {0: 'batch'}
                    dynamic.pop('images')
                else:
                    dynamic['output0'] = {0: 'batch', 1: 'anchors'}  # shape(1,25200,85)

        torch.onnx.export(
            self.model.cpu() if dynamic else self.model,  # --dynamic only compatible with cpu
            self.im.cpu() if dynamic else self.im,
            f,
            verbose=False,
            opset_version=opset_version,
            do_constant_folding=True,  # WARNING: DNN inference with torch>=1.12 may require do_constant_folding=False
            input_names=['images'],
            output_names=output_names,
            dynamic_axes=dynamic or None)

        # Checks
        model_onnx = onnx.load(f)  # load onnx model
        # onnx.checker.check_model(model_onnx)  # check onnx model

        '''Dynamic value version'''
        if self.args.format == 'onnx_e2e':
            import onnx_graphsurgeon as gs
            graph = gs.import_onnx(model_onnx)
            max_output_boxes_per_class = gs.Variable('max_output_boxes_per_class', np.int64, shape=(1,))
            iou_threshold = gs.Variable('iou_threshold', np.float32, shape=(1,))
            score_threshold = gs.Variable('score_threshold', np.float32, shape=(1,))
            graph.inputs += [max_output_boxes_per_class, iou_threshold, score_threshold]
            for node in graph.nodes:
                if 'NonMaxSuppression' in node.name:
                    if self.args.format == 'engine':
                        node.inputs[2] = gs.Constant('max_output_boxes_per_class', np.array([self.args.max_det], dtype=np.int64))
                        node.inputs[3] = iou_threshold
                        node.inputs.append(score_threshold)
                    else:
                        node.inputs[2] = max_output_boxes_per_class
                        node.inputs[3] = iou_threshold
                        node.inputs.append(score_threshold)
                    node.outputs[0].name = 'nms_output'
            graph.cleanup().toposort()
            new_onnx_model = gs.export_onnx(graph)
            new_onnx_model.ir_version = 6
            onnx.save_model(new_onnx_model, f)
            model_onnx = onnx.load(f)

        '''Constance value version'''  # For Tensorrt will encounter the OOM issue.
        if self.args.format == 'engine' or self.args.format == 'openvino':
            import onnx_graphsurgeon as gs
            graph = gs.import_onnx(model_onnx)
            score_threshold = gs.Constant('score_threshold', np.array([self.args.conf], dtype=np.float32))
            for node in graph.nodes:
                if 'NonMaxSuppression' in node.name:
                    node.inputs.append(score_threshold)
                    node.inputs[2] = gs.Constant('max_output_boxes_per_class', np.array([self.args.max_det], dtype=np.int64))
            graph.cleanup().toposort()
            new_onnx_model = gs.export_onnx(graph)
            new_onnx_model.ir_version = 6
            onnx.save_model(new_onnx_model, f)
            model_onnx = onnx.load(f)

        # Simplify
        if self.args.simplify:
            try:
                import onnxsim

                LOGGER.info(f'{prefix} simplifying with onnxsim {onnxsim.__version__}...')
                # subprocess.run(f'onnxsim {f} {f}', shell=True)
                model_onnx, check = onnxsim.simplify(model_onnx)
                assert check, 'Simplified ONNX model could not be validated'
            except Exception as e:
                LOGGER.info(f'{prefix} simplifier failure: {e}')

        # Metadata
        for k, v in self.metadata.items():
            meta = model_onnx.metadata_props.add()
            meta.key, meta.value = k, str(v)

        onnx.save(model_onnx, f)
        if self.args.format == 'openvino':
            try:
                import openvino.runtime as ov  # noqa
            except Exception as e:
                print('[Warning] openvino not found or some problems.')
                print(e)
            import subprocess
            fp = 'FP16' if self.args.half else 'FP32'
            cmd = f'mo -w "{f}" -o "{os.path.dirname(f)}" --data_type {fp}'
            LOGGER.info(f"\n{prefix} running '{cmd.strip()}'")
            subprocess.run(cmd, shell=True)
        elif self.args.format == 'engine':
            import subprocess
            # This will cause some bug if cuda version in system is differenr from cuda in pytorch
            # try:
            #     import tensorrt as trt
            # except Exception as e:
            #     print(f'[Warning] tensorrt not found or some problem.')
            #     print(e)
            fp = '--fp16' if self.args.half else ''
            _outpath = os.path.join(os.path.dirname(f), os.path.basename(f)[:-4] + 'engine')
            cmd = f'C:\\Python3.8\\Trt\\trtexec.exe --onnx="{f}" --saveEngine="{_outpath}" {fp}'
            LOGGER.info(f"\n{prefix} running '{cmd.strip()}'")
            subprocess.run(cmd, shell=True)
        elif self.args.format == 'onnx_e2e':
            if self.args.half:
                import onnx
                from onnxconverter_common import float16
                model = onnx.load(f)
                model_fp16 = float16.convert_float_to_float16(model)
                onnx.save(model_fp16, f)
        return f, model_onnx

    @try_export
    def _export_saved_model(self, prefix=colorstr('TensorFlow SavedModel:')):

        # YOLOv8 TensorFlow SavedModel export)
        import onnx2tf
        import tensorflow as tf  # noqa
        from slmdptyu.solomon.ultralytics.yolo.utils.tf_ops import TF_model_w_PP

        LOGGER.info(f'\n{prefix} starting export with tensorflow {tf.__version__}...')
        f = Path(str(self.file).replace(self.file.suffix, '_saved_model'))
        if f.is_dir():
            try:
                shutil.rmtree(f)  # delete output folder
            except Exception as e:
                LOGGER.info(f'\n[ERROR saved_model] get error message: {e}.')

        # Export to ONNX
        self.args.simplify = True
        f_onnx, _ = self._export_onnx()
        # Export to TF
        int8 = '-oiqt -qt per-tensor' if self.args.int8 else ''
        cmd = f'onnx2tf -i "{f_onnx}" -o "{f}" -nuo --non_verbose {int8}'
        LOGGER.info(f"\n{prefix} running '{cmd.strip()}'")
        # subprocess.run(cmd, shell=True)
        # prevent long path
        shutil.move(f_onnx, str(Path.home() / Path(f_onnx).name))
        onnx2tf.convert(
            input_onnx_file_path=str(Path.home() / Path(f_onnx).name),
            output_folder_path=str(Path.home() / "model_final_saved_model"),
            not_use_onnxsim=True,
            non_verbose=True
        )
        shutil.move(str(Path.home() / Path(f_onnx).name), f_onnx)
        shutil.move(str(Path.home() / "model_final_saved_model"), f)

        # add post-process to model
        tf_model = TF_model_w_PP(
            f,
            nc=self.model.nc,
            nk=self.model.kpt_shape[0] if hasattr(self.model, 'kpt_shape') else 0,
            w_mask=self.args.train_mask,
            w_keypoints=self.args.train_keypoints,
            w_angle=self.args.train_angle,
            topk=self.args.max_det,
            iou_thr=self.args.iou,
            conf_thr=self.args.conf,
        )
        file = Path(f_onnx).with_suffix('.tflite')
        tf_model.export(file)
        new_file = file.parent / 'model_final.tflite'
        try:
            os.remove(new_file)
        except Exception:
            pass
        os.rename(file, new_file)
        file = new_file

        yaml_save(f / 'metadata.yaml', self.metadata)  # add metadata.yaml

        # Remove/rename TFLite models
        if self.args.int8:
            for file in f.rglob('*_dynamic_range_quant.tflite'):
                file.rename(file.with_stem(file.stem.replace('_dynamic_range_quant', '_int8')))
            for file in f.rglob('*_integer_quant_with_int16_act.tflite'):
                file.unlink()  # delete extra fp16 activation TFLite files

        # Add TFLite metadata
        self._add_tflite_metadata(file)

        # Load saved_model
        keras_model = tf.saved_model.load(f, tags=None, options=None)

        return str(f), keras_model

    @try_export
    def _export_tflite(self, keras_model, nms, agnostic_nms, prefix=colorstr('TensorFlow Lite:')):
        # YOLOv8 TensorFlow Lite export
        import tensorflow as tf  # noqa

        LOGGER.info(f'\n{prefix} starting export with tensorflow {tf.__version__}...')
        # saved_model = Path(str(self.file).replace(self.file.suffix, '_saved_model'))
        f = self.file.with_suffix('.tflite')

        return str(f), None  # noqa

    def _add_tflite_metadata(self, file):
        # Add metadata to *.tflite models per https://www.tensorflow.org/lite/models/convert/metadata
        from tflite_support import flatbuffers  # noqa
        from tflite_support import metadata as _metadata  # noqa
        from tflite_support import metadata_schema_py_generated as _metadata_fb  # noqa

        # Create model info
        model_meta = _metadata_fb.ModelMetadataT()
        model_meta.name = self.metadata['name']
        model_meta.version = self.metadata['version']
        model_meta.author = self.metadata['author']
        model_meta.description = str(self.metadata)

        # Label file
        tmp_file = Path(file).parent / 'temp_meta.txt'
        with open(tmp_file, 'w') as f:
            f.write(str(self.metadata))

        label_file = _metadata_fb.AssociatedFileT()
        label_file.name = tmp_file.name
        label_file.type = _metadata_fb.AssociatedFileType.TENSOR_AXIS_LABELS

        # Create input info
        input_meta = _metadata_fb.TensorMetadataT()
        input_meta.name = 'images'
        input_meta.description = 'images'
        input_meta.content = _metadata_fb.ContentT()
        input_meta.content.contentProperties = _metadata_fb.ImagePropertiesT()
        input_meta.content.contentProperties.colorSpace = _metadata_fb.ColorSpaceType.RGB
        input_meta.content.contentPropertiesType = _metadata_fb.ContentProperties.ImageProperties

        inputs = [input_meta]

        # Create output info
        output1 = _metadata_fb.TensorMetadataT()
        output1.name = 'valid'
        output1.description = 'valid'
        output1.associatedFiles = [label_file]
        output2 = _metadata_fb.TensorMetadataT()
        output2.name = 'rois'
        output2.description = 'rois'
        output2.associatedFiles = [label_file]
        output3 = _metadata_fb.TensorMetadataT()
        output3.name = 'scores'
        output3.description = 'scores'
        output3.associatedFiles = [label_file]
        output4 = _metadata_fb.TensorMetadataT()
        output4.name = 'class_ids'
        output4.description = 'class_ids'
        output4.associatedFiles = [label_file]
        outputs = [output1, output2, output3, output4]
        if self.args.train_mask:
            output5 = _metadata_fb.TensorMetadataT()
            output5.name = 'masks'
            output5.description = 'masks'
            output5.associatedFiles = [label_file]
            outputs.append(output5)
        if self.args.train_keypoints:
            output6 = _metadata_fb.TensorMetadataT()
            output6.name = 'keypoints'
            output6.description = 'keypoints'
            output6.associatedFiles = [label_file]
            outputs.append(output6)
        if self.args.train_angle:
            output7 = _metadata_fb.TensorMetadataT()
            output7.name = 'angles_class'
            output7.description = 'angles_class'
            output7.associatedFiles = [label_file]
            outputs.append(output7)

        # Create subgraph info
        subgraph = _metadata_fb.SubGraphMetadataT()
        subgraph.inputTensorMetadata = inputs
        subgraph.outputTensorMetadata = outputs
        model_meta.subgraphMetadata = [subgraph]

        b = flatbuffers.Builder(0)
        b.Finish(model_meta.Pack(b), _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
        metadata_buf = b.Output()

        populator = _metadata.MetadataPopulator.with_model_file(str(file))
        populator.load_metadata_buffer(metadata_buf)
        populator.load_associated_files([str(tmp_file)])
        populator.populate()
        tmp_file.unlink()
