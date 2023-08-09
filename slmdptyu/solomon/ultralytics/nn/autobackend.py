import ast
import contextlib
import zipfile
from pathlib import Path
from urllib.parse import urlparse
import numpy as np
import torch
import torch.nn as nn

from slmdptyu.solomon.ultralytics.yolo.utils import LOGGER, yaml_load
from slmdptyu.ultralytics.ultralytics.yolo.utils.checks import check_suffix
from slmdptyu.ultralytics.ultralytics.yolo.utils.downloads import is_url
from slmdptyu.ultralytics.ultralytics.nn.autobackend import  check_class_names


class AutoBackend(nn.Module):

    def __init__(self, 
                 weights='yolov8n.pt', 
                 device=torch.device('cpu'), 
                 dnn=False, 
                 data=None, 
                 fp16=False, 
                 fuse=True,
                 verbose=True):
        """
        MultiBackend class for python inference on various platforms using Ultralytics YOLO.

        Args:
            weights (str): The path to the weights file. Default: 'yolov8n.pt'
            device (torch.device): The device to run the model on.
            dnn (bool): Use OpenCV's DNN module for inference if True, defaults to False.
            data (str), (Path): Additional data.yaml file for class names, optional
            fp16 (bool): If True, use half precision. Default: False
            fuse (bool): Whether to fuse the model or not. Default: True
            verbose (bool): Whether to run in verbose mode or not. Default: True

        Supported formats and their naming conventions:
            | Format                | Suffix           |
            |-----------------------|------------------|
            | PyTorch               | *.pt             |
            | TorchScript           | *.torchscript    |
            | ONNX Runtime          | *.onnx           |
            | ONNX OpenCV DNN       | *.onnx dnn=True  |
            | OpenVINO              | *.xml            |
            | CoreML                | *.mlmodel        |
            | TensorRT              | *.engine         |
            | TensorFlow SavedModel | *_saved_model    |
            | TensorFlow GraphDef   | *.pb             |
            | TensorFlow Lite       | *.tflite         |
            | TensorFlow Edge TPU   | *_edgetpu.tflite |
            | PaddlePaddle          | *_paddle_model   |
        """
        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)
        nn_module = isinstance(weights, torch.nn.Module)
        pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle, _, triton = self._model_type(w)
        fp16 &= pt or jit or onnx or engine or nn_module  # FP16
        nhwc = coreml or saved_model or pb or tflite or edgetpu  # BHWC formats (vs torch BCWH)
        stride = 32  # default stride
        model, metadata = None, None
        model = None  # TODO: resolves ONNX inference, verify effect on other backends
        cuda = torch.cuda.is_available() and device.type != 'cpu'  # use CUDA

        # NOTE: special case: in-memory pytorch model
        if nn_module:
            model = weights.to(device)
            model = model.fuse(verbose=verbose) if fuse else model
            if hasattr(model, 'kpt_shape'):
                kpt_shape = model.kpt_shape  # pose-only
            stride = max(int(model.stride.max()), 32)  # model stride
            names = model.module.names if hasattr(model, 'module') else model.names  # get class names
            model.half() if fp16 else model.float()
            self.model = model  # explicitly assign for to(), cpu(), cuda(), half()
            pt = True
        elif pt:  # PyTorch
            from slmdptyu.solomon.ultralytics.nn.tasks import attempt_load_weights
            model = attempt_load_weights(weights if isinstance(weights, list) else w,
                                         device=device,
                                         inplace=True,
                                         fuse=fuse)
            if hasattr(model, 'kpt_shape'):
                kpt_shape = model.kpt_shape  # pose-only
            stride = max(int(model.stride.max()), 32)  # model stride
            names = model.module.names if hasattr(model, 'module') else model.names  # get class names
            model.half() if fp16 else model.float()
            self.model = model  # explicitly assign for to(), cpu(), cuda(), half()
        elif onnx:  # ONNX Runtime
            LOGGER.info(f'Loading {w} for ONNX Runtime inference...')
            import onnxruntime
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
            session = onnxruntime.InferenceSession(w, providers=providers)
            output_names = [x.name for x in session.get_outputs()]
            metadata = session.get_modelmeta().custom_metadata_map  # metadata
        elif tflite:  # https://www.tensorflow.org/lite/guide/python#install_tensorflow_lite_for_python
            import tensorflow as tf
            Interpreter, load_delegate = tf.lite.Interpreter, tf.lite.experimental.load_delegate
            interpreter = Interpreter(model_path=w)

            interpreter.allocate_tensors()  # allocate
            input_details = interpreter.get_input_details()  # inputs
            output_details = interpreter.get_output_details()  # outputs
            # load metadata
            with contextlib.suppress(zipfile.BadZipFile):
                with zipfile.ZipFile(w, 'r') as model:
                    meta_file = model.namelist()[0]
                    metadata = ast.literal_eval(model.read(meta_file).decode('utf-8'))
        elif xml:
            import openvino.runtime as ov
            core = ov.Core()
            ov_device = 'GPU' if 'GPU' in core.available_devices else 'AUTO'
            compiled_model = core.compile_model(w, ov_device)
            infer_request = compiled_model.create_infer_request()
        elif engine:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
            
            from slmdptyu.solomon.ultralytics.yolo.utils.trt_ops import HostDeviceMem, allocate_buffers, do_inference_v2, TRT_LOGGER, plugins, EXPLICIT_BATCH
            
            with open(w, "rb") as f:
                runtime = trt.Runtime(TRT_LOGGER) 
                engine = runtime.deserialize_cuda_engine(f.read())
                context = engine.create_execution_context()
                # assert context is not None, "[Error] It might be environment issue between pytorch and tensorrt for CUDA,CUDNN"
        else:
            from slmdptyu.ultralytics.ultralytics.yolo.engine.exporter import EXPORT_FORMATS_TABLE
            raise TypeError(f"model='{w}' is not a supported model format. "
                            'See https://docs.ultralytics.com/tasks/detection/#export for help.'
                            f'\n\n{EXPORT_FORMATS_TABLE}')

        # Load external metadata YAML
        if isinstance(metadata, (str, Path)) and Path(metadata).exists():
            metadata = yaml_load(metadata)
        if metadata:
            for k, v in metadata.items():
                if k in ('stride', 'batch'):
                    metadata[k] = int(v)
                elif k in ('imgsz', 'names', 'kpt_shape') and isinstance(v, str):
                    metadata[k] = eval(v)
            stride = metadata['stride']
            task = metadata['task']
            batch = metadata['batch']
            imgsz = metadata['imgsz']
            names = eval(metadata['class_names'])
            kpt_shape = metadata.get('kpt_shape')
        elif not (pt or triton or nn_module):
            LOGGER.warning(f"WARNING ⚠️ Metadata not found for 'model={weights}'")

        # Check names
        if 'names' not in locals():  # names missing
            names = self._apply_default_class_names(data)
        names = check_class_names(names)

        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, im, augment=False, visualize=False, topk=100, iou_thr=0.5, conf_thr=0.6):
        """
        Runs inference on the YOLOv8 MultiBackend model.

        Args:
            im (torch.Tensor): The image tensor to perform inference on.
            augment (bool): whether to perform data augmentation during inference, defaults to False
            visualize (bool): whether to visualize the output predictions, defaults to False

        Returns:
            (tuple): Tuple containing the raw output tensor, and processed output for visualization (if visualize=True)
        """
        b, ch, h, w = im.shape  # batch, channel, height, width
        _fp_type = np.float16 if self.fp16 else np.float32
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()  # to FP16
        if self.nhwc:
            im = im.permute(0, 2, 3, 1)  # torch BCHW to numpy BHWC shape(1,320,192,3)

        if self.pt or self.nn_module:  # PyTorch
            y = self.model(im, augment=augment, visualize=visualize) if augment or visualize else self.model(im)
        elif self.onnx:  # ONNX Runtime
            im = im.cpu().numpy()  # torch to numpy
            if len(self.session.get_inputs()) == 1:
                y = self.session.run(self.output_names, {self.session.get_inputs()[0].name: im})
            elif len(self.session.get_inputs()) == 4:
                y = self.session.run(
                    self.output_names, 
                    {
                        self.session.get_inputs()[0].name: im,
                        self.session.get_inputs()[1].name: np.array([topk]).astype(np.int64),
                        self.session.get_inputs()[2].name: np.array([iou_thr]).astype(_fp_type),
                        self.session.get_inputs()[3].name: np.array([conf_thr]).astype(_fp_type),
                    }
                )
            else:
                raise
        elif self.xml: # Openvino Runtime
            im = im.cpu().numpy()  # torch to numpy
            input_data_dict = {'images': im,}
            y = self.compiled_model(input_data_dict)
            y = [y[self.compiled_model.outputs[i]] for i in range(len(self.compiled_model.outputs))] 
        elif self.engine: #Tensorrt Runtime
            im = im.cpu().numpy()  # torch to numpy
            inputs, outputs, bindings, stream = self.allocate_buffers(self.engine)
            y = self.do_inference_v2(self.context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        else:  # TensorFlow (SavedModel, GraphDef, Lite, Edge TPU)
            im = im.cpu().numpy()
            if self.saved_model:  # SavedModel
                y = self.model(im, training=False) if self.keras else self.model(im)
                if not isinstance(y, list):
                    y = [y]
            elif self.pb:  # GraphDef
                y = self.frozen_func(x=self.tf.constant(im))
                if len(y) == 2 and len(self.names) == 999:  # segments and names not defined
                    ip, ib = (0, 1) if len(y[0].shape) == 4 else (1, 0)  # index of protos, boxes
                    nc = y[ib].shape[1] - y[ip].shape[3] - 4  # y = (1, 160, 160, 32), (1, 116, 8400)
                    self.names = {i: f'class{i}' for i in range(nc)}
            else:  # Lite or Edge TPU
                input = self.input_details
                int8 = input[0]['dtype'] == np.int8  # is TFLite quantized int8 model
                if int8:
                    scale, zero_point = input['quantization']
                    im = (im / scale + zero_point).astype(np.int8)  # de-scale
                for input_detail, value in zip(
                    input, [
                        im, 
                        # np.array(topk), # topk
                        # np.array(iou_thr, dtype=np.float32), # iou_thr
                        # np.array(conf_thr, dtype=np.float32) # conf_thr
                        ]
                    ):
                    self.interpreter.set_tensor(input_detail['index'], value)
                self.interpreter.invoke()
                y = []
                for output in sorted(self.output_details, key=lambda x: x['name']):
                    x = self.interpreter.get_tensor(output['index'])
                    if int8:
                        scale, zero_point = output['quantization']
                        x = (x.astype(np.float32) - zero_point) * scale  # re-scale
                    y.append(x)
            # TF segment fixes: export is reversed vs ONNX export and protos are transposed
            # if len(y) == 2 :  # segment with (det, proto) output order reversed
            #     if len(y[1].shape) != 4:
            #         y = list(reversed(y))  # should be y = (1, 116, 8400), (1, 160, 160, 32)
            #     y[1] = np.transpose(y[1], (0, 3, 1, 2))  # should be y = (1, 116, 8400), (1, 32, 160, 160)
            # elif len(y) == 3:
            #     if len(y[1].shape) != 4:
            #         y = list(reversed(y))  # should be y = (1, 116, 8400), (1, 160, 160, 32)
            #     y[1] = np.transpose(y[1], (0, 3, 1, 2))  # should be y = (1, 116, 8400), (1, 32, 160, 160)
            #     y[2] = np.transpose(y[2], (0, 4, 1, 2, 3))  # should be y = (1, 116, 8400), (1, 32, 160, 160)
            y = [
                x if isinstance(x, np.ndarray) or isinstance(x, np.int32) else x.numpy() for x in y
            ]
            # y[0][..., :4] *= [w, h, w, h]  # xywh normalized to pixels

        # for x in y:
        #     print(type(x), len(x)) if isinstance(x, (list, tuple)) else print(type(x), x.shape)  # debug shapes
        if isinstance(y, (list, tuple)):
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            return self.from_numpy(y)

    def from_numpy(self, x):
        """
         Convert a numpy array to a tensor.

         Args:
             x (np.ndarray): The array to be converted.

         Returns:
             (torch.Tensor): The converted tensor
         """
        return torch.tensor(x).to(self.device) if isinstance(x, np.ndarray) else x

    def warmup(self, imgsz=(1, 3, 640, 640)):
        """
        Warm up the model by running one forward pass with a dummy input.

        Args:
            imgsz (tuple): The shape of the dummy input tensor in the format (batch_size, channels, height, width)

        Returns:
            (None): This method runs the forward pass and don't return any value
        """
        warmup_types = self.pt, self.jit, self.onnx, self.engine, self.saved_model, self.pb, self.triton, self.nn_module
        if any(warmup_types) and (self.device.type != 'cpu' or self.triton):
            im = torch.empty(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
            for _ in range(2 if self.jit else 1):  #
                self.forward(im)  # warmup

    @staticmethod
    def _apply_default_class_names(data):
        with contextlib.suppress(Exception):
            return yaml_load(data)['names']
        return {i: f'class{i}' for i in range(999)}  # return default if above errors

    @staticmethod
    def _model_type(p='path/to/model.pt'):
        """
        This function takes a path to a model file and returns the model type

        Args:
            p: path to the model file. Defaults to path/to/model.pt
        """
        # Return model type from model path, i.e. path='path/to/model.onnx' -> type=onnx
        # types = [pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle]
        from slmdptyu.solomon.ultralytics.yolo.engine.exporter import export_formats
        sf = list(export_formats().Suffix)  # export suffixes
        if not is_url(p, check=False) and not isinstance(p, str):
            check_suffix(p, sf)  # checks
        url = urlparse(p)  # if url may be Triton inference server
        types = [s in Path(p).name for s in sf]
        types[8] &= not types[9]  # tflite &= not edgetpu
        triton = not any(types) and all([any(s in url.scheme for s in ['http', 'grpc']), url.netloc])
        return types + [triton]
    