import os
import sys
import cv2
import numpy as np
from pathlib import Path

SLMDPTYU = Path(__file__).resolve().parent
AIKIT = SLMDPTYU.parent
sys.path.insert(0, str(SLMDPTYU))
sys.path.insert(0, str(AIKIT))

try:
    from slmutils.ir_base import IRModel, load_or_check_image
    from slmutils.annotator import Annotator
    from slmutils.general import timer, cvtCS2PyImg
    from slmutils.application_tools import get_word_process
except Exception:
    raise ImportError


class ObjectDetectionV2IR(IRModel):
    def __init__(self, model_path, device='GPU', scenario=''):
        self.scenario = scenario
        super().__init__(model_path, device)
        print(self.__class__.__name__, self.backend)
        if self.scenario:
            self.cfg = {
                "kpts_coordinates": self.kpts_coordinates,
                "max_value": self.max_gauge_value
            }
            self.word_process = get_word_process(
                scenario=scenario,
                class_names=np.array(self.class_names),
                cfg=self.cfg
            )

    def initialize_onnx_model(self):
        super().initialize_onnx_model()
        # General variables for Object Detection.
        self.has_box = 'box' in self.task or 'boxes' in self.task
        self.has_mask = 'mask' in self.task or 'masks' in self.task
        self.has_keypoint = 'keypoint' in self.task or 'kpts' in self.task
        self.has_angle = 'angle' in self.task or 'angles' in self.task
        self.kpts_names = eval(self.metadata.get('kpts_names'))

        # Exclusive variables for Gauge Tool.
        if self.scenario:
            kpts_coordinates = eval(self.metadata.get('kpts_coordinates'))
            self.kpts_coordinates = np.array(
                kpts_coordinates, dtype=np.float64).reshape(-1, 2) if kpts_coordinates is not None else None
            self.max_gauge_value = eval(self.metadata.get('max_gauge_value'))

    def initialize_openvino_model(self):
        super().initialize_openvino_model()
        # General variables for Object Detection.
        self.has_box = 'box' in self.task or 'boxes' in self.task
        self.has_mask = 'mask' in self.task or 'masks' in self.task
        self.has_keypoint = 'keypoint' in self.task or 'kpts' in self.task
        self.has_angle = 'angle' in self.task or 'angles' in self.task
        self.kpts_names = eval(self.model.get_rt_info(['framework', 'kpts_names']))

        # Exclusive variables for Gauge Tool.
        if self.scenario:
            kpts_coordinates = eval(self.model.get_rt_info(['framework', 'kpts_coordinates']))
            self.kpts_coordinates = np.array(
                kpts_coordinates, dtype=np.float64).reshape(-1, 2) if kpts_coordinates is not None else None
            self.max_gauge_value = eval(self.model.get_rt_info(['framework', 'max_gauge_value']))

    def initialize_tensorrt_model(self):
        # TODO
        super().initialize_tensorrt_model()

    def initialize_tflite_model(self):
        super().initialize_tflite_model()
        self.has_box = True
        self.has_mask = False
        self.has_keypoint = False
        self.has_angle = False
        self.input_names = ['img']
        self.output_names = ['valid', 'rois', 'scores', 'class_ids']
        if 'mask' in self.metadata['task'] or 'masks' in self.metadata['task']:
            self.output_names.append('masks')
            self.has_mask = True
        if 'keypoint' in self.metadata['task'] or 'kpts' in self.metadata['task']:
            self.output_names.append('keypoints')
            self.has_keypoint = True
        if 'angle' in self.metadata['task'] or 'angles' in self.metadata['task']:
            self.output_names.append('angles')
            self.has_angle = True
        self.inputs = self.interpreter.get_input_details()
        self.outputs = self.interpreter.get_output_details()
        # Exclusive variables for Gauge Tool.
        if self.scenario:
            kpts_coordinates = self.metadata.get('kpts_coordinates', None)
            self.kpts_coordinates = np.array(
                kpts_coordinates, dtype=np.float64).reshape(-1, 2) if kpts_coordinates is not None else None
            self.max_gauge_value = self.metadata.get("max_gauge_value", None)

    def detect(self, x, *args):
        x = load_or_check_image(x)
        self.ori_shape = x.shape
        x = self.preprocess(x, (self.img_hw[0], self.img_hw[1]))

        if self.backend == "onnx":
            if len(args) != 3:
                max_det, iou_thresh, score_thresh = 300, 0.5, 0.5
            else:
                max_det, iou_thresh, score_thresh = args
            result = self._detect_onnx(x, max_det, iou_thresh, score_thresh)
        elif self.backend == "openvino":
            result = self._detect_openvino(x)
        elif self.backend == "tensorrt":
            result = self._detect_tensorrt(x)
        elif self.backend == "tflite":
            result = self._detect_tflite(x)

        if self.scenario:
            return self.word_process(results=result)
        else:
            return result

    def detect_csharp(self, arg_h, arg_w, arg_channel, arg_bytearray, *args):
        x = cvtCS2PyImg(arg_h, arg_w, arg_channel, arg_bytearray)
        x = x[:, :, ::-1]  # RGB to BGR
        return self.detect(x, *args)

    def _detect_onnx(self, x, max_det=300, iou_thresh=0.5, score_thresh=0.5):
        max_det = np.array([max_det], dtype=np.int64)
        iou_thresh = np.array([iou_thresh], dtype=np.float32)
        score_thresh = np.array([score_thresh], dtype=np.float32)

        if self.fp16:
            x = x.astype(np.float16)
            max_det = max_det.astype(np.float16)
            iou_thresh = iou_thresh.astype(np.float16)
            score_thresh = score_thresh.astype(np.float16)

        try:
            outputs = self.inference(x, max_det, iou_thresh, score_thresh)
        except Exception as e:
            print(e)
            print('[Logging] Onnxruntime something error! Return empty arrays.')
            outputs = [np.array([]) for _ in self.output_names]
            result = self.map_outputs_to_results(outputs)
            return result

        result = self.map_outputs_to_results(outputs)
        if len(outputs[0]) == 0:
            print('[Logging] Detect nothing! Return empty arrays.')
            return result

        result = self.postprocess(result, post_mode=1)
        return result

    def _detect_openvino(self, x):
        if self.fp16:
            x = x.astype(np.float16)

        try:
            outputs = self.inference(x)
        except Exception as e:
            print(e)
            print('[Logging] Onnxruntime something error! Return empty arrays.')
            outputs = [np.array([]) for _ in self.output_names]
            result = self.map_outputs_to_results(outputs)
            return result

        result = self.map_outputs_to_results(outputs)
        if len(outputs[0]) == 0:
            print('[Logging] Detect nothing! Return empty arrays.')
            return result

        result = self.postprocess(result, post_mode=1)
        return result

    def _detect_tensorrt(self, x):
        pass

    def _detect_tflite(self, x):
        result = self.inference(x)
        result = self.postprocess(result, post_mode=1)
        return result

    # @timer
    def preprocess(self, im, size):  # size=> HW, im=>BGR
        im = LetterBox(size, auto=False, stride=32)(image=im)
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous
        im = im / 255
        return im[None, ...]

    @timer
    def inference(self, x, *args):
        if self.backend == "onnx":
            max_det, iou_thresh, score_thresh = args
            if len(self.input_names) == 1:
                outputs = self.sess.run(self.output_names, {self.input_names[0]: x, })
            else:
                outputs = self.sess.run(
                    self.output_names,
                    {
                        self.input_names[0]: x,
                        self.input_names[1]: max_det,
                        self.input_names[2]: iou_thresh,
                        self.input_names[3]: score_thresh,
                    }
                )
        elif self.backend == "openvino":
            outputs = self.compiled_model(x)
            outputs = [outputs[self.compiled_model.outputs[i]] for i in range(len(self.compiled_model.outputs))]
        elif self.backend == "tensorrt":
            pass
        elif self.backend == "tflite":
            x = x.transpose(0, 2, 3, 1)
            self.interpreter.set_tensor(self.inputs[0]['index'], x)
            self.interpreter.invoke()
            result = {}
            for name, output in zip(self.output_names, self.outputs):
                result[name] = self.interpreter.get_tensor(output['index'])
            return result
        return outputs

    # @timer
    def postprocess(self,
                    result,
                    post_mode=1,
                    post_lib='np',  # 'torch'
                    **kwargs):
        """
        Describe the purpose or functionality of the function.

        Args:
            outputs (dict): Description of the output parameters.
                - 'rois' (type): shape (N, 4)
                - 'scores' (type): shape (N,)
                - 'class_ids' (type): : shape (N,) start from 0~
                - 'masks' (type, optional): shape (N, H, W)
                - 'keypoints' (type, optional): shape (N, NK, 3)
                - 'angles_class' (type, optional): shape (N,72)

        """
        #  Filter out redundant invalid objects
        if self.backend in ['openvino', 'tflite']:
            if self.backend == 'openvino':
                valid = np.sum(result['valid'] != -1)
            elif self.backend == 'tflite':
                valid = result['valid']
            result['rois'] = result['rois'][:valid]
            result['scores'] = result['scores'][:valid]
            result['class_ids'] = result['class_ids'][:valid]
            if self.has_mask:
                result['masks'] = result['masks'][:valid]
            if self.has_keypoint:
                result['keypoints'] = result['keypoints'][:valid]
            if self.has_angle:
                result['angles_class'] = result['angles_class'][:valid]

        final_result = {}

        # [TODO]
        # post_mode == 0

        if post_lib == 'torch':
            import torch  # noqa
            import torch.nn.functional as F  # noqa

        if post_mode == 0:  # 0 => yolo-base+post
            pass
        elif post_mode == 1:  # 1=> end2end fast
            rois = result['rois']
            final_result['scores'] = scores = result['scores']
            final_result['class_ids'] = class_ids = result['class_ids']
            if self.has_mask:
                masks = result['masks']
                masks = [
                    mask[box[1]:box[3], box[0]:box[2]]
                    for box, mask in zip(
                        (rois / 4).round().astype(int),
                        masks.astype(np.float32)
                    )
                ]
            final_result['rois'] = (scale_boxes(self.img_hw, rois, self.ori_shape) + 0.5).astype(np.int32)
            if self.has_mask:
                masks = [
                    cv2.resize(mask, (box[2] - box[0], box[3] - box[1]))
                    for box, mask in zip(final_result['rois'], masks)
                ]
                final_result['masks'] = masks
            if self.has_keypoint:
                keypoints = result['keypoints'].reshape(-1, len(self.kpts_names), 3)
                keypoints = scale_coords(self.img_hw, keypoints, self.ori_shape)
                final_result['keypoints'] = keypoints
            if self.has_angle:
                final_result['angles'] = result['angles']
        elif post_mode == 2:  # 2=> end2end slow
            ori_shape = self.ori_shape
            rois = (scale_boxes(self.img_hw, result['rois'], ori_shape) + 0.5).astype(np.int32)
            scores = result['scores']
            class_ids = result['class_ids']
            final_result['rois'] = rois
            final_result['scores'] = scores
            final_result['class_ids'] = class_ids
            if self.has_mask:
                masks = result['masks'].transpose(1, 2, 0)
                mh, mw = masks.shape[:2]
                gain = min(mh / ori_shape[0], mw / ori_shape[1])  # gain  = old / new
                pad = (mw - ori_shape[1] * gain) / 2, (mh - ori_shape[0] * gain) / 2  # wh padding
                top, left = int(pad[1]), int(pad[0])  # y, x
                bottom, right = int(mh - pad[1]), int(mw - pad[0])
                masks = masks[top:bottom, left:right]
                masks = cv2.resize(masks, (ori_shape[1], ori_shape[0]))
                if len(masks.shape) == 2:
                    masks = masks[None]
                else:
                    masks = masks.transpose(2, 0, 1)

                crop_masks = []
                for bbox, mask in zip(rois, masks):
                    crop_mask = mask[bbox[1]:bbox[3], bbox[0]:bbox[2]].copy()
                    crop_masks.append(crop_mask)

                final_result['masks'] = crop_masks
            if self.has_keypoint:
                keypoints = result['keypoints'].reshape(-1, len(self.kpts_names), 3)[..., :2]
                keypoints = scale_coords(self.img_hw, keypoints, ori_shape)
                final_result['keypoints'] = keypoints
            if self.has_angle:
                pass
        return final_result

    def map_outputs_to_results(self, outputs):
        result = {}
        output_mapping = {
            'valid': 'valid',
            'bboxes': 'rois',
            'rois': 'rois',
            'scores': 'scores',
            'classes': 'class_ids',
            'classids': 'class_ids',  # legacy
            'class_ids': 'class_ids',
            'masks': 'masks',
            'keypoints': 'keypoints',
            'kpts': 'keypoints',
            'angles_class': 'angles_class'
        }

        for i, name in enumerate(self.output_names):
            if name in output_mapping:
                result[output_mapping[name]] = outputs[i]

        return result

    def draw_detections(self, img_path, result, out_path):
        annotator = Annotator(show_bboxes=True,
                              show_masks=self.has_mask,
                              show_kpts=self.has_keypoint,
                              show_scores=True,
                              show_label=True if self.scenario else False,
                              class_names=self.class_names,
                              colors=None,
                              )
        img = load_or_check_image(img_path)
        result_img = annotator(img, result)
        cv2.imwrite(os.path.join(out_path, os.path.basename(img_path)), result_img)


ObjectGaugeV2IR = ObjectDetectionV2IR


# Preprocessing functions
class LetterBox:
    """Resize image and padding for detection, instance segmentation, pose"""

    def __init__(self, new_shape=(640, 640), auto=False, scaleFill=False, scaleup=True, stride=32):
        self.new_shape = new_shape
        self.auto = auto
        self.scaleFill = scaleFill
        self.scaleup = scaleup
        self.stride = stride

    def __call__(self, labels=None, image=None):
        if labels is None:
            labels = {}
        img = labels.get('img') if image is None else image
        shape = img.shape[:2]  # current shape [height, width]
        new_shape = labels.pop('rect_shape', self.new_shape)
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if self.auto:  # minimum rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # wh padding
        elif self.scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2
        if labels.get('ratio_pad'):
            labels['ratio_pad'] = (labels['ratio_pad'], (dw, dh))  # for evaluation

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                 value=(114, 114, 114))  # add border

        if len(labels):
            labels = self._update_labels(labels, ratio, dw, dh)
            labels['img'] = img
            labels['resized_shape'] = new_shape
            return labels
        else:
            return img

    def _update_labels(self, labels, ratio, padw, padh):
        """Update labels"""
        labels['instances'].convert_bbox(format='xyxy')
        labels['instances'].denormalize(*labels['img'].shape[:2][::-1])
        labels['instances'].scale(*ratio)
        labels['instances'].add_padding(padw, padh)
        return labels


# Postprocessing functions
def clip_boxes(boxes, shape):
    """
    It takes a list of bounding boxes and a shape (height, width) and clips the bounding boxes to the
    shape

    Args:
      boxes (torch.Tensor): the bounding boxes to clip
      shape (tuple): the shape of the image
    """

    if isinstance(boxes, np.ndarray):  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2

    else:  # faster individually
        import torch
        assert isinstance(boxes, torch.Tensor), "boxes type isn't numpy or torch"
        boxes[..., 0].clamp_(0, shape[1])  # x1
        boxes[..., 1].clamp_(0, shape[0])  # y1
        boxes[..., 2].clamp_(0, shape[1])  # x2
        boxes[..., 3].clamp_(0, shape[0])  # y2


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    """
    Rescales bounding boxes (in the format of xyxy) from the shape of the image they were originally specified in
    (img1_shape) to the shape of a different image (img0_shape).

    Args:
      img1_shape (tuple): The shape of the image that the bounding boxes are for, in the format of (height, width).
      boxes (torch.Tensor): the bounding boxes of the objects in the image, in the format of (x1, y1, x2, y2)
      img0_shape (tuple): the shape of the target image, in the format of (height, width).
      ratio_pad (tuple): a tuple of (ratio, pad) for scaling the boxes. If not provided, the ratio and pad will be
                         calculated based on the size difference between the two images.

    Returns:
      boxes (torch.Tensor): The scaled bounding boxes, in the format of (x1, y1, x2, y2)
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes


def clip_coords(coords, shape):
    """
    Clip line coordinates to the image boundaries.

    Args:
        coords (torch.Tensor) or (numpy.ndarray): A list of line coordinates.
        shape (tuple): A tuple of integers representing the size of the image in the format (height, width).

    Returns:
        (None): The function modifies the input `coordinates` in place, by clipping each coordinate to the image boundaries.
    """
    if isinstance(coords, np.ndarray):  # np.array (faster grouped)
        coords[..., 0] = coords[..., 0].clip(0, shape[1])  # x
        coords[..., 1] = coords[..., 1].clip(0, shape[0])  # y
    else:  # faster individually
        import torch
        assert isinstance(coords, torch.Tensor), "coords type isn't numpy or torch"

        coords[..., 0].clamp_(0, shape[1])  # x
        coords[..., 1].clamp_(0, shape[0])  # y


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None, normalize=False):
    """
    Rescale segment coordinates (xyxy) from img1_shape to img0_shape

    Args:
      img1_shape (tuple): The shape of the image that the coords are from.
      coords (torch.Tensor): the coords to be scaled
      img0_shape (tuple): the shape of the image that the segmentation is being applied to
      ratio_pad (tuple): the ratio of the image size to the padded image size.
      normalize (bool): If True, the coordinates will be normalized to the range [0, 1]. Defaults to False

    Returns:
      coords (torch.Tensor): the segmented image.
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[..., 0] -= pad[0]  # x padding
    coords[..., 1] -= pad[1]  # y padding
    coords[..., 0] /= gain
    coords[..., 1] /= gain
    clip_coords(coords, img0_shape)
    if normalize:
        coords[..., 0] /= img0_shape[1]  # width
        coords[..., 1] /= img0_shape[0]  # height
    return coords


if __name__ == "__main__":
    model_path = r'D:\AI_Vision_Project\New_Feature\openvino_projects\Instance Segmentation5 meta Tool1\Models\Model_2023_06_28\model_final_onnxe2e.onnx'
    img_path = r'D:\AI_Vision_Project\New_Feature\openvino_projects\Instance Segmentation5 meta Tool1\Images\9_Image.png'
    device = "GPU"  # GPU,CPU
    InsSeg_onnx = ObjectDetectionV2IR(model_path, device)
    for i in range(5):
        result = InsSeg_onnx(img_path)
    out_path = os.path.join(os.path.dirname(model_path), 'onnx')
    os.makedirs(out_path, exist_ok=True)
    InsSeg_onnx.draw_detections(img_path, result, out_path)

    model_path = r'D:\AI_Vision_Project\New_Feature\openvino_projects\Instance Segmentation5 meta Tool1\Models\Model_2023_06_28\model_final.xml'
    img_path = r'D:\AI_Vision_Project\New_Feature\openvino_projects\Instance Segmentation5 meta Tool1\Images\9_Image.png'
    device = "CPU"  # GPU,CPU
    InsSeg_ov = ObjectDetectionV2IR(model_path, device)
    for i in range(5):
        result = InsSeg_ov(img_path)
    out_path = os.path.join(os.path.dirname(model_path), 'openvino')
    os.makedirs(out_path, exist_ok=True)
    InsSeg_ov.draw_detections(img_path, result, out_path)

    model_path = r'D:\AI_Vision_Project\New_Feature\openvino_projects\Feature Keypoint5 meta Tool2\Models\Model_2023_07_06\model_final_onnxe2e.onnx'
    img_path = r'D:\AI_Vision_Project\New_Feature\openvino_projects\Feature Keypoint5 meta Tool2\Images\9_Image.png'
    device = "GPU"  # GPU,CPU
    FKp_onnx = ObjectDetectionV2IR(model_path, device)
    for i in range(5):
        result = FKp_onnx(img_path)
    out_path = os.path.join(os.path.dirname(model_path), 'onnx')
    os.makedirs(out_path, exist_ok=True)
    FKp_onnx.draw_detections(img_path, result, out_path)

    model_path = r'D:\AI_Vision_Project\New_Feature\openvino_projects\Feature Keypoint5 meta Tool2\Models\Model_2023_07_06\model_final.xml'
    img_path = r'D:\AI_Vision_Project\New_Feature\openvino_projects\Feature Keypoint5 meta Tool2\Images\9_Image.png'
    device = "CPU"  # GPU,CPU
    FKp_ov = ObjectDetectionV2IR(model_path, device)
    for i in range(5):
        result = FKp_ov(img_path)
    out_path = os.path.join(os.path.dirname(model_path), 'openvino')
    os.makedirs(out_path, exist_ok=True)
    FKp_ov.draw_detections(img_path, result, out_path)

    model_path = r'D:\AI_Vision_Project\New_Feature\openvino_projects\Instance Keypoint5 meta Tool1\Models\Model_2023_07_06\model_final.onnx'
    img_path = r'D:\AI_Vision_Project\New_Feature\openvino_projects\Instance Keypoint5 meta Tool1\Images\9_Image.png'
    device = "GPU"  # GPU,CPU
    InsSeg_onnx = ObjectDetectionV2IR(model_path, device)
    for i in range(5):
        result = InsSeg_onnx(img_path)
    out_path = os.path.join(os.path.dirname(model_path), 'onnx')
    os.makedirs(out_path, exist_ok=True)
    InsSeg_onnx.draw_detections(img_path, result, out_path)

    model_path = r'D:\AI_Vision_Project\New_Feature\openvino_projects\Instance Keypoint5 meta Tool1\Models\Model_2023_07_06\model_final.xml'
    img_path = r'D:\AI_Vision_Project\New_Feature\openvino_projects\Instance Keypoint5 meta Tool1\Images\9_Image.png'
    device = "CPU"  # GPU,CPU
    InsSeg_ov = ObjectDetectionV2IR(model_path, device)
    for i in range(5):
        result = InsSeg_ov(img_path)
    out_path = os.path.join(os.path.dirname(model_path), 'openvino')
    os.makedirs(out_path, exist_ok=True)
    InsSeg_ov.draw_detections(img_path, result, out_path)
