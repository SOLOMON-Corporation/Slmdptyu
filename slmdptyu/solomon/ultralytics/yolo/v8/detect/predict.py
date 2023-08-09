import cv2
import torch
import torch.nn.functional as F
import numpy as np
from slmdptyu.solomon.ultralytics.yolo.engine.predictor import BaseSolPredictor
from slmdptyu.solomon.ultralytics.yolo.utils import ROOT, ops
from slmdptyu.solomon.ultralytics.yolo.engine.results import ResultsSol
from slmdptyu.solomon.ultralytics.yolo.utils.convert import MeanShiftCircle
from slmdptyu.solomon.ultralytics.yolo.utils import DEFAULT_CFG


class DetectionSolPredictor(BaseSolPredictor):

    def preprocess(self, img):
        img = (img if isinstance(img, torch.Tensor) else torch.from_numpy(img)).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_imgs):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det,
                                        nc=len(self.model.names),
                                        classes=self.args.classes)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
            if not isinstance(orig_imgs, torch.Tensor):
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            path, _, _, _, _ = self.batch
            img_path = path[i] if isinstance(path, list) else path
            if self.args.train_angle:
                angles_features = torch.nn.Softmax(-1)(pred[:, 6:78].reshape(-1, 9, 8)).reshape(-1, 72)
                angles = MeanShiftCircle(angles_features.cpu().numpy())
                angles = torch.Tensor(np.array(angles)*180/np.pi)
            else:
                angles = None

            results.append(
                ResultsSol(
                    orig_img=orig_img, 
                    path=img_path, 
                    names=self.model.names, 
                    boxes=pred[:, :6],
                    angles=angles
                )
            )
        return results

    def postprocess_angle(self, angle_class):
        angles_features = torch.nn.Softmax(-1)(angle_class.reshape(-1, 9, 8)).reshape(-1, 72)
        angles = MeanShiftCircle(angles_features.cpu().numpy())
        angles = torch.Tensor(np.array(angles)*180/np.pi)
        return angles

    def assign_results_onnx(self, preds, img, orig_img):
        results = []
        orig_img = orig_img[1] if isinstance(orig_img, list) else orig_img
        shape = orig_img.shape
        boxes = ops.scale_boxes(img.shape[2:], preds[1], shape).round()
        scores = preds[2][..., None]
        class_ids = preds[3][..., None]

        path, _, _, _, _ = self.batch
        img_path = path[1] if isinstance(path, list) else path

        if self.args.train_mask:
            masks = (preds[4].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            mh, mw = masks.shape[:2]
            gain = min(mh / shape[0], mw / shape[1])  # gain  = old / new
            pad = (mw - shape[1] * gain) / 2, (mh - shape[0] * gain) / 2  # wh padding
            top, left = int(pad[1]), int(pad[0])  # y, x
            bottom, right = int(mh - pad[1]), int(mw - pad[0])
            masks = masks[top:bottom, left:right]
            masks = cv2.resize(masks, (shape[:2][1], shape[:2][0])).transpose(2, 0, 1)
            masks = torch.Tensor(masks/255)
        else:
            masks = None

        if self.args.train_keypoints:
            keypoints = ops.scale_coords(img.shape[2:], preds[-2], shape)
        else:
            keypoints = None

        if self.args.train_angle:
            angles = self.postprocess_angle(preds[-1])
        else:
            angles = None


        pred = torch.cat((boxes, scores, class_ids), 1)
        
        results.append(
            ResultsSol(
                orig_img=orig_img, 
                path=img_path, 
                names=self.model.names, 
                boxes=pred,
                masks=masks,
                keypoints=keypoints,
                angles=angles
            )
        )
        return results

    def assign_results_trt(self, preds, img, orig_img, engine):
        import tensorrt as trt
        import packaging.version

        valid, bboxes, scores, classids, masks, kpts, angles = None,None,None,None,None,None,None
        num_inputs = 3
        for i in range(num_inputs, len(engine)):
            if packaging.version.parse(trt.__version__) < packaging.version.parse("8.5.0.0"):
                tensor_name = engine.get_binding_name(i)
                tensor_shape = tuple(engine.get_binding_shape(tensor_name))
            else:
                tensor_name = engine.get_tensor_name(i)
                tensor_shape = tuple(engine.get_tensor_shape(tensor_name))
            
            if tensor_name == 'bboxes' or tensor_name == 'masks' or tensor_name == 'kpts' or tensor_name == 'angles' :
                preds[i-num_inputs] = preds[i-num_inputs].reshape(tensor_shape)
            
            if tensor_name == 'valid': valid=preds[i-num_inputs]
            elif tensor_name == 'bboxes': bboxes=preds[i-num_inputs]
            elif tensor_name == 'scores': scores=preds[i-num_inputs]
            elif tensor_name == 'classids': classids=preds[i-num_inputs]
            elif tensor_name == 'masks': masks=preds[i-num_inputs]
            elif tensor_name == 'kpts': kpts=preds[i-num_inputs]
            elif tensor_name == 'angles': angles=preds[i-num_inputs]
        new_preds = [valid, bboxes, scores, classids, masks, kpts, angles]
        preds = list(filter(lambda x: x is not None, new_preds))
        
        valid = (preds[0]>0).sum()
        for i in range(len(preds)):
            preds[i] = preds[i][:valid]
        
        return self.assign_results_onnx(preds, img, orig_img)

    def assign_results_tflite(self, preds, img, orig_img):
        results = []
        orig_img = orig_img[1] if isinstance(orig_img, list) else orig_img
        shape = orig_img.shape
        valid = preds[0]
        boxes = ops.scale_boxes(
            img.shape[2:], 
            preds[1][:valid][:, [1, 0, 3, 2]], # y1x1y2x2 to x1y1x2y2
            shape
        ).round()
        scores = preds[2][:valid][..., None]
        class_ids = preds[3][:valid][..., None].to(torch.float32)

        path, _, _, _, _ = self.batch
        img_path = path[1] if isinstance(path, list) else path

        if self.args.train_angle:
            angles_features = preds[-1][:valid].reshape(-1, 9, 8)
            angles_features = torch.nn.Softmax(-1)(angles_features).reshape(-1, 72)
            angles = MeanShiftCircle(angles_features.cpu().numpy())
            angles = torch.Tensor(np.array(angles)*180/np.pi)
        else:
            angles = None
        pred = torch.cat((boxes, scores, class_ids), 1)

        results.append(
            ResultsSol(
                orig_img=orig_img, 
                path=img_path, 
                names=self.model.names, 
                boxes=pred,
                angles=angles
            )
        )
        return results
