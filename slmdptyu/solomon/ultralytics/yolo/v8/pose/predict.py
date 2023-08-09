import torch
import numpy as np
from slmdptyu.solomon.ultralytics.yolo.utils import ROOT, ops
from slmdptyu.solomon.ultralytics.yolo.v8.detect.predict import DetectionSolPredictor
from slmdptyu.solomon.ultralytics.yolo.engine.results import ResultsSol
from slmdptyu.solomon.ultralytics.yolo.utils.convert import MeanShiftCircle
from slmdptyu.solomon.ultralytics.yolo.utils import DEFAULT_CFG


class PoseSolPredictor(DetectionSolPredictor):

    def postprocess(self, preds, img, orig_img):
        # TODO: filter by classes
        preds = ops.non_max_suppression(preds[0],
                                    self.args.conf,
                                    self.args.iou,
                                    agnostic=self.args.agnostic_nms,
                                    max_det=self.args.max_det,
                                    classes=self.args.classes,
                                    nc=len(self.model.names))
        ki = 78 if self.args.train_angle else 6
        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_img[i] if isinstance(orig_img, list) else orig_img
            shape = orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()
            pred_kpts = pred[:, ki:].view(len(pred), *self.model.kpt_shape) if len(pred) else pred[:, 6:]
            pred_kpts = ops.scale_coords(img.shape[2:], pred_kpts, shape)
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
                    keypoints=pred_kpts,
                    angles=angles
                )
            )
        return results
