import torch
import numpy as np
import torch.nn.functional as F
from slmdptyu.solomon.ultralytics.yolo.utils import ROOT, ops
from slmdptyu.solomon.ultralytics.yolo.v8.detect.predict import DetectionSolPredictor
from slmdptyu.solomon.ultralytics.yolo.engine.results import ResultsSol
from slmdptyu.solomon.ultralytics.yolo.utils.convert import MeanShiftCircle
from slmdptyu.solomon.ultralytics.yolo.utils import DEFAULT_CFG


class SegmentationSolPredictor(DetectionSolPredictor):

    def postprocess(self, preds, img, orig_imgs):
        # TODO: filter by classes
        p = ops.non_max_suppression(preds[0],
                                    self.args.conf,
                                    self.args.iou,
                                    agnostic=self.args.agnostic_nms,
                                    max_det=self.args.max_det,
                                    nc=len(self.model.names),
                                    classes=self.args.classes)
        results = []
        proto = preds[1][-1] if len(preds[1]) == 3 else preds[1]  # second output is len 3 if pt, but only 1 if exported
        for i, pred in enumerate(p):
            orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
            path, _, _, _, _ = self.batch
            img_path = path[i] if isinstance(path, list) else path
            if not len(pred):  # save empty boxes
                results.append(
                    ResultsSol(
                        orig_img=orig_img, 
                        path=img_path, 
                        names=self.model.names, 
                        boxes=pred[:, :6])
                    )
                continue

            if self.args.train_angle:
                mi = 78
            else:
                mi = 6

            if self.args.retina_masks:
                if not isinstance(orig_imgs, torch.Tensor):
                    pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
                masks = ops.process_mask_native(proto[i], pred[:, mi:mi+32], pred[:, :4], orig_img.shape[:2])  # HWC
            else:
                masks = ops.process_mask(proto[i], pred[:, mi:mi+32], pred[:, :4], img.shape[2:], upsample=True)  # HWC
                if not isinstance(orig_imgs, torch.Tensor):
                    pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)

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
                    masks=masks,
                    angles=angles
                )
            )
        return results
