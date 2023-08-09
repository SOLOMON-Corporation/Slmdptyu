import os
import sys
import torch
import numpy as np
from pathlib import Path
from pycocotools.coco import COCO
from slmmetric.Evaluator import Evaluator
from slmdptyu.solomon.ultralytics.yolo.utils import ops
from slmdptyu.solomon.ultralytics.yolo.utils.metrics import SolMetrics
from slmdptyu.solomon.ultralytics.yolo.utils.convert import MeanShiftCircle
from slmdptyu.solomon.ultralytics.yolo.v8.detect.val import DetectionSolValidator

class PoseSolValidator(DetectionSolValidator):

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None):
        super().__init__(dataloader, save_dir, pbar, args)
        self.args.task = 'pose'
        self.metrics = SolMetrics(save_dir=self.save_dir)

    def preprocess(self, batch):
        batch = super().preprocess(batch)
        batch['keypoints'] = batch['keypoints'].to(self.device).float()
        return batch

    def postprocess(self, preds):
        preds = ops.non_max_suppression(preds[0],
                                        self.args.conf,
                                        self.args.iou,
                                        labels=self.lb,
                                        multi_label=True,
                                        agnostic=self.args.single_cls,
                                        max_det=self.args.max_det,
                                        nc=self.nc)
        return preds  

    def update_metrics(self, preds, batch):
        # Metrics
        for si, pred in enumerate(preds):
            idx = batch["batch_idx"] == si
            cls = batch["cls"][idx]
            kpts = batch['keypoints'][idx]
            nl, npr = cls.shape[0], pred.shape[0]  # number of labels, predictions
            nk = kpts.shape[1] 
            shape = batch["ori_shape"][si]
            self.seen += 1

            if npr == 0:
                continue

            if self.args.train_angle:
                mi = 78
            else:
                mi = 6

            # Predictions
            if self.args.single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            ops.scale_boxes(
                img1_shape=batch["img"][si].shape[1:], 
                boxes=predn[:, :4], 
                img0_shape=shape, 
                ratio_pad=batch["ratio_pad"][si])  # native-space pred
            pred_kpts = predn[:, mi:].view(npr, nk, -1)
            ops.scale_coords(
                img1_shape=batch['img'][si].shape[1:], 
                coords=pred_kpts, 
                img0_shape=shape, 
                ratio_pad=batch['ratio_pad'][si])

            # Angle
            if self.args.train_angle:
                angles_features = torch.nn.Softmax(-1)(predn[:, 6:78].reshape(-1, 9, 8)).reshape(-1, 72)
                pred_angles = MeanShiftCircle(angles_features.cpu().numpy())
                pred_angles = np.array(pred_angles)*180/np.pi
            else:
                pred_angles = None

            # Evaluate
            if nl and self.args.plots:
                tbox = [
                    self.coco_gt["annotations"][i]['bbox'] 
                        for i in torch.where(idx==True)[0].cpu().numpy()
                ]
                tbox = np.array(tbox)
                tbox[:, 2:] += tbox[:, :2]
                tbox = torch.from_numpy(tbox).to(torch.float32).to(cls.device)

            # Plots
            if self.args.plots: #  and self.batch_i < 3
                # filter top 20 to plot
                self.batch_idx.append(torch.Tensor(([si] * len(predn[:, 5][:20]))))
                box = ops.xyxy2xywh(predn[:20])  # xywh
                self.plot_bboxs.append(box.cpu())
                self.plot_kpts.append(pred_kpts[:20].cpu())
                if self.args.train_angle:
                    self.plot_angles.append(torch.Tensor(pred_angles)[:20])

            # Save
            if self.args.save_json:
                self.pred_to_json(predn, batch["im_file"][si], pred_kpts, pred_angles)
            # if self.args.save_txt:
            #    save_one_txt(predn, save_conf, shape, file=save_dir / 'labels' / f'{path.stem}.txt')

    def pred_to_json(self, predn, filename, pred_keypoints, pred_angles=None):
        name = Path(filename).name
        image_id = self.imagename_id_dict[name]
        box = ops.xyxy2xywh(predn[:, :4])  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
        for i, (p, b) in enumerate(zip(predn.tolist(), box.tolist())):
            annotation_info = {
                'id': self.annotation_id,
                'image_id': image_id,
                'category_id': int(p[5]) + 1,
                'iscrowd': 0,
                'bbox': [round(x, 3) for x in b],
                'score': round(p[4], 5),
                'keypoints': pred_keypoints[i].tolist(),
                'angle': -1
            }
            if isinstance(pred_angles, np.ndarray):
                annotation_info['angle'] = pred_angles[i]
            self.coco_pred['annotations'].append(annotation_info)
            self.annotation_id += 1

    def eval_json(self, stats):
        GPath = self.data_dict["val"]
        DPath = self.save_dir / 'predictions.json'

        gt_coco = COCO(GPath) # Load groundtruth annotations
        Det_coco = COCO(DPath) # Load detection annotations
        
        parameters = {
            'iouType': 'bbox',
            'iouThr': self.args.eval_iou, 
            'pdjThr': 0.0046875,
            'angThr': 2.5,
            'fScore': self.args.eval_fscore,
            'isUnknown': False,
            'calKeypoints': True,
            'calAngle': self.with_angle,
            'reflect_lines': self.reflect_lines,
            'showAP': True,
            'plot': False,
            'save_dir': GPath
        }

        evaluator = Evaluator(gt_coco, Det_coco)
        metricsPerClass = evaluator.PlotPrecisionRecallCurve(**parameters)   

        self.metrics.results = metricsPerClass
        return metricsPerClass
