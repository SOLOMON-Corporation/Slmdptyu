import os
import sys
import cv2
import json
import torch
import numpy as np
from pathlib import Path
from skimage.io import imread
from pycocotools.coco import COCO
from slmmetric.Evaluator import Evaluator
from slmdptyu.solomon.ultralytics.yolo.data.build import build_dataloader
from slmdptyu.solomon.ultralytics.yolo.engine.validator import BaseSolValidator
from slmdptyu.solomon.ultralytics.yolo.utils import ops, LOGGER, yaml_load
from slmdptyu.solomon.ultralytics.yolo.utils.metrics import SolMetrics
from slmdptyu.solomon.ultralytics.yolo.utils.plotting import plot_batch
from slmdptyu.solomon.ultralytics.yolo.utils.convert import MeanShiftCircle
from slmdptyu.ultralytics.ultralytics.yolo.utils.torch_utils import de_parallel

class DetectionSolValidator(BaseSolValidator):

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.args.task = 'detect'
        self.data_dict = yaml_load(args.data, append_filename=True)
        self.nkpoints = self.data_dict.get('kpt_shape', [0, 3])[0]
        self.with_angle = self.data_dict.get('with_angle', False)
        self.reflect_lines = self.data_dict.get('reflect_lines', [0])
        self.is_coco = False
        self.class_map = None
        self.metrics = SolMetrics(save_dir=self.save_dir)
        self.iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()
        self.coco_data = COCO(self.data_dict['val'])
        self.coco_gt = self.coco_data.dataset
        self.coco_pred = self.coco_data.dataset.copy()
        self.coco_pred['annotations'] = []
        self.imagename_id_dict = {
            img_info['file_name']: img_info['id']
                for img_info in self.coco_gt['images']
        }
        self.annotation_id = 1
        if args.plots:
            for folder in ['val_imgs_gt', 'val_imgs_pred']:
                try:
                    os.mkdir(os.path.join(args.save_dir, folder))
                except:
                    pass

    def preprocess(self, batch):
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = (batch["img"].half() if self.args.half else batch["img"].float()) / 255
        for k in ["batch_idx", "cls", "bboxes", "angles", "angles_valid"]:
            if k in batch.keys():
                batch[k] = batch[k].to(self.device)

        nb = len(batch["img"])
        self.lb = [torch.cat([batch["cls"], batch["bboxes"]], dim=-1)[batch["batch_idx"] == i]
                   for i in range(nb)] if self.args.save_hybrid else []  # for autolabelling

        return batch

    def init_metrics(self, model):
        val = self.data.get(self.args.split, '')  # validation path
        self.is_coco = isinstance(val, str) and 'coco' in val and val.endswith(f'{os.sep}val2017.txt')  # is COCO
        self.class_map = ops.coco80_to_coco91_class() if self.is_coco else list(range(1000))
        self.args.save_json |= self.is_coco and not self.training  # run on final val if training COCO
        self.names = model.names
        self.nc = len(model.names)
        self.metrics.names = self.names
        self.metrics.plot = self.args.plots
        self.batch_idx = []
        self.plot_bboxs = []
        self.plot_angles = []
        self.plot_masks = []
        self.plot_kpts = []
        self.seen = 0
        self.jdict = []
        self.stats = []

    def get_desc(self):
        mertrics = ["class"] + ['AP', f'F{self.args.eval_fscore}', 'a', 'p', 'r', 'score'] \
                + ['TP', 'FP', 'FN'] + [f"KP_{i+1}" for i in range(self.nkpoints)]
        if self.with_angle: 
            mertrics += ['ang']
        return ('%8s' * (len(mertrics))) % tuple(mertrics)

    def postprocess(self, preds):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        labels=self.lb,
                                        multi_label=True,
                                        agnostic=self.args.single_cls,
                                        max_det=self.args.max_det,
                                        nc=self.nc)
        return preds

    def update_metrics(self, preds, batch):
        self.annotation_id = 1
        # Metrics
        for si, pred in enumerate(preds):
            idx = batch["batch_idx"] == si
            cls = batch["cls"][idx]
            nl, npr = cls.shape[0], pred.shape[0]  # number of labels, predictions
            shape = batch["ori_shape"][si]
            self.seen += 1

            if npr == 0:
                continue

            # Predictions
            if self.args.single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            ops.scale_boxes(
                img1_shape=batch["img"][si].shape[1:], 
                boxes=predn[:, :4], 
                img0_shape=shape,
                ratio_pad=batch["ratio_pad"][si])  # native-space pred
            
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
                if self.args.train_angle:
                    self.plot_angles.append(torch.Tensor(pred_angles)[:20])

            # Save
            if self.args.save_json:
                self.pred_to_json(predn, batch["im_file"][si], pred_angles)
            # if self.args.save_txt:
            #     file = self.save_dir / 'labels' / f'{Path(batch["im_file"][si]).stem}.txt'
            #     self.save_one_txt(predn, self.args.save_conf, shape, file)

    def get_stats(self):
        with open(str(self.save_dir / "predictions.json"), 'w') as f:
            json.dump(self.coco_pred, f, indent=4)  # flatten and save
        self.coco_pred['annotations'] = []
        self.annotation_id = 1

        result = self.eval_json(self.stats)
        return result

    def print_results(self):
        pf = '%8s' + '%8.3g'*(len(self.metrics.keys) + self.nkpoints + self.with_angle)  # print format
        for k, v in self.metrics.results.items():
            LOGGER.info(pf % tuple([k] + list(v.values())))

    def get_dataloader(self, dataset_path, batch_size):
        # TODO: manage splits differently
        # calculate stride - check if model is initialized
        gs = max(int(de_parallel(self.model).stride if self.model else 0), 32)
        return build_dataloader(
            self.args, 
            batch_size, 
            coco_path=dataset_path, 
            stride=gs, 
            data_info=self.data, 
            mode="val")[0]

    def plot_val_samples(self, batch, ni):
        class_names = list(self.names.values())
        images = (batch['img']*255).round().cpu().numpy().astype(np.uint8).transpose(0, 2, 3, 1)[..., ::-1]
        imgsz_h, imgsz_w = images.shape[1:3]
        batch_results = []

        batch_idx = batch["batch_idx"]
        for idx in batch_idx.unique().cpu().numpy().astype(int):
            rois = (batch['bboxes'][batch_idx == idx]).cpu().numpy()
            rois[:, [0, 2]] *= imgsz_w
            rois[:, [1, 3]] *= imgsz_h
            rois = np.concatenate(
                [
                    rois[:, [0]] - rois[:, [2]]/2, 
                    rois[:, [1]] - rois[:, [3]]/2, 
                    rois[:, [0]] + rois[:, [2]]/2, 
                    rois[:, [1]] + rois[:, [3]]/2
                ], 1
            )
            class_ids = batch['cls'][batch_idx == idx].cpu().numpy().flatten()
            results = {
                'rois': rois,
                'class_ids': class_ids
            }
            if self.args.train_mask:
                masks = batch['masks'][batch_idx == idx].cpu().numpy().transpose(1, 2, 0)
                masks = cv2.resize(masks, (imgsz_w, imgsz_h), interpolation=cv2.INTER_LINEAR)
                masks = masks.transpose(2, 0, 1)
                results['masks'] = [
                    mask[box[1]:box[3], box[0]:box[2]]
                    for box, mask in zip(rois.round().astype(int), masks)
                ]
            if self.args.train_keypoints:
                keypoints = batch['keypoints'][batch_idx == idx][..., :2].cpu().numpy()
                keypoints[..., 0] *= imgsz_w
                keypoints[..., 1] *= imgsz_h
                results['keypoints'] = keypoints
            batch_results.append(results)

        plot_batch(
            class_names, 
            images,
            file_names = batch['im_file'],
            batch_results=batch_results, 
            save_dir=self.save_dir/'val_imgs_gt',
            plot_cfg = {
                'show_bboxes': True,
                'show_masks':  self.args.train_mask,
                'show_kpts':   self.args.train_keypoints,
                'show_scores': False,
                'show_label':  False
            }
        )

    def plot_predictions(self, batch, preds, ni):
        class_names = list(self.names.values())
        images = [imread(file)[..., ::-1] for file in batch['im_file']]
        batch_results = []

        for idx in range(len(self.batch_idx)):
            rois = self.plot_bboxs[idx][:, :4].cpu().numpy()
            rois = np.concatenate(
                [
                    rois[:, [0]] - rois[:, [2]]/2,
                    rois[:, [1]] - rois[:, [3]]/2,
                    rois[:, [0]] + rois[:, [2]]/2,
                    rois[:, [1]] + rois[:, [3]]/2
                ], 1
            ).round().astype(int)
            h, w = batch['ori_shape'][idx]
            rois[:, [0, 2]] = np.clip(rois[:, [0, 2]], 0, w)
            rois[:, [1, 3]] = np.clip(rois[:, [1, 3]], 0, h)
            class_ids = self.plot_bboxs[idx][:, 5].cpu().numpy().flatten()
            scores = self.plot_bboxs[idx][:, 4].cpu().numpy().flatten()
            results = {
                'rois': rois,
                'class_ids': class_ids,
                'scores': scores
            }
            if self.args.train_mask:
                masks = [
                    mask[box[1]:box[3], box[0]:box[2]]
                    for box, mask in zip(
                        rois,
                        self.plot_masks[idx].cpu().numpy()
                    )
                ]
                results['masks'] = masks
            if self.args.train_keypoints:
                keypoints = self.plot_kpts[idx].cpu().numpy()
                results['keypoints'] = keypoints[..., :2]
            batch_results.append(results)

        plot_batch(
            class_names, 
            images,
            file_names = batch['im_file'],
            batch_results=batch_results, 
            save_dir=self.save_dir/'val_imgs_pred',
            plot_cfg = {
                'show_bboxes': True,
                'show_masks':  self.args.train_mask,
                'show_kpts':   self.args.train_keypoints,
                'show_scores': True,
                'show_label': False
            }
        )

        self.batch_idx.clear()
        self.plot_bboxs.clear()
        self.plot_angles.clear()
        self.plot_kpts.clear()
        self.plot_masks.clear()

    def pred_to_json(self, predn, filename, pred_angles=None):
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
                'segmentation': [[]],
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
            'pdjThr': 0.008,
            'angThr': 2.5,
            'fScore': self.args.eval_fscore,
            'isUnknown': False,
            'calKeypoints': False,
            'calAngle': self.with_angle,
            'reflect_lines': self.reflect_lines,
            'showAP': True,
            'plot': False,
            'save_dir': GPath
        }

        evaluator = Evaluator(gt_coco, Det_coco)
        metricsPerClass = evaluator.PlotPrecisionRecallCurve(**parameters)  


        results = metricsPerClass['all']

        self.metrics.results = metricsPerClass
        return metricsPerClass
