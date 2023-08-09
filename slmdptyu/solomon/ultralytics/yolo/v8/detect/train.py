import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from copy import copy, deepcopy
from slmdptyu.solomon.ultralytics import __version__
from slmdptyu.solomon.ultralytics.yolo.v8.detect.val import DetectionSolValidator
from slmdptyu.solomon.ultralytics.yolo.data.build import build_dataloader
from slmdptyu.solomon.ultralytics.yolo.utils import RANK
from slmdptyu.solomon.ultralytics.yolo.engine.trainer import BaseSolTrainer
from slmdptyu.solomon.ultralytics.yolo.utils.plotting import plot_batch
from slmdptyu.solomon.ultralytics.nn.tasks import DetectionSolModel
from slmdptyu.solomon.ultralytics.yolo.utils.tal import TaskAlignedSolAssigner
from slmdptyu.solomon.ultralytics.yolo.utils import DEFAULT_CFG
from slmdptyu.ultralytics.ultralytics.yolo.utils.loss import BboxLoss
from slmdptyu.ultralytics.ultralytics.yolo.utils.ops import xywh2xyxy
from slmdptyu.ultralytics.ultralytics.yolo.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors
from slmdptyu.ultralytics.ultralytics.yolo.utils.torch_utils import de_parallel

# BaseTrainer python usage
class DetectionSolTrainer(BaseSolTrainer):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        if overrides['plots']:
            try:
                os.mkdir(os.path.join(overrides['save_dir'], 'aug_imgs'))
            except:
                pass

    def get_dataloader(self, dataset_path, batch_size, mode='train', rank=0):
        # TODO: manage splits differently
        # calculate stride - check if model is initialized
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_dataloader(
            self.args, 
            batch_size, 
            coco_path=dataset_path, 
            stride=gs, 
            rank=rank, 
            mode=mode,
            rect=mode == 'val', 
            data_info=self.data)[0]

    def preprocess_batch(self, batch):
        batch['img'] = batch['img'].to(self.device, non_blocking=True).float() / 255
        return batch

    def set_model_attributes(self):
        # nl = de_parallel(self.model).model[-1].nl  # number of detection layers (to scale hyps)
        # self.args.box *= 3 / nl  # scale to layers
        # self.args.cls *= self.data["nc"] / 80 * 3 / nl  # scale to classes and layers
        # self.args.cls *= (self.args.imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
        self.model.nc = self.data['nc']  # attach number of classes to model
        self.model.names = self.data['names']  # attach class names to model
        self.model.kpts_names = self.data['kpts_names']
        self.model.max_gauge_value = self.data['max_gauge_value']
        self.model.kpts_coordinates = self.data['kpts_coordinates']
        self.model.img = self.data['img']
        self.model.args = self.args  # attach hyperparameters to model
        # TODO: self.model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = DetectionSolModel(
            cfg, nc=self.data['nc'], with_angle=self.data['with_angle'], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        if self.args.train_angle:
            self.loss_names = 'box_loss', 'cls_loss', 'dfl_loss', 'ang_loss'
        else:
            self.loss_names = 'box_loss', 'cls_loss', 'dfl_loss'
        return DetectionSolValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))

    def criterion(self, preds, batch):
        if not hasattr(self, 'compute_loss'):
            self.compute_loss = Loss(de_parallel(self.model))
        return self.compute_loss(preds, batch)

    def label_loss_items(self, loss_items=None, prefix='train'):
        """
        Returns a loss dict with labelled training loss items tensor
        """
        # Not needed for classification but necessary for segmentation & detection
        keys = [f'{prefix}/{x}' for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            return keys

    def progress_string(self):
        return ('\n' + '%11s' *
                (4 + len(self.loss_names))) % ('Epoch', 'GPU_mem', *self.loss_names, 'Instances', 'Size')

    def plot_training_samples(self, batch, ni):
        class_names = list(self.data['names'].values())
        images = (batch['img']*255).round().cpu().numpy().astype(np.uint8).transpose(0, 2, 3, 1)[..., ::-1]
        imgsz = images.shape[1]
        batch_results = []

        batch_idx = batch['batch_idx']
        for idx in batch_idx.unique().cpu().numpy().astype(int):
            rois = (batch['bboxes'][batch_idx == idx] * imgsz).cpu().numpy()
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
                'rois': rois.round(),
                'class_ids': class_ids
            }
            if self.args.train_mask:
                masks = batch['masks'][batch_idx == idx].cpu().numpy().transpose(1, 2, 0)
                masks = cv2.resize(masks, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
                if len(masks.shape) == 2:
                    masks = masks[..., None]
                masks = masks.transpose(2, 0, 1)
                results['masks'] = [
                    mask[box[1]:box[3], box[0]:box[2]]
                    for box, mask in zip(rois.round().astype(int), masks)
                ]
            if self.args.train_keypoints:
                keypoints = batch['keypoints'][batch_idx == idx][..., :2].cpu().numpy()
                keypoints *= imgsz
                results['keypoints'] = keypoints
            batch_results.append(results)

        plot_batch(
            class_names, 
            images,
            file_names = batch['im_file'],
            batch_results=batch_results, 
            save_dir=self.save_dir/'aug_imgs',
            plot_cfg = {
                'show_bboxes': True,
                'show_masks':  self.args.train_mask,
                'show_kpts':   self.args.train_keypoints,
                'show_scores': False,
                'show_label': False
            }
        )

    def plot_metrics(self):
        pass

    def save_model(self):
        ckpt = {
            'epoch': self.epoch,
            'best_fitness': self.best_fitness,
            'model': deepcopy(de_parallel(self.model)).half(),
            'ema': deepcopy(self.ema.ema).half(),
            'updates': self.ema.updates,
            'optimizer': self.optimizer.state_dict(),
            'train_args': vars(self.args),  # save as dict
            'date': datetime.now().isoformat(),
            'version': __version__}

        # Save best and delete
        if self.best_fitness == self.fitness:
            torch.save(ckpt, self.best)
        del ckpt

# Criterion class for computing training losses
class Loss:
    small_section = 9
    big_section = 8
    unit = 360 // big_section // small_section # 5 
    table = np.arange(0, 360, unit).reshape(big_section, small_section).T

    def __init__(self, model):  # model must be de-paralleled

        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.ce = nn.CrossEntropyLoss()
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.no
        self.with_angle = m.with_angle
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        if self.with_angle:
            self.assigner = TaskAlignedSolAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        else:
            self.assigner = TaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=self.use_dfl).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets, batch_size, scale_tensor):
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, targets.shape[1]-1, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), targets.shape[1]-1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch):
        feats = preds[1] if isinstance(preds, tuple) else preds

        if self.with_angle:
            loss = torch.zeros(4, device=self.device)  # box, cls, dfl, ang
            pred_distri, pred_scores, pred_angle_feats = torch.cat([xi.view(feats[0].shape[0], self.no+72, -1) for xi in feats], 2).split(
                (self.reg_max * 4, self.nc, 72), 1)
            pred_angle_feats = pred_angle_feats.permute(0, 2, 1).contiguous()
        else:
            loss = torch.zeros(3, device=self.device)  # box, cls, dfl
            pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
                (self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        # targets
        if self.with_angle:
            targets = torch.cat(
                (
                    batch['batch_idx'].view(-1, 1), 
                    batch['cls'].view(-1, 1), 
                    batch['bboxes'],
                    torch.cat(
                        self.transAngle_arr_tensor(batch['angles'][:, 0].flatten()), 
                    1),
                    batch['angles'][:, [1]]
                ), 1)
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes, gt_angles_cls, gt_angles_valid = targets.split((1, 4, 9, 1), 2)  # cls, xyxy, angle
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

            _, target_bboxes, target_scores, target_angles, target_angles_valid, fg_mask, _ = self.assigner(
                pred_scores.detach().sigmoid(), 
                (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
                anchor_points * stride_tensor, 
                gt_labels, 
                gt_angles_cls,  
                gt_angles_valid, 
                gt_bboxes, 
                mask_gt
            )
        else:
            # targets
            targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1)
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

            _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
                pred_scores.detach().sigmoid(), 
                (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
                anchor_points * stride_tensor, 
                gt_labels, 
                gt_bboxes, 
                mask_gt
            )

        target_bboxes /= stride_tensor
        target_scores_sum = max(target_scores.sum(), 1)

        # cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # bbox loss
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                              target_scores_sum, fg_mask)
            
        # angle loss
        if self.with_angle and fg_mask.sum():
            target_angles_valid = torch.tile(target_angles_valid.to(torch.bool), [1, 1, 8])
            target_angles = target_angles.to(dtype)
            for i in range(9):
                loss[3] += self.bce(
                    pred_angle_feats[..., i*8: (i+1)*8][target_angles_valid], 
                    target_angles[..., i*8: (i+1)*8][target_angles_valid]
                    ).sum() / max(target_angles[..., i*8: (i+1)*8][target_angles_valid].sum(), 1)
            loss[3] *= self.hyp.ang

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)
    
    def transAngle_arr_tensor(self, ori_angles):
        
        device = ori_angles.device
        angle_inputs = ori_angles[ori_angles>=0]
        angles_classes_0 = 8*torch.ones(ori_angles.shape[0], dtype=torch.int64, requires_grad=False).to(device)
        angles_classes_1 = 8*torch.ones(ori_angles.shape[0], dtype=torch.int64, requires_grad=False).to(device)
        angles_classes_2 = 8*torch.ones(ori_angles.shape[0], dtype=torch.int64, requires_grad=False).to(device)
        angles_classes_3 = 8*torch.ones(ori_angles.shape[0], dtype=torch.int64, requires_grad=False).to(device)
        angles_classes_4 = 8*torch.ones(ori_angles.shape[0], dtype=torch.int64, requires_grad=False).to(device)
        angles_classes_5 = 8*torch.ones(ori_angles.shape[0], dtype=torch.int64, requires_grad=False).to(device)
        angles_classes_6 = 8*torch.ones(ori_angles.shape[0], dtype=torch.int64, requires_grad=False).to(device)
        angles_classes_7 = 8*torch.ones(ori_angles.shape[0], dtype=torch.int64, requires_grad=False).to(device)
        angles_classes_8 = 8*torch.ones(ori_angles.shape[0], dtype=torch.int64, requires_grad=False).to(device)

        if(angle_inputs.shape[0] != 0):
            tables = torch.Tensor(np.tile(self.table, (angle_inputs.shape[0], 1, 1))).to(device)
            distances = ((tables - angle_inputs[:, np.newaxis, np.newaxis])+180)%360 -180
            idx = torch.argmin(distances.abs(), axis=-1)
            angles_classes_0[ori_angles>=0] = idx[:, 0]
            angles_classes_1[ori_angles>=0] = idx[:, 1]
            angles_classes_2[ori_angles>=0] = idx[:, 2]
            angles_classes_3[ori_angles>=0] = idx[:, 3]
            angles_classes_4[ori_angles>=0] = idx[:, 4]
            angles_classes_5[ori_angles>=0] = idx[:, 5]
            angles_classes_6[ori_angles>=0] = idx[:, 6]
            angles_classes_7[ori_angles>=0] = idx[:, 7]
            angles_classes_8[ori_angles>=0] = idx[:, 8]
        angles_classes_0[ori_angles==-1] = -1
        angles_classes_1[ori_angles==-1] = -1
        angles_classes_2[ori_angles==-1] = -1
        angles_classes_3[ori_angles==-1] = -1
        angles_classes_4[ori_angles==-1] = -1
        angles_classes_5[ori_angles==-1] = -1
        angles_classes_6[ori_angles==-1] = -1
        angles_classes_7[ori_angles==-1] = -1
        angles_classes_8[ori_angles==-1] = -1
        
        return (
            angles_classes_0.reshape(-1, 1), 
            angles_classes_1.reshape(-1, 1), 
            angles_classes_2.reshape(-1, 1), 
            angles_classes_3.reshape(-1, 1), 
            angles_classes_4.reshape(-1, 1), 
            angles_classes_5.reshape(-1, 1), 
            angles_classes_6.reshape(-1, 1), 
            angles_classes_7.reshape(-1, 1), 
            angles_classes_8.reshape(-1, 1)
        )
