import torch
import torch.nn.functional as F
from copy import copy
from slmdptyu.solomon.ultralytics.yolo.v8.detect.train import DetectionSolTrainer
from slmdptyu.solomon.ultralytics.yolo.v8.segment.val import SegmentationSolValidator
from slmdptyu.solomon.ultralytics.yolo.utils import RANK
from slmdptyu.ultralytics.ultralytics.yolo.utils.ops import crop_mask, xyxy2xywh
from slmdptyu.ultralytics.ultralytics.yolo.utils.tal import make_anchors
from slmdptyu.ultralytics.ultralytics.yolo.utils.torch_utils import de_parallel
from slmdptyu.solomon.ultralytics.yolo.v8.detect.train import Loss
from slmdptyu.solomon.ultralytics.yolo.utils import DEFAULT_CFG
from slmdptyu.solomon.ultralytics.nn.tasks import SegmentationSolModel

# BaseTrainer python usage
class SegmentationSolTrainer(DetectionSolTrainer):

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        if overrides is None:
            overrides = {}
        if not overrides['train_keypoints']:
            overrides['task'] = 'segment'
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = SegmentationSolModel(
            cfg, ch=3, nc=self.data['nc'], 
            with_angle=self.data['with_angle'], 
            verbose=verbose and RANK == -1
        )
        if weights:
            model.load(weights)

        return model

    def get_validator(self):
        if self.args.train_angle:
            self.loss_names = 'box_loss', 'seg_loss', 'cls_loss', 'dfl_loss', 'ang_loss'
        else:
            self.loss_names = 'box_loss', 'seg_loss', 'cls_loss', 'dfl_loss'
        return SegmentationSolValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))

    def criterion(self, preds, batch):
        if not hasattr(self, 'compute_loss'):
            self.compute_loss = SegLoss(de_parallel(self.model), overlap=self.args.overlap_mask)
        return self.compute_loss(preds, batch)


# Criterion class for computing training losses
class SegLoss(Loss):

    def __init__(self, model, overlap=True):  # model must be de-paralleled
        super().__init__(model)
        self.nm = model.model[-1].nm  # number of masks
        self.overlap = overlap

    def __call__(self, preds, batch):
        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]

        if self.with_angle:
            loss = torch.zeros(5, device=self.device)  # box, seg, cls, dfl
            pred_distri, pred_scores, pred_angle_feats = torch.cat([xi.view(feats[0].shape[0], self.no+72, -1) for xi in feats], 2).split(
                (self.reg_max * 4, self.nc, 72), 1)
            pred_angle_feats = pred_angle_feats.permute(0, 2, 1).contiguous()
        else:
            loss = torch.zeros(4, device=self.device)  # box, seg, cls, dfl
            pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
                (self.reg_max * 4, self.nc), 1)
        
        batch_size, _, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width

        # b, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_masks = pred_masks.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        batch_idx = batch['batch_idx'].view(-1, 1)

        # pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        # targets
        try:
            if self.with_angle:
                targets = torch.cat(
                    (
                        batch_idx, 
                        batch['cls'].view(-1, 1), 
                        batch['bboxes'].to(dtype),
                        torch.cat(
                            self.transAngle_arr_tensor(batch['angles'][:, 0].flatten()), 
                        1),
                        batch['angles'][:, [1]]
                    ), 1)
                targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
                gt_labels, gt_bboxes, gt_angles, gt_angles_valid = targets.split((1, 4, 9, 1), 2)  # cls, xyxy, angle
                mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

                _, target_bboxes, target_scores, target_angles, target_angles_valid, fg_mask, target_gt_idx = self.assigner(
                    pred_scores.detach().sigmoid(), 
                    (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
                    anchor_points * stride_tensor, 
                    gt_labels, 
                    gt_angles, 
                    gt_angles_valid, 
                    gt_bboxes, 
                    mask_gt
                )
            else:
                targets = torch.cat((batch_idx, batch['cls'].view(-1, 1), batch['bboxes'].to(dtype)), 1)
                targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
                gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
                mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

                _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
                    pred_scores.detach().sigmoid(), 
                    (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
                    anchor_points * stride_tensor, 
                    gt_labels, 
                    gt_bboxes, 
                    mask_gt
                )
        except RuntimeError as e:
            raise TypeError('ERROR ❌ segment dataset incorrectly formatted or not a segment dataset.\n'
                            "This error can occur when incorrectly training a 'segment' model on a 'detect' dataset, "
                            "i.e. 'yolo train model=yolov8n-seg.pt data=coco128.yaml'.\nVerify your dataset is a "
                            "correctly formatted 'segment' dataset using 'data=coco128-seg.yaml' "
                            'as an example.\nSee https://docs.ultralytics.com/tasks/segment/ for help.') from e

        target_scores_sum = max(target_scores.sum(), 1)

        # cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[2] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # bbox loss
        if fg_mask.sum():
            # bbox loss
            loss[0], loss[3] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes / stride_tensor,
                                              target_scores, target_scores_sum, fg_mask)
            # masks loss
            masks = batch['masks'].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode='nearest')[0]

            for i in range(batch_size):
                if fg_mask[i].sum():
                    mask_idx = target_gt_idx[i][fg_mask[i]]
                    if self.overlap:
                        gt_mask = torch.where(masks[[i]] == (mask_idx + 1).view(-1, 1, 1), 1.0, 0.0)
                    else:
                        gt_mask = masks[batch_idx.view(-1) == i][mask_idx]
                    xyxyn = target_bboxes[i][fg_mask[i]] / imgsz[[1, 0, 1, 0]]
                    marea = xyxy2xywh(xyxyn)[:, 2:].prod(1)
                    mxyxy = xyxyn * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=self.device)
                    loss[1] += self.single_mask_loss(gt_mask, pred_masks[i][fg_mask[i]], proto[i], mxyxy, marea)  # seg
                # WARNING: lines below prevents Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
                else:
                    loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        # WARNING: lines below prevent Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        if self.with_angle and fg_mask.sum():
            target_angles_valid = torch.tile(target_angles_valid.to(torch.bool), [1, 1, 8])
            target_angles = target_angles.to(dtype)
            for i in range(9):
                loss[4] += self.bce(
                    pred_angle_feats[..., i*8: (i+1)*8][target_angles_valid], 
                    target_angles[..., i*8: (i+1)*8][target_angles_valid]
                    ).sum() / max(target_angles[..., i*8: (i+1)*8][target_angles_valid].sum(), 1)
            loss[4] *= self.hyp.ang


        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.box / batch_size  # seg gain
        loss[2] *= self.hyp.cls  # cls gain
        loss[3] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    def single_mask_loss(self, gt_mask, pred, proto, xyxy, area):
        # Mask loss for one image
        pred_mask = (pred @ proto.view(self.nm, -1)).view(-1, *proto.shape[1:])  # (n, 32) @ (32,80,80) -> (n,80,80)
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).mean()


def train(cfg=DEFAULT_CFG, use_python=False):
    model = cfg.model or 'yolov8n-seg.pt'
    data = cfg.data or "coco128-seg.yaml"  # or yolo.ClassificationDataset("mnist")
    device = cfg.device if cfg.device is not None else ''

    args = dict(model=model, data=data, device=device)
    if use_python:
        from slmdptyu.solomon.ultralytics import YOLOSOL
        YOLOSOL(model).train(**args)
    else:
        trainer = SegmentationSolTrainer(overrides=args)
        trainer.train()


if __name__ == '__main__':
    train()

