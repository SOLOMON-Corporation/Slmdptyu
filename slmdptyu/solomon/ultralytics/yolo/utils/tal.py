# modify from ultralytics/yolo/utils/tal.py

import torch
# import torch.nn as nn
import torch.nn.functional as F

from slmdptyu.ultralytics.ultralytics.yolo.utils.checks import check_version
# from slmdptyu.ultralytics.ultralytics.yolo.utils.metrics import bbox_iou

# import numpy as np
from slmdptyu.ultralytics.ultralytics.yolo.utils.tal import TaskAlignedAssigner, select_highest_overlaps

TORCH_1_10 = check_version(torch.__version__, '1.10.0')


class TaskAlignedSolAssigner(TaskAlignedAssigner):

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_angles_cls, gt_angles_valid, gt_bboxes, mask_gt):
        """This code referenced to
           https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py

        Args:
            pd_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            pd_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            anc_points (Tensor): shape(num_total_anchors, 2)
            gt_labels (Tensor): shape(bs, n_max_boxes, 1)
            gt_bboxes (Tensor): shape(bs, n_max_boxes, 4)
            mask_gt (Tensor): shape(bs, n_max_boxes, 1)
        Returns:
            target_labels (Tensor): shape(bs, num_total_anchors)
            target_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            target_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            fg_mask (Tensor): shape(bs, num_total_anchors)
        """
        self.bs = pd_scores.size(0)
        self.n_max_boxes = gt_bboxes.size(1)

        if self.n_max_boxes == 0:
            device = gt_bboxes.device
            return (torch.full_like(pd_scores[..., 0], self.bg_idx).to(device), 
                    torch.zeros_like(pd_bboxes).to(device),
                    torch.zeros_like(pd_scores).to(device), 
                    torch.zeros_like(pd_bboxes[..., [0]*9]).to(device), 
                    torch.zeros_like(pd_scores[..., 0]).to(device),
                    torch.zeros_like(pd_scores[..., 0]).to(device))

        mask_pos, align_metric, overlaps = self.get_pos_mask(pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points,
                                                             mask_gt)

        target_gt_idx, fg_mask, mask_pos = select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes)

        # assigned target
        target_labels, target_bboxes, target_scores, target_angles, target_angles_valid = self.get_targets(
            gt_labels, gt_bboxes, gt_angles_cls, gt_angles_valid, target_gt_idx, fg_mask)

        # normalize
        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(axis=-1, keepdim=True)  # b, max_num_obj
        pos_overlaps = (overlaps * mask_pos).amax(axis=-1, keepdim=True)  # b, max_num_obj
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric
        target_angles = target_angles * norm_align_metric

        return target_labels, target_bboxes, target_scores, target_angles, target_angles_valid.bool(), fg_mask.bool(), target_gt_idx

    def get_targets(self, gt_labels, gt_bboxes, gt_angles_cls, gt_angles_valid, target_gt_idx, fg_mask):
        """
        Args:
            gt_labels: (b, max_num_obj, 1)
            gt_bboxes: (b, max_num_obj, 4)
            target_gt_idx: (b, h*w)
            fg_mask: (b, h*w)
        """

        # assigned target labels, (b, 1)
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes  # (b, h*w)
        target_labels = gt_labels.long().flatten()[target_gt_idx]  # (b, h*w)

        # assigned target boxes, (b, max_num_obj, 4) -> (b, h*w)
        target_bboxes = gt_bboxes.view(-1, 4)[target_gt_idx]

        # assigned target scores
        target_labels.clamp(0)
        target_scores = F.one_hot(target_labels, self.num_classes)  # (b, h*w, 80)
        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)  # (b, h*w, 80)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)

        # assigned target angles
        target_angles_tmp = []
        fg_angles_mask = fg_mask[:, :, None].repeat(1, 1, 8)
        for i in range(9):
            target_angles = gt_angles_cls[..., i].long().flatten()[target_gt_idx]
            target_angles = F.one_hot(target_angles, 8)
            target_angles = torch.where(fg_angles_mask > 0, target_angles, 0)
            target_angles_tmp.append(target_angles)
        target_angles = torch.cat(target_angles_tmp, -1)

        target_angles_valid = gt_angles_valid.view(-1, 1)[target_gt_idx]

        return target_labels, target_bboxes, target_scores, target_angles, target_angles_valid
    