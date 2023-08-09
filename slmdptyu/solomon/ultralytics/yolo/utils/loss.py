import torch
import torch.nn as nn

class KeypointLoss(nn.Module):

    def forward(self, pred_kpts, gt_kpts, kpt_mask, diagonal):
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]) ** 2 + (pred_kpts[..., 1] - gt_kpts[..., 1]) ** 2
        kpt_loss_factor = (torch.sum(kpt_mask != 0) + torch.sum(kpt_mask == 0)) / (torch.sum(kpt_mask != 0) + 1e-9)
        # return kpt_loss_factor * (d / (diagonal*0.05) * kpt_mask).mean(dim=0).sum()
        return kpt_loss_factor * (d * kpt_mask).mean()