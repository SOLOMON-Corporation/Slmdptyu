import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from slmdptyu.ultralytics.ultralytics.yolo.utils.ops import (crop_mask,
    non_max_suppression, scale_boxes, xyxy2xywh, Profile, process_mask_native, 
    scale_coords, process_mask
)


def process_mask_upsample(protos, masks_in, bboxes, shape):
    """
    It takes the output of the mask head, and applies the mask to the bounding boxes. This produces masks of higher
    quality but is slower.

    Args:
      protos (torch.Tensor): [mask_dim, mask_h, mask_w]
      masks_in (torch.Tensor): [n, mask_dim], n is number of masks after nms
      bboxes (torch.Tensor): [n, 4], n is number of masks after nms
      shape (tuple): the size of the input image (h,w)

    Returns:
      (torch.Tensor): The upsampled masks.
    """
    c, mh, mw = protos.shape  # CHW
    masks = (masks_in @ protos.float().view(c, -1)).sigmoid().view(-1, mh, mw)
    masks = F.interpolate(masks[None], shape, mode='bilinear', align_corners=False)[0]  # CHW
    masks = crop_mask(masks, bboxes)  # CHW
    return masks


class DetectPostprocess:
    def __init__(self, conf=0.25, iou=0.5, agnostic_nms=False, max_det=100, nc= 6, with_angle=False):
        self.conf = conf
        self.iou = iou
        self.agnostic_nms = agnostic_nms
        self.max_det = max_det
        self.nc = nc
        self.with_angle = with_angle
        self.mi = 78 if self.with_angle else 6

    def batched_nms(
            self,
            prediction,
            conf_thres=0.25,
            iou_thres=0.5,
            agnostic=False,
            max_det=300,
            nc=80,
            nk=17,
            max_wh=7680,
            w_mask=False,
            w_keypoints=False,
            w_angle=False
    ):
        device = prediction.device
        bs = prediction.shape[0]  # batch size
        nc = nc or (prediction.shape[1] - 4)  # number of classes
        prediction = prediction.transpose(1, 2)
        if w_keypoints and w_angle:
            box, cls, ang, key = prediction.split((4, nc, 72, nk*3), dim=2)
        elif w_keypoints:
            box, cls, key = prediction.split((4, nc, nk*3), dim=2)
        elif w_mask and w_angle:
            box, cls, ang, mask = prediction.split((4, nc, 72, 32), dim=2)
        elif w_mask:
            box, cls, mask = prediction.split((4, nc, 32), dim=2)
        elif w_angle:
            box, cls, ang = prediction.split((4, nc, 72), dim=2)
        else:
            box, cls = prediction.split((4, nc), dim=2)

        scores, cls = cls.max(dim=2, keepdim=True)
        result = scores.topk(k=max_det, dim=1)
        
        # filter
        scores = result.values.squeeze(2)
        cls = torch.gather(cls, dim=1, index=result.indices).squeeze(2)
        box = torch.gather(box, dim=1, index=result.indices.expand(1, -1, 4))
        if w_mask:
            mask = torch.gather(mask, dim=1, index=result.indices.expand(1, -1, 32))
        if w_keypoints:
            key = torch.gather(key, dim=1, index=result.indices.expand(1, -1, nk*3))
        if w_angle:
            ang = torch.gather(ang, dim=1, index=result.indices.expand(1, -1, 72))

        # xywh2xyxy
        boxes = torch.cat((
            (box[..., [0]] - box[..., [2]]/2),  # x1
            (box[..., [1]] - box[..., [3]]/2),  # y1
            (box[..., [0]] + box[..., [2]]/2),  # x2
            (box[..., [1]] + box[..., [3]]/2)), # y2
            axis=2
        )

        # NMS
        batch_valid = torch.zeros((bs), dtype=torch.int64, device=device)
        batch_scores = torch.zeros((bs, max_det), dtype=torch.float32, device=device)
        batch_boxes = torch.zeros((bs, max_det, 4), dtype=torch.float32, device=device)
        batch_clsids = torch.zeros((bs, max_det), dtype=torch.float32, device=device)
        if w_mask:
            batch_masks = torch.zeros((bs, max_det, 32), dtype=torch.float32, device=device)
        if w_keypoints:
            batch_kpts = torch.zeros((bs, max_det, nk*3), dtype=torch.float32, device=device)
        if w_angle:
            batch_angles = torch.zeros((bs, max_det, 72), dtype=torch.float32, device=prediction.device)
        
        for i in range(bs):
            ind = torchvision.ops.nms(boxes[i], scores[i], iou_thres)
            score_filter = scores[i, ind] > conf_thres
            ind = ind[score_filter]
            valid = score_filter.sum()
            batch_valid[i] = valid
            batch_scores[i, :valid] = scores[i, ind]
            batch_boxes[i, :valid] = boxes[i, ind]
            batch_clsids[i, :valid] = cls[i, ind]
            if w_mask:
                batch_masks[i, :valid] = mask[i, ind]
            if w_keypoints:
                batch_kpts[i, :valid] = key[i, ind]
            if w_angle:
                batch_angles[i, :valid] = ang[i, ind]

        output = [batch_valid, batch_boxes, batch_scores, batch_clsids]
        if w_mask:
            output.append(batch_masks)
        if w_keypoints:
            output.append(batch_kpts)
        if w_angle:
            output.append(batch_angles)

        return output

    def nms_wo_thresh(
            self,
            prediction,
            conf_thres=0.25,
            iou_thres=0.5,
            agnostic=False,
            max_det=300,
            nc=80,
            nk=17,
            max_wh=7680,
            w_mask=False,
            w_keypoints=False,
            w_angle=False
    ):
        device = prediction.device
        bs = prediction.shape[0]  # batch size
        nc = nc or (prediction.shape[1] - 4)  # number of classes
        prediction = prediction.transpose(1, 2)
        if w_keypoints and w_mask and w_angle:
            box, cls, ang, mask, key = prediction.split((4, nc, 72, 32, nk*3), dim=2)
        elif w_keypoints and w_mask:
            box, cls, mask, key = prediction.split((4, nc, 32, nk*3), dim=2)
        elif w_keypoints and w_angle:
            box, cls, ang, key = prediction.split((4, nc, 72, nk*3), dim=2)
        elif w_keypoints:
            box, cls, key = prediction.split((4, nc, nk*3), dim=2)
        elif w_mask and w_angle:
            box, cls, ang, mask = prediction.split((4, nc, 72, 32), dim=2)
        elif w_mask:
            box, cls, mask = prediction.split((4, nc, 32), dim=2)
        elif w_angle:
            box, cls, ang = prediction.split((4, nc, 72), dim=2)
        else:
            box, cls = prediction.split((4, nc), dim=2)

        scores, cls = cls.max(dim=2)
        # xywh2xyxy
        boxes = torch.cat((
            (box[..., [0]] - box[..., [2]]/2),  # x1
            (box[..., [1]] - box[..., [3]]/2),  # y1
            (box[..., [0]] + box[..., [2]]/2),  # x2
            (box[..., [1]] + box[..., [3]]/2)), # y2
            axis=2
        )
        boxes, scores, cls, = boxes.squeeze(0), scores.squeeze(0), cls.squeeze(0)
        i = torchvision.ops.nms(boxes, scores, self.iou)
        output=[i, boxes[i], scores[i], cls[i]]
        if w_mask:
            masks = mask.squeeze(0)
            output.append(masks[i])
        if w_keypoints:
            kpts = key.squeeze(0)
            output.append(kpts[i])
        if w_angle:
            angles = ang.squeeze(0)
            output.append(angles[i])
            
        return output
        

    def decode_angle(self, ang_classes):
        ang_classes = ang_classes.to(torch.float32)
        degs = ang_classes*45. + torch.arange(0, 9).to(ang_classes.device)*5.
        y = torch.sin(degs/180*np.pi).sum(-1)
        x = torch.cos(degs/180*np.pi).sum(-1)
        angles = torch.atan(y/x)*180/np.pi + (x < 0)*180 + \
            torch.logical_and(y < 0, x > 0)*360
        return angles

    def postprocess(self, preds):
        output = self.batched_nms(
            preds,
            self.conf,
            self.iou,
            agnostic=self.agnostic_nms,
            max_det=self.max_det,
            nc=self.nc,
            w_angle=self.with_angle
        )
        if self.with_angle:
            valid, boxes, scores, clsids, angles = output
            angles = angles.reshape(-1, self.max_det, 9, 8).argmax(-1)
            angles = self.decode_angle(angles)
            return valid, boxes, scores, clsids, angles
        else:
            valid, boxes, scores, clsids = output
            return valid, boxes, scores, clsids

    def postprocess_single(self, preds):
        output = self.nms_wo_thresh(
            preds,
            self.conf,
            self.iou,
            agnostic=self.agnostic_nms,
            max_det=self.max_det,
            nc=self.nc,
            w_angle=self.with_angle
        )
        if self.with_angle:
            valid, boxes, scores, clsids, angles = output
            # angles = angles.reshape(-1, 9, 8).argmax(-1)
            # angles = self.decode_angle(angles)
            return valid, boxes, scores, clsids, angles
        else:
            valid, boxes, scores, clsids = output
            return valid, boxes, scores, clsids


class SegmentPostprocess(DetectPostprocess):
    def crop_mask(self, masks, boxes):
        b, n, h, w = masks.shape
        x1, y1, x2, y2 = torch.chunk(boxes[..., None], 4, 2)  # x1 shape(1,1,n)
        r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, None, :]  # rows shape(1,w,1)
        c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, None, :, None]  # cols shape(h,1,1)
        r_valid = torch.logical_and((r >= x1), (r < x2))
        c_valid = torch.logical_and((c >= y1), (c < y2))
        m_valid = torch.logical_and(r_valid, c_valid)
        return torch.where(m_valid, masks, torch.zeros_like(masks))

    def process_mask(self, protos, masks_in, bboxes, shape):
        _, _, mh, mw = protos.shape  # CHW
        ih, iw = shape
        masks = torch.einsum("bkm,bmij->bkij", masks_in, protos.float()).sigmoid()

        downsampled_bboxes = bboxes.clone()
        downsampled_bboxes[..., 0] *= mw / iw
        downsampled_bboxes[..., 2] *= mw / iw
        downsampled_bboxes[..., 3] *= mh / ih
        downsampled_bboxes[..., 1] *= mh / ih

        masks = self.crop_mask(masks, downsampled_bboxes)  # CHW
        return masks.gt_(0.5).float()

    def crop_mask_single(self, masks, boxes):
        n, h, w = masks.shape
        x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)  # x1 shape(1,1,n)
        r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]  # rows shape(1,w,1)
        c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]  # cols shape(h,1,1)
    
        return masks * ((r >= x1).float() * (r < x2).float() * (c >= y1).float() * (c < y2).float())

    def process_mask_single(self, protos, masks_in, bboxes, shape):
        c, mh, mw = protos.shape  # CHW
        ih, iw = shape
        masks = (masks_in @ protos.float().view(c, -1)).sigmoid().view(-1, mh, mw)

        # downsampled_bboxes = bboxes.clone()
        # downsampled_bboxes[..., 0] *= mw / iw
        # downsampled_bboxes[..., 2] *= mw / iw
        # downsampled_bboxes[..., 3] *= mh / ih
        # downsampled_bboxes[..., 1] *= mh / ih

        # masks = self.crop_mask_single(masks, downsampled_bboxes)  # CHW
        return masks

    def postprocess(
            self, 
            preds,  
            impreshape=None, 
        ):
        output = self.batched_nms(
            preds[0],
            self.conf,
            self.iou,
            agnostic=self.agnostic_nms,
            max_det=self.max_det,
            nc=self.nc,
            w_mask=True,
            w_angle=self.with_angle
        )
        proto = preds[1]
        if self.with_angle:
            valid, boxes, scores, clsids, masks, angles = output
            masks = self.process_mask(
                proto, 
                masks, 
                boxes, 
                impreshape
            )
            angles = angles.reshape(-1, self.max_det, 9, 8).argmax(-1)
            angles = self.decode_angle(angles)
            return valid, boxes, scores, clsids, masks, angles
        else:
            valid, boxes, scores, clsids, masks = output
            masks = self.process_mask(
                proto, 
                masks, 
                boxes, 
                impreshape 
            )
            return valid, boxes, scores, clsids, masks

    def postprocess_single(
            self, 
            preds,  
            impreshape=None, 
        ):
        output = self.nms_wo_thresh(
            preds[0],
            self.conf,
            self.iou,
            agnostic=self.agnostic_nms,
            max_det=self.max_det,
            nc=self.nc,
            w_mask=True,
            w_angle=self.with_angle
        )
        proto = preds[1][0]
        if self.with_angle:
            valid, boxes, scores, clsids, masks, angles = output
            # angles = angles.reshape(-1, 9, 8).argmax(-1)
            # angles = self.decode_angle(angles)
            masks = self.process_mask_single(
                proto, 
                masks, 
                boxes, 
                impreshape 
            )
            return valid, boxes, scores, clsids, masks, angles
        else:
            valid, boxes, scores, clsids, masks = output
            masks = self.process_mask_single(
                proto, 
                masks, 
                boxes, 
                impreshape 
            )
            return valid, boxes, scores, clsids, masks


class PosePostprocess(SegmentPostprocess):
    def __init__(
            self, conf=0.25, iou=0.5, agnostic_nms=False, 
            max_det=100, nc=6, nk=17, with_angle=False
        ):
        super().__init__(
            conf=conf, iou=iou, agnostic_nms=agnostic_nms, 
            max_det=max_det, nc=nc, with_angle=with_angle)
        self.nk = nk

    def postprocess(self, preds):
        output = self.batched_nms(
            preds,
            self.conf,
            self.iou,
            agnostic=self.agnostic_nms,
            max_det=self.max_det,
            nc=self.nc,
            nk=self.nk,
            w_mask=False,
            w_keypoints=True,
            w_angle=self.with_angle
        )
        if self.with_angle:
            valid, boxes, scores, clsids, kpts, angles = output
            angles = angles.reshape(-1, self.max_det, 9, 8).argmax(-1)
            angles = self.decode_angle(angles)
            return valid, boxes, scores, clsids, kpts, angles
        else:
            valid, boxes, scores, clsids, kpts = output
            return valid, boxes, scores, clsids, kpts

    def postprocess_single(self, preds):
        output = self.nms_wo_thresh(
            preds,
            self.conf,
            self.iou,
            agnostic=self.agnostic_nms,
            max_det=self.max_det,
            nc=self.nc,
            nk=self.nk,
            w_mask=False,
            w_keypoints=True,
            w_angle=self.with_angle
        )
        if self.with_angle:
            valid, boxes, scores, clsids, kpts, angles = output
            # angles = angles.reshape(-1, 9, 8).argmax(-1)
            # angles = self.decode_angle(angles)
            return valid, boxes, scores, clsids, kpts, angles
        else:
            valid, boxes, scores, clsids, kpts = output
            return valid, boxes, scores, clsids, kpts


class SegPosePostprocess(SegmentPostprocess):
    def __init__(
            self, conf=0.25, iou=0.5, agnostic_nms=False, 
            max_det=100, nc=6, nk=17, with_angle=False
        ):
        super().__init__(
            conf=conf, iou=iou, agnostic_nms=agnostic_nms, 
            max_det=max_det, nc=nc, with_angle=with_angle)
        self.nk = nk

    def postprocess_single(
            self, 
            preds,  
            impreshape=None, 
        ):
        output = self.nms_wo_thresh(
            preds[0],
            self.conf,
            self.iou,
            agnostic=self.agnostic_nms,
            max_det=self.max_det,
            nc=self.nc,
            nk=self.nk,
            w_mask=True,
            w_keypoints=True,
            w_angle=self.with_angle
        )
        proto = preds[1][0]
        if self.with_angle:
            valid, boxes, scores, clsids, masks, kpts, angles = output
            masks = self.process_mask_single(
                proto, 
                masks, 
                boxes, 
                impreshape 
            )
            # angles = angles.reshape(-1, 9, 8).argmax(-1)
            # angles = self.decode_angle(angles)
            return valid, boxes, scores, clsids, masks, kpts, angles
        else:
            valid, boxes, scores, clsids, masks, kpts = output
            masks = self.process_mask_single(
                proto, 
                masks, 
                boxes, 
                impreshape 
            )
            return valid, boxes, scores, clsids, masks, kpts
