# modify from ultralytics/nn/modules.py
"""
Common modules
"""

import math

import torch
import torch.nn as nn

from slmdptyu.ultralytics.ultralytics.yolo.utils.tal import dist2bbox, make_anchors
from slmdptyu.solomon.ultralytics.yolo.utils.ops import (
    DetectPostprocess, SegmentPostprocess, PosePostprocess, SegPosePostprocess)

from slmdptyu.ultralytics.ultralytics.nn.modules import Proto, Conv, DFL


# Model heads below ----------------------------------------------------------------------------------------------------


class DetectSol(nn.Module):
    # YOLOv8 Detect head for detection models
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init
    w_onnx_end2end2_single=False
    def __init__(self, nc=80, with_angle=False, ch=()):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        self.with_angle = with_angle
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], self.nc)  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        if self.with_angle:
            c6 = max(ch[0], 8) 
            self.cv6 = nn.ModuleList(
                nn.Sequential(Conv(x, c6, 3), Conv(c6, c6, 3), nn.Conv2d(c6, 72, 1)) for x in ch)

    def forward(self, x):
        shape = x[0].shape  # BCHW
        if self.with_angle:
            for i in range(self.nl):
                x[i] = torch.cat(
                    (
                        self.cv2[i](x[i]), 
                        self.cv3[i](x[i]), 
                        self.cv6[i](x[i])
                    ), 1)
        else:
            for i in range(self.nl):
                x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.with_angle:
            x_cat = torch.cat([xi.view(shape[0], self.no+72, -1) for xi in x], 2)
            if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):  # avoid TF FlexSplitV ops
                box = x_cat[:, :self.reg_max * 4]
                cls = x_cat[:, self.reg_max * 4:-72]
                ang = x_cat[:, -72:]
            else:
                box, cls_ang = x_cat.split((self.reg_max * 4, self.nc+72), 1)
                cls, ang = cls_ang.split((self.nc, 72), 1)
            dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
            y = torch.cat((dbox, cls.sigmoid(), ang), 1)

        else:
            x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
            if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):  # avoid TF FlexSplitV ops
                box = x_cat[:, :self.reg_max * 4]
                cls = x_cat[:, self.reg_max * 4:]
            else:
                box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
            dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
            y = torch.cat((dbox, cls.sigmoid()), 1)
            
        if self.export and self.w_onnx_end2end2_single:
            if isinstance(self, (PoseSol, SegmentSol, SegPoseSol)):
                return y
            else:
                postprocess = DetectPostprocess(
                    conf=0.25, 
                    iou=0.5,
                    nc=self.nc, 
                    with_angle=self.with_angle
                )
                return postprocess.postprocess_single(y)
        elif self.export:
            return y 
        else:
            return (y, x)

    def bias_init(self):
        # Initialize Detect() biases, WARNING: requires stride availability
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        
        if self.with_angle:
            for a, b, ang, s in zip(m.cv2, m.cv3, m.cv6, m.stride):  # from
                a[-1].bias.data[:] = 1.0  # box
                b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)
                ang[-1].bias.data[:72] = math.log(5 / 8 / (640 / s) ** 2)
        else:
            for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
                a[-1].bias.data[:] = 1.0  # box
                b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)


class SegmentSol(DetectSol):
    # YOLOv8 Segment head for segmentation models
    def __init__(self, nc=80, nm=32, npr=256, with_angle=False, ch=()):
        super().__init__(nc, with_angle, ch)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos
        self.detect = DetectSol.forward

        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch)

    def forward(self, x):
        p = self.proto(x[0])  # mask protos
        bs = p.shape[0]  # batch size

        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)  # mask coefficients
        x = self.detect(self, x)
        if self.training:
            return x, mc, p
        elif self.export and self.w_onnx_end2end2_single:
            pred_output = (torch.cat([x, mc], 1), p)
            impreshape = [p.shape[2]*4,p.shape[3]*4]
            postprocess = SegmentPostprocess(
                conf=0.25, 
                iou=0.5,
                nc=self.nc, 
                with_angle=self.with_angle
            )
            return postprocess.postprocess_single(
                pred_output, 
                impreshape=impreshape
            )
        elif self.export:
            return (torch.cat([x, mc], 1), p)
        else:
            return (torch.cat([x[0], mc], 1), (x[1], mc, p))


class PoseSol(DetectSol):
    # YOLOv8 Pose head for keypoints models
    def __init__(self, nc=80, kpt_shape=(17, 3), with_angle=False, ch=()):
        super().__init__(nc, with_angle, ch)
        self.kpt_shape = kpt_shape  # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
        self.nk = kpt_shape[0] * kpt_shape[1]  # number of keypoints total
        self.detect = DetectSol.forward

        c4 = max(ch[0] // 4, self.nk)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)) for x in ch)

    def forward(self, x):
        bs = x[0].shape[0]  # batch size
        kpt = torch.cat([self.cv4[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], -1)  # (bs, 17*3, h*w)
        x = self.detect(self, x)
        if self.training:
            return x, kpt
        pred_kpt = self.kpts_decode(kpt)
        
        if self.export and self.w_onnx_end2end2_single:
            postprocess = PosePostprocess(
                conf=0.25, 
                iou=0.5,
                nc=self.nc,
                nk=self.nk//3, 
                with_angle=self.with_angle
            )
            return postprocess.postprocess_single(torch.cat([x, pred_kpt], 1))
        elif self.export:
            return torch.cat([x, pred_kpt], 1)
        else: 
            return (torch.cat([x[0], pred_kpt], 1), (x[1], kpt))

    def kpts_decode(self, kpts):
        ndim = self.kpt_shape[1]
        y = kpts.clone()
        if ndim == 3:
            y[:, 2::3].sigmoid_()  # inplace sigmoid
        y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
        y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
        return y


class SegPoseSol(DetectSol):
    # YOLOv8 Pose head for keypoints models
    def __init__(self, nc=80, nm=32, npr=256, kpt_shape=(17, 3), with_angle=False, ch=()):
        super().__init__(nc, with_angle, ch)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos
        self.kpt_shape = kpt_shape  # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
        self.nk = kpt_shape[0] * kpt_shape[1]  # number of keypoints total
        self.detect = DetectSol.forward

        c4 = max(ch[0] // 4, self.nk)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)) for x in ch)
        c5 = max(ch[0] // 4, self.nm)
        self.cv5 = nn.ModuleList(nn.Sequential(Conv(x, c5, 3), Conv(c5, c5, 3), nn.Conv2d(c5, self.nm, 1)) for x in ch)

    def forward(self, x):
        p = self.proto(x[0])  # mask protos
        bs = x[0].shape[0]  # batch size
        kpt = torch.cat([self.cv4[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], -1)  # (bs, 17*3, h*w)
        mc = torch.cat([self.cv5[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)  # mask coefficients
        x = self.detect(self, x)
        if self.training:
            return x, kpt, mc, p
        pred_kpt = self.kpts_decode(kpt)

        if self.export and self.w_onnx_end2end2_single:
            pred_output = (torch.cat([x, mc, pred_kpt], 1), p)
            impreshape = [p.shape[2]*4,p.shape[3]*4]
            postprocess = SegPosePostprocess(
                conf=0.25, 
                iou=0.5,
                nc=self.nc,
                nk=self.nk//3, 
                with_angle=self.with_angle
            )
            return postprocess.postprocess_single(
                pred_output, 
                impreshape=impreshape
            )
        elif self.export:
            return (torch.cat([x, mc, pred_kpt], 1), p)
        else: 
            return (torch.cat([x[0], mc, pred_kpt], 1), (x[1], kpt, mc, p))

    def kpts_decode(self, kpts):
        ndim = self.kpt_shape[1]
        y = kpts.clone()
        if ndim == 3:
            y[:, 2::3].sigmoid_()  # inplace sigmoid
        y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
        y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
        return y


class Gaugev2Sol(DetectSol):
    # YOLOv8 Pose head for keypoints models
    def __init__(self, nc=80, pointer_shape=(1, 3), scaler_shape = (1, 10), 
                 pointer_idx=[True, False, False, False, False, False, False, False, False, False, False], 
                 with_angle=False, ch=()):
        super().__init__(nc, with_angle, ch)
        self.pointer_shape = pointer_shape  
        self.scaler_shape = scaler_shape
        self.kpt_shape = (self.pointer_shape[0] + self.scaler_shape[0], 3) # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
        self.np = pointer_shape[0] * pointer_shape[1]  
        self.ns = scaler_shape[0] * scaler_shape[1]
        self.nk = self.kpt_shape[0] * self.kpt_shape[1]   # number of keypoints total
        self.detect = DetectSol.forward

        c4 = max(ch[0] // 4, self.ns)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.ns, 1)) for x in ch)
        self.cv5 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.np, 1)) for x in ch)
        
        self.pointer_idx = torch.tensor(pointer_idx, device=self.cv4[0][0].conv.weight.device)
        
    def forward(self, x):
        bs = x[0].shape[0]  # batch size
        kpt_scaler = torch.cat([self.cv4[i](x[i]).view(bs, self.ns//3, 3, -1) for i in range(self.nl)], -1)  # (bs, 17*3, h*w)
        kpt_pointer = torch.cat([self.cv5[i](x[i]).view(bs, self.np//3, 3, -1) for i in range(self.nl)], -1)  # (bs, 17*3, h*w)
        
        # kpt = torch.zeros((bs, self.kpt_shape[0], self.kpt_shape[1], kpt_scaler.shape[-1]), device=kpt_scaler.device).type(kpt_pointer.type())
        # kpt[:, self.pointer_idx, :] = kpt_pointer
        # kpt[:, ~self.pointer_idx, :] = kpt_scaler
        # kpt = kpt.view(bs, self.nk, -1)
        kpt = torch.cat([kpt_pointer, kpt_scaler], 1).view(bs, self.nk, -1)

        x = self.detect(self, x)
        if self.training:
            return x, kpt
        pred_kpt = self.kpts_decode(kpt)
        
        if self.export and self.w_onnx_end2end2_single:
            postprocess = PosePostprocess(
                conf=0.25, 
                iou=0.5,
                nc=self.nc,
                nk=self.nk//3, 
                with_angle=self.with_angle
            )
            return postprocess.postprocess_single(torch.cat([x, pred_kpt], 1))
        elif self.export:
            return torch.cat([x, pred_kpt], 1)
        else: 
            return (torch.cat([x[0], pred_kpt], 1), (x[1], kpt))

    def kpts_decode(self, kpts):
        ndim = self.kpt_shape[1]
        y = kpts.clone()
        if ndim == 3:
            y[:, 2::3].sigmoid_()  # inplace sigmoid
        y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
        y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
        return y
