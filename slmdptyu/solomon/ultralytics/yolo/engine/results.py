import torch
from copy import deepcopy
from slmdptyu.ultralytics.ultralytics.yolo.data.augment import LetterBox
from slmdptyu.solomon.ultralytics.yolo.utils import deprecation_warn 
from slmdptyu.ultralytics.ultralytics.yolo.utils.plotting import colors

from slmdptyu.ultralytics.ultralytics.yolo.engine.results import Results, Boxes, Masks

class ResultsSol(Results):
    def __init__(self, orig_img, path, names, boxes=None, masks=None, probs=None, keypoints=None, angles=None) -> None:
        self.orig_img = orig_img
        self.orig_shape = orig_img.shape[:2]
        self.boxes = Boxes(boxes, self.orig_shape) if boxes is not None else None  # native size boxes
        self.masks = MasksSol(masks, self.orig_shape, self.boxes) if masks is not None else None  # native size or imgsz masks
        self.keypoints = keypoints if keypoints is not None else None
        self.speed = {'preprocess': None, 'inference': None, 'postprocess': None}  # milliseconds per image
        self.angles = angles if angles is not None else None
        self.probs = probs if probs is not None else None
        self.names = names
        self.path = path
        self._keys = ('boxes', 'masks', 'probs', 'keypoints', 'angles')

    def new(self):
        return ResultsSol(orig_img=self.orig_img, path=self.path, names=self.names)


class MasksSol(Masks):
    def __init__(self, masks, orig_shape, boxes:Boxes) -> None:
        super().__init__(masks, orig_shape)
        self.boxes = boxes

    def croped(self):
        return [
            mask[box[1]:box[3], box[0]:box[2]]*0.6
            for box, mask 
            in zip(
                self.boxes.xyxy.cpu().numpy().astype(int), 
                self.masks.cpu().numpy()
            )
        ]

