import numpy as np
from typing import List
from slmdptyu.ultralytics.ultralytics.yolo.utils.instance import Instances

class InstancesSol(Instances):
    def __init__(
        self, 
        bboxes, 
        segments=None, 
        keypoints=None,
        angles=None,
        bbox_format='xywh', 
        normalized=True) -> None:
        super().__init__(
            bboxes=bboxes, 
            segments=segments, 
            bbox_format=bbox_format,
            normalized=normalized)

        self.keypoints = keypoints
        self.angles = angles

    def scale(self, scale_w, scale_h, bbox_only=False):
        super().scale(scale_w, scale_h, bbox_only)
        if bbox_only:
            return
        if self.angles is not None:
            self.angles[..., 0] *= scale_w
            self.angles[..., 1] *= scale_h

    def denormalize(self, w, h):
        if not self.normalized:
            return
        super().denormalize(w, h)
        if self.angles is not None:
            self.angles[..., 0] *= w
            self.angles[..., 1] *= h

    def normalize(self, w, h):
        if self.normalized:
            return
        super().normalize(w, h)
        if self.angles is not None:
            self.angles[..., 0] /= w
            self.angles[..., 1] /= h

    def add_padding(self, padw, padh):
        super().add_padding(padw, padh)
        if self.angles is not None:
            self.angles[..., 0] += padw
            self.angles[..., 1] += padh

    def __getitem__(self, index) -> 'InstancesSol':
        """
        Args:
            index: int, slice, or a BoolArray

        Returns:
            Instances: Create a new :class:`Instances` by indexing.
        """
        segments = self.segments[index] if len(self.segments) else self.segments
        keypoints = self.keypoints[index] if self.keypoints is not None else None
        angles = self.angles[index] if self.angles is not None else None
        bboxes = self.bboxes[index]
        bbox_format = self._bboxes.format
        return InstancesSol(
            bboxes=bboxes,
            segments=segments,
            keypoints=keypoints,
            angles=angles,
            bbox_format=bbox_format,
            normalized=self.normalized,
        )

    def flipud(self, h):
        super().flipud(h)
        if self.angles is not None:
            self.angles[..., 1] = h - self.angles[..., 1]

    def fliplr(self, w):
        super().fliplr(w)
        if self.angles is not None:
            self.angles[..., 0] = w - self.angles[..., 0]

    def update(
        self, 
        bboxes, 
        segments=None, 
        keypoints=None,
        angles=None,
        ):
        super().update(bboxes, segments=segments, keypoints=keypoints)
        if angles is not None:
            self.angles = angles

    @classmethod
    def concatenate(cls, instances_list: List["InstancesSol"], axis=0) -> "InstancesSol":
        """
        Concatenates a list of Boxes into a single Bboxes

        Arguments:
            instances_list (list[Bboxes])
            axis

        Returns:
            Boxes: the concatenated Boxes
        """
        assert isinstance(instances_list, (list, tuple))
        if not instances_list:
            return cls(np.empty(0))
        assert all(isinstance(instance, InstancesSol) for instance in instances_list)

        if len(instances_list) == 1:
            return instances_list[0]

        use_keypoint = instances_list[0].keypoints is not None
        use_angle = instances_list[0].angles is not None
        bbox_format = instances_list[0]._bboxes.format
        normalized = instances_list[0].normalized

        cat_boxes = np.concatenate([ins.bboxes for ins in instances_list], axis=axis)
        cat_segments = np.concatenate([b.segments for b in instances_list], axis=axis)
        cat_keypoints = np.concatenate([b.keypoints for b in instances_list], axis=axis) if use_keypoint else None
        cat_angles = np.concatenate([b.angles for b in instances_list], axis=axis) if use_angle else None

        return cls(
            cat_boxes, 
            cat_segments, 
            cat_keypoints,
            cat_angles, 
            bbox_format, 
            normalized)
