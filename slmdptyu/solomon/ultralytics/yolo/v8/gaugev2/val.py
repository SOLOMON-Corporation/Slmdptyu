from slmdptyu.solomon.ultralytics.yolo.utils.metrics import SolMetrics
from slmdptyu.solomon.ultralytics.yolo.v8.pose.val import PoseSolValidator
from slmdptyu.solomon.ultralytics.yolo.utils import DEFAULT_CFG

class Gaugev2SolValidator(PoseSolValidator):

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None):
        super().__init__(dataloader, save_dir, pbar, args)
        self.args.task = 'gaugev2'
        self.metrics = SolMetrics(save_dir=self.save_dir)


def val(cfg=DEFAULT_CFG, use_python=False):
    model = cfg.model or "yolov8n-seg.pt"
    data = cfg.data or "coco128-seg.yaml"

    args = dict(model=model, data=data)
    if use_python:
        from slmdptyu.solomon.ultralytics import YOLOSOL
        YOLOSOL(model).val(**args)
    else:
        validator = Gaugev2SolValidator(args=args)
        validator(model=args['model'])


if __name__ == "__main__":
    val()
