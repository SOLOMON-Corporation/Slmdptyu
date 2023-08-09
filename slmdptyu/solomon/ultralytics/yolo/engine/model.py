import sys
from pathlib import Path

from slmdptyu.solomon.ultralytics.nn.tasks import attempt_load_one_weight, yaml_model_load
from slmdptyu.ultralytics.ultralytics.yolo.cfg import get_cfg
from slmdptyu.solomon.ultralytics.yolo.engine.exporter import ExporterSol
from slmdptyu.solomon.ultralytics.yolo.utils import LOGGER, RANK, ROOT, is_git_dir, yaml_load
from slmdptyu.solomon.ultralytics.yolo.utils import DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS, DEFAULT_CFG
from slmdptyu.ultralytics.ultralytics.yolo.utils.checks import check_imgsz
from slmdptyu.ultralytics.ultralytics.yolo.utils.torch_utils import smart_inference_mode

import os
import cv2
from pycocotools.coco import COCO
from slmdptyu.ultralytics.ultralytics import YOLO
from slmdptyu.solomon.ultralytics.nn.tasks import DetectionSolModel, SegmentationSolModel, PoseSolModel, SegPoseSolModel, Gaugev2SolModel, guess_model_task
from slmdptyu.solomon.ultralytics.yolo.v8.detect.train import DetectionSolTrainer
from slmdptyu.solomon.ultralytics.yolo.v8.detect.val import DetectionSolValidator
from slmdptyu.solomon.ultralytics.yolo.v8.detect.predict import DetectionSolPredictor
from slmdptyu.solomon.ultralytics.yolo.v8.segment.train import SegmentationSolTrainer
from slmdptyu.solomon.ultralytics.yolo.v8.segment.val import SegmentationSolValidator
from slmdptyu.solomon.ultralytics.yolo.v8.segment.predict import SegmentationSolPredictor
from slmdptyu.solomon.ultralytics.yolo.v8.pose.train import PoseSolTrainer
from slmdptyu.solomon.ultralytics.yolo.v8.pose.val import PoseSolValidator
from slmdptyu.solomon.ultralytics.yolo.v8.pose.predict import PoseSolPredictor
from slmdptyu.solomon.ultralytics.yolo.v8.segpose.train import SegPoseSolTrainer
from slmdptyu.solomon.ultralytics.yolo.v8.segpose.val import SegPoseSolValidator
from slmdptyu.solomon.ultralytics.yolo.v8.segpose.predict import SegPoseSolPredictor
from slmdptyu.solomon.ultralytics.yolo.v8.gaugev2.train import Gaugev2SolTrainer
from slmdptyu.solomon.ultralytics.yolo.v8.gaugev2.val import Gaugev2SolValidator
from slmdptyu.solomon.ultralytics.yolo.v8.gaugev2.predict import Gaugev2SolPredictor
from slmdptyu.ultralytics.ultralytics.yolo.data.augment import LetterBox


# Map head to model, trainer, validator, and predictor classes
TASK_MAP = {
    'detect': [
        DetectionSolModel,
        DetectionSolTrainer,
        DetectionSolValidator,
        DetectionSolPredictor],
    'segment': [
        SegmentationSolModel,
        SegmentationSolTrainer,
        SegmentationSolValidator,
        SegmentationSolPredictor],
    'pose': [
        PoseSolModel,
        PoseSolTrainer,
        PoseSolValidator,
        PoseSolPredictor],
    'segpose': [
        SegPoseSolModel,
        SegPoseSolTrainer,
        SegPoseSolValidator,
        SegPoseSolPredictor],
    'gaugev2': [
        Gaugev2SolModel,
        Gaugev2SolTrainer,
        Gaugev2SolValidator,
        Gaugev2SolPredictor],
}


class YOLOSOL(YOLO):
    """
    YOLO (You Only Look Once) object detection model.

    Args:
        model (str, Path): Path to the model file to load or create.

    Attributes:
        predictor (Any): The predictor object.
        model (Any): The model object.
        trainer (Any): The trainer object.
        task (str): The type of model task.
        ckpt (Any): The checkpoint object if the model loaded from *.pt file.
        cfg (str): The model configuration if loaded from *.yaml file.
        ckpt_path (str): The checkpoint file path.
        overrides (dict): Overrides for the trainer object.
        metrics (Any): The data for metrics.

    Methods:
        __call__(source=None, stream=False, **kwargs):
            Alias for the predict method.
        _new(cfg:str, verbose:bool=True) -> None:
            Initializes a new model and infers the task type from the model definitions.
        _load(weights:str, task:str='') -> None:
            Initializes a new model and infers the task type from the model head.
        _check_is_pytorch_model() -> None:
            Raises TypeError if the model is not a PyTorch model.
        reset() -> None:
            Resets the model modules.
        info(verbose:bool=False) -> None:
            Logs the model info.
        fuse() -> None:
            Fuses the model for faster inference.
        predict(source=None, stream=False, **kwargs) -> List[ultralytics.yolo.engine.results.Results]:
            Performs prediction using the YOLO model.

    Returns:
        list(ultralytics.yolo.engine.results.Results): The prediction results.
    """

    def _new(self, cfg: str, task=None, verbose=True):
        """
        Initializes a new model and infers the task type from the model definitions.

        Args:
            cfg (str): model configuration file
            task (str) or (None): model task
            verbose (bool): display model info on load
        """
        cfg_dict = yaml_model_load(cfg)
        self.cfg = cfg
        self.task = task or guess_model_task(cfg_dict)
        self.model = TASK_MAP[self.task][0](cfg_dict, verbose=verbose and RANK == -1)  # build model
        self.overrides['model'] = self.cfg

        # Below added to allow export from yamls
        args = {**DEFAULT_CFG_DICT, **self.overrides}  # combine model and default args, preferring model args
        self.model.args = {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS}  # attach args to model
        self.model.task = self.task

    def _load(self, weights: str, task=None):
        """
        Initializes a new model and infers the task type from the model head.

        Args:
            weights (str): model checkpoint to be loaded
            task (str) or (None): model task
        """
        suffix = Path(weights).suffix
        if suffix in ['.pth', '.pt']:
            self.model, self.ckpt = attempt_load_one_weight(weights)
            self.task = self.model.args['task']
            self.overrides = self.model.args = self._reset_ckpt_args(self.model.args)
            self.ckpt_path = self.model.pt_path
        else:
            self.model, self.ckpt = weights, None
            self.task = task or guess_model_task(weights)
            self.ckpt_path = weights
        self.overrides['model'] = weights
        self.overrides['task'] = self.task

    @smart_inference_mode()
    def predict(self, source=None, stream=False, **kwargs):
        """
        Perform prediction using the YOLO model.

        Args:
            source (str | int | PIL | np.ndarray): The source of the image to make predictions on.
                          Accepts all source types accepted by the YOLO model.
            stream (bool): Whether to stream the predictions or not. Defaults to False.
            **kwargs : Additional keyword arguments passed to the predictor.
                       Check the 'configuration' section in the documentation for all available options.

        Returns:
            (List[ultralytics.yolo.engine.results.Results]): The prediction results.
        """
        if source is None:
            source = ROOT / 'assets' if is_git_dir() else 'https://ultralytics.com/images/bus.jpg'
            LOGGER.warning(f"WARNING ⚠️ 'source' is missing. Using 'source={source}'.")
        is_cli = (sys.argv[0].endswith('yolo') or sys.argv[0].endswith('ultralytics')) and any(
            x in sys.argv for x in ('predict', 'track', 'mode=predict', 'mode=track'))
        overrides = self.overrides.copy()
        overrides['conf'] = 0.25
        overrides.update(kwargs)  # prefer kwargs
        overrides['mode'] = kwargs.get('mode', 'predict')
        assert overrides['mode'] in ['track', 'predict']
        if not is_cli:
            overrides['save'] = kwargs.get('save', False)  # do not save by default if called in Python
        if not self.predictor:
            self.task = overrides.get('task') or self.task
            self.predictor = TASK_MAP[self.task][3](overrides=overrides, _callbacks=self.callbacks)
            self.predictor.setup_model(model=self.model, verbose=is_cli)
        else:  # only update args if predictor is already setup
            self.predictor.args = get_cfg(self.predictor.args, overrides)
        return self.predictor.predict_cli(source=source) if is_cli else self.predictor(source=source, stream=stream)

    @smart_inference_mode()
    def val(self, data=None, **kwargs):
        """
        Validate a model on a given dataset .

        Args:
            data (str): The dataset to validate on. Accepts all formats accepted by yolo
            **kwargs : Any other args accepted by the validators. To see all args check 'configuration' section in docs
        """
        overrides = self.overrides.copy()
        overrides['rect'] = True  # rect batches as default
        overrides.update(kwargs)
        overrides['mode'] = 'val'
        args = get_cfg(cfg=DEFAULT_CFG, overrides=overrides)
        args.data = data or args.data
        if 'task' in overrides:
            self.task = args.task
        else:
            args.task = self.task
        if args.imgsz == DEFAULT_CFG.imgsz and not isinstance(self.model, (str, Path)):
            args.imgsz = self.model.args['imgsz']  # use trained imgsz unless custom value is passed
        args.imgsz = check_imgsz(args.imgsz, max_dim=1)

        validator = TASK_MAP[self.task][2](args=args, save_dir=Path(args.save_dir), _callbacks=self.callbacks)
        validator(model=self.model)
        self.metrics_data = validator.metrics

        return validator.metrics

    def export(self, **kwargs):
        """
        Export model.
        Args:
            **kwargs : Any other args accepted by the predictors. To see all args check 'configuration' section in docs
        """
        self._check_is_pytorch_model()
        overrides = self.overrides.copy()
        overrides.update(kwargs)
        overrides['mode'] = 'export'
        args = get_cfg(cfg=DEFAULT_CFG, overrides=overrides)
        args.task = self.task
        # if args.imgsz == DEFAULT_CFG.imgsz:
        #     args.imgsz = self.model.args['imgsz']  # use trained imgsz unless custom value is passed
        args.batch = 1  # default to 1 if not modified

        return ExporterSol(overrides=args)(model=self.model)

    def train(self, **kwargs):
        """
        Trains the model on a given dataset.

        Args:
            **kwargs (Any): Any number of arguments representing the training configuration.
        """
        self._check_is_pytorch_model()
        # if self.session:  # Ultralytics HUB session
        #     if any(kwargs):
        #         LOGGER.warning('WARNING ⚠️ using HUB training arguments, ignoring local training arguments.')
        #     kwargs = self.session.train_args
        #     self.session.check_disk_space()
        # check_pip_update_available()
        overrides = self.overrides.copy()
        overrides.update(kwargs)
        if kwargs.get('cfg'):
            LOGGER.info(f"cfg file passed. Overriding default params with {kwargs['cfg']}.")
            overrides = yaml_load(kwargs['cfg'])
        overrides['mode'] = 'train'
        if not overrides.get('data'):
            raise AttributeError("Dataset required but missing, i.e. pass 'data=coco128.yaml'")
        if overrides.get('resume'):
            overrides['resume'] = self.ckpt_path
        self.task = overrides.get('task') or self.task
        self.trainer = TASK_MAP[self.task][1](overrides=overrides, _callbacks=self.callbacks)
        if not overrides.get('resume'):  # manually set model only if not resuming
            weights = overrides['model']
            self.ckpt_path = weights
            model_weights, self.ckpt = attempt_load_one_weight(weights)
            self.trainer.model = self.trainer.get_model(weights=model_weights, cfg=self.model.yaml)
            self.model = self.trainer.model
        self.trainer.hub_session = self.session  # attach optional HUB session
        self.trainer.train()
        # update model and cfg after training
        if RANK in (-1, 0):
            if self.trainer.best.is_file() and os.path.exists(self.trainer.best) and os.path.getsize(self.trainer.best) > 512:
                self.model, _ = attempt_load_one_weight(str(self.trainer.best))
                self.overrides = self.model.args
                self.metrics = getattr(self.trainer.validator, 'metrics', None)  # TODO: no metrics returned by DDP
