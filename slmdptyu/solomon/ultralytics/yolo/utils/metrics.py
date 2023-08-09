# modify from ultralytics/yolo/utils/metrics.py
"""
Model validation metrics
"""
# import math
# import warnings
from pathlib import Path

# import matplotlib.pyplot as plt
# import numpy as np
# import torch
# import torch.nn as nn

# from ultralytics.yolo.utils import LOGGER, SimpleClass, TryExcept


class SolMetrics:

    def __init__(self, save_dir=Path('.'), plot=False, names=()) -> None:
        self.save_dir = save_dir
        self.plot = plot
        self.names = names

    def process(self, tp_m, tp_b, conf, pred_cls, target_cls):
        pass

    @property
    def keys(self):
        return [
            'metrics/AP', 
            'metrics/f-score', 'metrics/accuracy', 'metrics/precision', 'metrics/recall', 'metrics/score',
            'TP', 'FP', 'FN'
        ]