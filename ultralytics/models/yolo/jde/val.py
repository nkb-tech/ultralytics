# Ultralytics YOLO 🚀, AGPL-3.0 license

from pathlib import Path, PosixPath

import numpy as np
import torch

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import ops
from ultralytics.utils.metrics import DetMetrics, box_iou, ReIDMetrics
from ultralytics.utils.torch_utils import smart_inference_mode


class JDEValidator(DetectionValidator):
    """
    A class extending the DetectionValidator class for validation based on a joint detection and embedding model.

    Example:
        ```python
        from ultralytics.models.yolo.jde import JDEValidator

        args = dict(model="yolov8n-jde.pt", data="coco8-seg.yaml")
        validator = JDEValidator(args=args)
        validator()
        ```
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initialize SegmentationValidator and set task to 'segment', metrics to SegmentMetrics."""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.plot_masks = None
        self.process = None
        self.args.task = "jde"
        self.metrics = DetMetrics(save_dir=self.save_dir, on_plot=self.on_plot)
        self.reid_metrics = ReIDMetrics()