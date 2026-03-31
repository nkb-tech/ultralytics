# Ultralytics YOLO 🚀, AGPL-3.0 license

from copy import copy
from pathlib import Path
from typing import Any

from ultralytics.models import yolo
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import JDEModel, yaml_model_load
from ultralytics.utils import DEFAULT_CFG, RANK


class JDETrainer(DetectionTrainer):
    """A trainer class for joint detection and embedding (JDE) models."""

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks=None):
        """Initialize JDETrainer and include tags in dynamic compile tensors."""
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)
        if "tags" not in self.dynamic_tensors:
            self.dynamic_tensors.append("tags")

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a JDE model and optionally load weights."""
        if isinstance(cfg, (str, Path)):
            cfg = yaml_model_load(cfg)
        model = JDEModel(
            cfg,
            nc=[1] if self.args.single_cls else self.data["nc"],
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Return a JDE validator instance for validation during training."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss", "reid_loss"
        return yolo.jde.JDEValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )
