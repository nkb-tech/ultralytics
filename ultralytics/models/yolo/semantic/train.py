# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Trainer for YOLO semantic segmentation. Backport from upstream v8.4.104 to fork 8.3.6."""

from copy import copy
from pathlib import Path

import torch

from ultralytics.data.utils import add_polygon_background
from ultralytics.models import yolo
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import SemanticSegmentationModel
from ultralytics.utils import DEFAULT_CFG, RANK


def _scalar(v):
    """Fork wraps nc/names as a 1-element list for multihead."""
    return v[0] if isinstance(v, (list, tuple)) and len(v) == 1 else v


class SemanticSegmentationTrainer(DetectionTrainer):
    """Trainer for YOLO semantic segmentation models.

    Examples:
        >>> args = dict(model="yolo26n-sem.pt", data="cityscapes8.yaml", epochs=3)
        >>> SemanticSegmentationTrainer(overrides=args).train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict | None = None, _callbacks=None):
        """Initialize SemanticSegmentationTrainer."""
        if overrides is None:
            overrides = {}
        overrides["task"] = "semantic"
        super().__init__(cfg, overrides, _callbacks)

    def get_dataset(self):
        """Parse dataset YAML, add background metadata for polygon labels."""
        paths = super().get_dataset()
        self.data = add_polygon_background(self.data)
        return paths

    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        """Return a SemanticSegmentationModel with optional pretrained weights."""
        model = SemanticSegmentationModel(cfg, nc=_scalar(self.data["nc"]), ch=3, verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def set_model_attributes(self):
        """Attach nc, names and hyperparameters to the model."""
        self.model.nc = _scalar(self.data["nc"])
        self.model.names = _scalar(self.data["names"])
        self.model.args = self.args
        cw = self.data.get("class_weights")
        dg = self.data.get("dice_gain")
        fg = self.data.get("focal_gamma")
        if cw is not None or dg is not None or fg is not None:
            from ultralytics.utils import LOGGER

            if cw is not None:
                self.model.class_weights = torch.tensor(cw, dtype=torch.float32)
                LOGGER.info(f"Semantic pixel class weights: {cw}")
            if dg is not None:
                self.model.dice_gain = float(dg)
                LOGGER.info(f"Semantic dice_gain: {dg}")
            if fg is not None:
                self.model.focal_gamma = float(fg)
                LOGGER.info(f"Semantic focal_gamma: {fg}")

    def get_validator(self):
        """Return a SemanticSegmentationValidator and register loss names."""
        self.loss_names = "ce_loss", "dice_loss", "aux_loss"  # must match SemanticSegmentationLoss
        return yolo.semantic.SemanticSegmentationValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def preprocess_batch(self, batch):
        """Move images and masks to device. Skip parent multi_scale — it would desync the mask."""
        batch["img"] = batch["img"].to(self.device, non_blocking=True).float() / 255
        if "semantic_mask" in batch:
            batch["semantic_mask"] = batch["semantic_mask"].to(self.device, non_blocking=True).long()
        data_name = Path(str(getattr(self.args, "data", ""))).stem.lower()
        # Freeze BN stats on Cityscapes only; analog OSD collapses if frozen.
        if str(self.args.model).endswith(".pt") and data_name in {"cityscapes", "cityscapes8"}:
            for m in self.model.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.momentum = 0.0
        batch.setdefault("cls", torch.zeros(batch["img"].shape[0], 1, device=self.device))  # dummy batch size
        return batch

    def plot_training_samples(self, batch, ni):
        pass

    def plot_training_labels(self):
        pass

    def plot_metrics(self):
        pass

    def validate(self):
        """Drop criterion created under inference_mode so train backward still works."""
        metrics, fitness = super().validate()
        self.model.criterion = None
        return metrics, fitness
