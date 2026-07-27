# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Semantic segmentation validator, backported from upstream 8.4.104.

Fork adaptations (see docs/backport/api-drift.md):
  - BaseValidator.__init__ takes `pbar` as the third positional arg.
  - DetectionValidator.preprocess() indexes batch["cls"]/["bboxes"], absent here -> own impl.
  - DetectionValidator.print_results() is multihead (self.metrics[i]) -> own impl.
  - BaseValidator has no get_dataset(); add_polygon_background is applied in build_dataset().
  - ConfusionMatrix has an incompatible signature in the fork -> disabled for semantic.
  - plot_images() uses the old positional signature -> plotting disabled for now.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ultralytics.data.build import build_yolo_dataset
from ultralytics.data.dataset import SemanticDataset
from ultralytics.data.utils import add_polygon_background
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER
from ultralytics.utils.metrics import SemanticMetrics


class SemanticSegmentationValidator(DetectionValidator):
    """Validator for semantic segmentation models (mIoU, pixel accuracy)."""

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initialize SemanticSegmentationValidator."""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.args.task = "semantic"
        self.dataset = None
        self.results_dir = None
        self.metrics = SemanticMetrics()
        self.image_shapes = {}
        self._semantic_target_shape = None

    def build_dataset(self, img_path, mode="val", batch=None):
        """Build a semantic dataset, adding polygon background metadata when needed."""
        self.data = add_polygon_background(self.data)
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, stride=self.stride)

    def init_metrics(self, model):
        """Initialize metrics with model class names."""
        names = getattr(model, "names", None) or self.data.get("names", {})
        if isinstance(names, (list, tuple)):  # fork stores multihead names as a list of dicts
            names = names[0]
        self.names = names
        self.nc = len(self.names)
        self.metrics = SemanticMetrics(names=self.names)
        self.seen = 0
        self.dataset = getattr(self.dataloader, "dataset", None)
        labels = getattr(self.dataset, "labels", []) if self.dataset is not None else []
        self.image_shapes = {lb["im_file"]: tuple(lb["shape"]) for lb in labels if "im_file" in lb and "shape" in lb}
        self.results_dir = None
        if self.args.save_json:
            self.results_dir = self.save_dir / "results"
            self.results_dir.mkdir(parents=True, exist_ok=True)

    def preprocess(self, batch):
        """Move images and masks to device (no bbox/cls keys in semantic batches)."""
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = (batch["img"].half() if self.args.half else batch["img"].float()) / 255
        batch["semantic_mask"] = batch["semantic_mask"].to(self.device, dtype=torch.int32)
        self._semantic_target_shape = tuple(batch["semantic_mask"].shape[-2:])
        return batch

    def postprocess(self, preds):
        """Convert logits or baked class maps to per-pixel class predictions."""
        if isinstance(preds, (tuple, list)):
            preds = preds[0]
        if preds.ndim == 3:  # [B, H, W] class map, argmax already baked in
            if tuple(preds.shape[-2:]) != self._semantic_target_shape:
                preds = F.interpolate(preds[:, None].float(), size=self._semantic_target_shape, mode="nearest")[:, 0]
            return preds.to(torch.int32)
        pred_hw = preds.shape[2:]
        if pred_hw[0] != self._semantic_target_shape[0] or pred_hw[1] != self._semantic_target_shape[1]:
            preds = F.interpolate(preds, size=self._semantic_target_shape, mode="bilinear", align_corners=False)
        return preds.argmax(dim=1).to(torch.int32) if self.nc > 1 else preds.gt(0).squeeze(1).to(torch.int32)

    def update_metrics(self, preds, batch):
        """Accumulate the confusion matrix over a batch."""
        if self.args.save_json:
            self.save_pred_masks(preds, batch)
        self.metrics.update_stats(preds, batch["semantic_mask"])
        self.seen += preds.shape[0]

    def save_pred_masks(self, preds, batch):
        """Save semantic predictions as single-channel PNG masks."""
        if self.results_dir is None:
            return
        im_files = batch.get("im_file", [])
        if not im_files:
            return
        preds = preds.cpu().numpy()
        if isinstance(self.dataset, SemanticDataset) and self.dataset.label_mapping:
            preds = self.dataset.convert_label(preds, inverse=True)
        preds = preds.astype(np.uint8, copy=False)
        for pred, im_file in zip(preds, im_files):
            orig_shape = self.image_shapes.get(im_file)
            if orig_shape and pred.shape != orig_shape:
                pred = cv2.resize(pred, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_NEAREST)
            Image.fromarray(pred).save(self.results_dir / Path(im_file).with_suffix(".png").name)

    def get_stats(self):
        """Compute mIoU / pixel accuracy from the accumulated matrix."""
        self.metrics.process(save_dir=self.save_dir, plot=self.args.plots, on_plot=self.on_plot)
        return self.metrics.results_dict

    def finalize_metrics(self, *args, **kwargs):
        """Attach timing info to the metrics object."""
        self.metrics.speed = self.speed

    def check_stats(self, stats):
        """No-op: semantic stats need no validation."""
        return stats

    def get_desc(self):
        """Header for the per-class results table."""
        return ("%22s" + "%11s" * 4) % ("Class", "Images", "Pixels", "mIoU", "PixAcc")

    def print_results(self):
        """Print overall and per-class metrics."""
        pf = "%22s" + "%11i" * 2 + "%11.3g" * 2
        nt = self.metrics.nt_per_class
        LOGGER.info(pf % ("all", self.seen, int(nt.sum()), self.metrics.miou, self.metrics.pixel_accuracy))
        if self.args.verbose and not self.training and self.nc > 1:
            iou = self.metrics.per_class_iou
            acc = self.metrics.per_class_pixel_accuracy
            for i, name in self.names.items():
                if i < len(iou):
                    LOGGER.info(pf % (name, int(self.metrics.nt_per_image[i]), int(nt[i]), iou[i], acc[i]))
        if self.args.save_json and self.results_dir is not None:
            LOGGER.info(f"Semantic prediction masks saved to {self.results_dir}")

    def plot_val_samples(self, batch, ni):
        """Disabled: fork's plot_images() has an incompatible signature for semantic masks."""

    def plot_predictions(self, batch, preds, ni):
        """Disabled: fork's plot_images() has an incompatible signature for semantic masks."""
