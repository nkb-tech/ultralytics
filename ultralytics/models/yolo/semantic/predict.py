# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import torch
import torch.nn.functional as F

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import DEFAULT_CFG, ops


class SemanticSegmentationPredictor(BasePredictor):
    """Predictor for semantic segmentation models.

    Examples:
        >>> from ultralytics.models.yolo.semantic import SemanticSegmentationPredictor
        >>> args = dict(model="yolo26n-sem.pt", source="path/to/image.jpg")
        >>> predictor = SemanticSegmentationPredictor(overrides=args)
        >>> predictor.predict_cli()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize SemanticSegmentationPredictor."""
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "semantic"

    @staticmethod
    def _class_map_dtype(num_classes: int) -> torch.dtype:
        return torch.uint8 if num_classes <= 256 else torch.int16 if num_classes <= 32768 else torch.int32

    def postprocess(self, preds, img, orig_imgs):
        """Logits [B,nc,H,W] or class map [B,H,W] → Results with semantic_mask."""
        if isinstance(preds, (tuple, list)):
            preds = preds[0]

        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]

        classes = (
            torch.as_tensor(self.args.classes, device=preds.device).flatten()
            if self.args.classes is not None and len(self.model.names) > 1
            else None
        )

        results = []
        for i, (pred, orig_img) in enumerate(zip(preds, orig_imgs)):
            img_path = self.batch[0][i] if isinstance(self.batch[0], list) else self.batch[0]
            class_map_input = pred.ndim == 2  # exported ArgMax: [H, W]
            pred = (pred[None, None] if class_map_input else pred[None]).float()
            if class_map_input:
                if pred.shape[2:] != img.shape[2:]:
                    pred = F.interpolate(pred, img.shape[2:], mode="nearest")
                class_map = ops.scale_masks(pred, orig_img.shape[:2], mode="nearest")[0, 0]
                class_map = class_map.to(self._class_map_dtype(int(class_map.max().item()) + 1))
            else:
                if pred.shape[2:] != img.shape[2:]:
                    pred = F.interpolate(pred, img.shape[2:], mode="bilinear")
                pred = ops.scale_masks(pred, orig_img.shape[:2])[0]
                dtype = self._class_map_dtype(max(pred.shape[0], 2))
                class_map = pred.argmax(0).to(dtype) if pred.shape[0] > 1 else pred.gt(0).squeeze(0).to(dtype)
            if classes is not None:
                class_map[~(class_map.unsqueeze(-1) == classes).any(-1)] = 255
            results.append(Results(orig_img, path=img_path, names=self.model.names, semantic_mask=class_map))
        return results
