import numpy as np
import torch
from typing import List, Tuple
from collections import defaultdict


def slice_image(
    im: np.ndarray, crop_size: int, overlap_ratio: float = 0.2
) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """Slice an image into overlapping crops.

    Args:
        im: Input image (H, W, C).
        crop_size: Side length of each square crop.
        overlap_ratio: Fractional overlap between adjacent crops.

    Returns:
        List of (crop_array, (x1, y1, x2, y2)) tuples in original-image coordinates.
    """
    h, w = im.shape[:2]
    coords = _slice_coordinates(h, w, crop_size, overlap_ratio)
    return [(im[y1:y2, x1:x2], (x1, y1, x2, y2)) for x1, y1, x2, y2 in coords]


def _slice_coordinates(
    img_h: int, img_w: int, crop_size: int, overlap_ratio: float
) -> List[Tuple[int, int, int, int]]:
    """Calculate grid of (x1, y1, x2, y2) crop coordinates."""
    if img_h <= crop_size and img_w <= crop_size:
        return [(0, 0, img_w, img_h)]

    step = crop_size - int(overlap_ratio * crop_size)

    def _starts(length):
        n = max(1, (length - crop_size + step - 1) // step + 1) if length > crop_size else 1
        s = np.arange(n) * step
        if length > crop_size and n > 1:
            s[-1] = length - crop_size
        return s

    return [
        (int(x), int(y), min(int(x) + crop_size, img_w), min(int(y) + crop_size, img_h))
        for y in _starts(img_h)
        for x in _starts(img_w)
    ]


class SAHIPredictAggregator:
    """Collects per-crop predictions and maps them to full-image coordinates."""

    def __init__(self, crop_size: int, overlap_ratio: float = 0.2):
        self.crop_size = crop_size
        self.overlap_ratio = overlap_ratio
        self._preds = defaultdict(list)  # img_key → [tensor, ...]
        self._device = None

    def reset(self):
        self._preds.clear()
        self._device = None

    def add_crop_predictions(
        self, img_key: str, preds: torch.Tensor, crop_coords: Tuple[int, int, int, int]
    ):
        """Shift xyxy boxes from crop-local to full-image coordinates and store."""
        x_off, y_off = crop_coords[0], crop_coords[1]
        shifted = preds.clone()
        shifted[:, [0, 2]] += x_off
        shifted[:, [1, 3]] += y_off
        if self._device is None:
            self._device = shifted.device
        self._preds[img_key].append(shifted)

    def aggregate_predictions(
        self, img_key: str, orig_shape: Tuple[int, int]
    ) -> torch.Tensor:
        """Concatenate all crops for *img_key*, clamp to image bounds, and return (N, C) tensor."""
        entries = self._preds.pop(img_key, [])
        if not entries:
            return torch.empty((0, 6), dtype=torch.float32, device=self._device or "cpu")

        out = torch.cat(entries, dim=0)
        h, w = orig_shape
        out[:, [0, 2]] = out[:, [0, 2]].clamp(0, w)
        out[:, [1, 3]] = out[:, [1, 3]].clamp(0, h)
        return out
