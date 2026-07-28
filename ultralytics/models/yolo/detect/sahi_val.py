import torch
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from ultralytics.utils import ops


def _to_tuple(val) -> tuple:
    """
    Recursively convert lists/arrays to tuples for hashability.
    
    Args:
        val: Value to convert (list, tuple, numpy array, or scalar)
        
    Returns:
        Tuple representation of the value
    """
    if isinstance(val, (list, tuple)):
        return tuple(_to_tuple(v) for v in val)
    if hasattr(val, 'tolist'):  # numpy array
        return tuple(_to_tuple(v) for v in val.tolist()) if val.ndim > 0 else val.item()
    return val


def _fix_ratio_pad(ratio_pad: tuple) -> Optional[tuple]:
    """
    Fix ratio_pad format from collate_fn artifacts.
    
    The dataloader's collate_fn can introduce extra nesting in ratio_pad.
    This function extracts the correct format.
    
    Expected format: ((scale_x, scale_y), (pad_w, pad_h))
    Possible artifact: (((scale_x, scale_y), (pad_w, pad_h)), (0, 0))
    
    Args:
        ratio_pad: Potentially malformed ratio_pad tuple
        
    Returns:
        Correctly formatted ratio_pad or None
    """
    if ratio_pad is None:
        return None
    
    # Convert lists to tuples recursively using _to_tuple
    ratio_pad = _to_tuple(ratio_pad)
    
    # Check for extra nesting: (((a, b), (c, d)), ...)
    if (isinstance(ratio_pad, tuple) and 
        len(ratio_pad) >= 1 and 
        isinstance(ratio_pad[0], tuple) and
        len(ratio_pad[0]) == 2 and
        isinstance(ratio_pad[0][0], tuple)):
        return ratio_pad[0]
    
    return ratio_pad


class SAHICropAggregator:
    """
    Aggregates predictions from SAHI crops back to full image coordinates.
    
    This class handles the core SAHI aggregation logic for detection:
    
    1. **Collecting**: Stores predictions from each crop as they are processed
    2. **Transforming**: Converts crop-space coordinates to original image coordinates
    3. **Tracking**: Monitors which crops have been processed for each image
    4. **Memory Management**: Stores predictions on CPU and cleans up after processing
    
    Attributes:
        validator: Reference to DetectionValidator for device and nc info
        image_crops: Dict mapping image keys to aggregated data
        expected_crops_per_image: Dict mapping image index to expected crop count
        
    Example:
        ```python
        aggregator = SAHICropAggregator(validator)
        aggregator.calculate_expected_crops(dataset)
        
        # During validation loop:
        aggregator.add_crop_predictions(batch, raw_preds, nms_preds)
        
        # Process completed images:
        for img_key in aggregator.get_completed_images():
            preds = aggregator.get_aggregated_predictions(img_key)
            # Apply NMS, compute metrics...
            aggregator.cleanup_image(img_key)
        ```
    """

    def __init__(self, validator):
        """
        Initialize SAHI crop aggregator.
        
        Args:
            validator: Reference to DetectionValidator instance
        """
        self.validator = validator
        self._batch_count = 0
        self.reset()

    def reset(self):
        """Reset aggregator state for a new validation run."""
        self.image_crops = defaultdict(lambda: {
            "predictions": [],
            "processed_crops": set(),
            "original_shape": None,
            "original_img_idx": None,
        })
        self.expected_crops_per_image = {}
        self._batch_count = 0

    def calculate_expected_crops(self, dataset):
        """Pre-calculate expected crop count per image from dataset.

        Args:
            dataset: SAHIDataset with slice_indices attribute.
        """
        if not hasattr(dataset, "slice_indices"):
            return
        self.expected_crops_per_image.clear()
        for img_idx, _, _ in dataset.slice_indices:
            self.expected_crops_per_image[img_idx] = self.expected_crops_per_image.get(img_idx, 0) + 1

    def add_crop_predictions(self, batch, raw_preds):
        """Add predictions from a batch of crops.

        Args:
            batch: Batch dict with SAHI metadata:
                original_img_idx, slice_idx, slice_coords, ori_shape,
                resized_shape, ratio_pad, is_full_image, img.
            raw_preds: Raw model predictions [batch, 4+sum(nc), anchors].
        """
        original_img_idx = batch.get("original_img_idx", [])
        slice_idx = batch.get("slice_idx", [])
        slice_coords = batch.get("slice_coords", [])
        ori_shapes = batch.get("ori_shape", [])
        resized_shapes = batch.get("resized_shape", [])
        ratio_pads = batch.get("ratio_pad", [])
        is_full_list = batch.get("is_full_image", [False] * len(original_img_idx))

        if not original_img_idx:
            return

        imgsz = tuple(int(x) for x in batch["img"].shape[2:])

        if isinstance(raw_preds, (list, tuple)):
            raw_preds = raw_preds[0]

        if raw_preds is None or raw_preds.ndim != 3:
            return

        batch_size = min(len(original_img_idx), raw_preds.shape[0])
        for i in range(batch_size):
            self._process_single_crop(
                img_idx=int(original_img_idx[i]),
                slice_idx=int(slice_idx[i]),
                slice_coord=tuple(int(c) for c in _to_tuple(slice_coords[i])),
                ori_shape=tuple(int(x) for x in _to_tuple(ori_shapes[i])),
                resized_shape=tuple(int(x) for x in _to_tuple(resized_shapes[i])),
                ratio_pad=_fix_ratio_pad(_to_tuple(ratio_pads[i])),
                is_full=bool(is_full_list[i]) if i < len(is_full_list) else False,
                crop_preds=raw_preds[i],
                imgsz=imgsz,
            )

    def _process_single_crop(
        self,
        img_idx,
        slice_idx,
        slice_coord,
        ori_shape: tuple,
        resized_shape: tuple,
        ratio_pad: tuple,
        is_full: bool,
        crop_preds: torch.Tensor,
        imgsz: tuple,
        conf: float = 0.001,
    ):
        """Process predictions from a single crop.

        Transforms box coordinates: model space -> crop space -> original image space.

        Args:
            img_idx: Original image index in dataset.
            slice_idx: Slice index within the image.
            slice_coord: (x1, y1, x2, y2) crop coordinates in original image.
            ori_shape: (h, w) of original image.
            resized_shape: (h, w) of crop after resize.
            ratio_pad: ((scale_x, scale_y), (pad_w, pad_h)) from letterbox.
            is_full: Whether this is the full image slice.
            crop_preds: Raw predictions [4+sum(nc), anchors] (xywh + sigmoid scores).
            imgsz: Model input size (h, w).
        """
        img_key = str(img_idx)

        # Initialize image data on first crop
        data = self.image_crops[img_key]
        if data["original_shape"] is None:
            data["original_shape"] = ori_shape
            data["original_img_idx"] = img_idx
        data["processed_crops"].add(slice_idx)

        # Transpose: [channels, anchors] -> [anchors, channels]
        preds = crop_preds.T  # [N, 4 + sum(nc)]
        nc_total = sum(self.validator.nc)

        # Filter by max class confidence (low threshold, final NMS filters more)
        preds = preds[preds[:, 4:4 + nc_total].max(dim=1)[0] > conf]
        if len(preds) == 0:
            return

        # Transform boxes: model space xywh -> original image xyxy
        boxes_xyxy = ops.xywh2xyxy(preds[:, :4].clone())

        if is_full:
            ops.scale_boxes(imgsz, boxes_xyxy, ori_shape, ratio_pad=ratio_pad)
        else:
            ops.scale_boxes(imgsz, boxes_xyxy, resized_shape, ratio_pad=ratio_pad)
            x_off, y_off = slice_coord[0], slice_coord[1]
            boxes_xyxy[:, 0] += x_off
            boxes_xyxy[:, 1] += y_off
            boxes_xyxy[:, 2] += x_off
            boxes_xyxy[:, 3] += y_off

        # Clip to image bounds
        h, w = ori_shape
        boxes_xyxy[:, 0].clamp_(0, w)
        boxes_xyxy[:, 1].clamp_(0, h)
        boxes_xyxy[:, 2].clamp_(0, w)
        boxes_xyxy[:, 3].clamp_(0, h)

        # Filter degenerate boxes
        valid = ((boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) > 1) & ((boxes_xyxy[:, 3] - boxes_xyxy[:, 1]) > 1)
        preds = preds[valid]
        boxes_xyxy = boxes_xyxy[valid]
        if len(preds) == 0:
            return

        # Store with transformed xywh coordinates (NMS expects xywh input)
        result = preds.clone()
        result[:, :4] = ops.xyxy2xywh(boxes_xyxy)
        data["predictions"].append(result.detach().cpu())

    def get_aggregated_predictions(self, img_key) -> torch.Tensor:
        """Get concatenated predictions for a complete image.

        Returns:
            Tensor [N, 4+sum(nc)] with xywh boxes in original image coords,
            or empty tensor if no predictions.
        """
        data = self.image_crops[img_key]
        if not data["predictions"]:
            nc_total = sum(self.validator.nc)
            return torch.empty((0, 4 + nc_total), device=self.validator.device)
        # Move back to validator device for NMS
        return torch.cat(data["predictions"], dim=0).to(self.validator.device)

    def is_image_complete(self, img_key: int):
        """Check if all crops for an image have been processed."""
        img_idx = self.image_crops[img_key].get("original_img_idx")
        if img_idx is None:
            return False
        expected = self.expected_crops_per_image.get(img_idx, 0)
        processed = len(self.image_crops[img_key]["processed_crops"])
        return expected > 0 and processed >= expected

    def get_completed_images(self) -> list[int]:
        """Get list of image keys that have all crops processed."""
        return [k for k in list(self.image_crops.keys()) if self.is_image_complete(k)]

    def cleanup_image(self, img_key: int) -> None:
        """Clean up stored data for a processed image."""
        if img_key in self.image_crops:
            self.image_crops[img_key]["predictions"].clear()
            del self.image_crops[img_key]
