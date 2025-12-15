"""
SAHI Detection Aggregator for Validation.

This module provides functionality to aggregate predictions from SAHI (Slicing Aided Hyper Inference)
crops back to full image coordinates for metric computation.

SAHI Workflow:
    1. Large images are split into overlapping crops (+ optional letterboxed full image)
    2. Model inference runs on each crop independently
    3. This aggregator collects predictions from all crops
    4. Predictions are transformed from crop coordinates to original image coordinates
    5. Final NMS is applied to merged predictions
    6. Metrics are computed against ground truth in original image space
"""

import torch
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from ultralytics.utils import LOGGER, ops


# ==================== Utility Functions ====================

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


# ==================== SAHI Crop Aggregator ====================

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
    
    # -------------------- Lifecycle Methods --------------------
        
    def reset(self):
        """Reset aggregator state for a new validation run."""
        self.image_crops = defaultdict(lambda: {
            'predictions': [],        # List of prediction tensors
            'processed_crops': set(), # Set of processed slice indices
            'original_shape': None,   # (h, w) of original image
            'original_img_idx': None, # Index in dataset
        })
        self.expected_crops_per_image = {}
        self._batch_count = 0
        
    def calculate_expected_crops(self, dataset):
        """
        Pre-calculate expected crop count per image from dataset.
        
        This enables tracking when all crops for an image have been processed.
        
        Args:
            dataset: SAHIDataset with slice_indices attribute
        """
        if not hasattr(dataset, 'slice_indices'):
            return
            
        self.expected_crops_per_image.clear()
        for img_idx, _, _ in dataset.slice_indices:
            self.expected_crops_per_image[img_idx] = self.expected_crops_per_image.get(img_idx, 0) + 1
    
    # -------------------- Prediction Collection --------------------
    
    def add_crop_predictions(
        self, 
        batch: Dict[str, Any], 
        preds_before_nms: torch.Tensor,
        preds_after_nms: List[torch.Tensor]
    ) -> bool:
        """
        Add predictions from a batch of crops to the aggregator.
        
        Extracts SAHI metadata from the batch and processes each crop's predictions,
        transforming coordinates from model space to original image space.
        
        Args:
            batch: Batch dict containing:
                - original_img_idx: List of original image indices
                - slice_idx: List of slice indices within each image
                - slice_coords: List of (x1, y1, x2, y2) crop coordinates
                - ori_shape: List of original image shapes (h, w)
                - resized_shape: List of resized crop shapes
                - ratio_pad: List of (ratio, padding) tuples
                - is_full_image: List of booleans indicating full image slices
            preds_before_nms: Raw model predictions [batch, channels, anchors]
            preds_after_nms: Post-NMS predictions (not used for detection)
            
        Returns:
            True if successful, False if metadata missing
        """
        # Extract SAHI metadata from batch
        original_img_idx = batch.get('original_img_idx', [])
        slice_idx = batch.get('slice_idx', [])
        slice_coords = batch.get('slice_coords', [])
        ori_shapes = batch.get('ori_shape', [])
        resized_shapes = batch.get('resized_shape', [])
        ratio_pads = batch.get('ratio_pad', [])
        is_full_image_list = batch.get('is_full_image', [False] * len(original_img_idx))

        if not original_img_idx:
            LOGGER.warning("SAHI metadata missing in batch, skipping aggregation")
            return False

        imgsz = batch['img'].shape[2:]  # Model input size (h, w)

        # Handle tuple output from model
        if isinstance(preds_before_nms, tuple):
            preds_before_nms = preds_before_nms[0] if preds_before_nms else None

        if preds_before_nms is None or not hasattr(preds_before_nms, 'shape'):
            return True  # No predictions, but not an error

        if len(preds_before_nms.shape) != 3:
            LOGGER.debug(f"Unexpected preds_before_nms shape: {preds_before_nms.shape}")
            return False

        # Process each crop in batch
        batch_size = min(len(original_img_idx), preds_before_nms.shape[0])

        for i in range(batch_size):
            self._process_single_crop(
                batch_idx=i,
                img_idx=int(original_img_idx[i]),
                slice_idx=int(slice_idx[i]),
                slice_coord=_to_tuple(slice_coords[i]),
                ori_shape=_to_tuple(ori_shapes[i]),
                resized_shape=_to_tuple(resized_shapes[i]),
                ratio_pad=_fix_ratio_pad(_to_tuple(ratio_pads[i])),
                is_full=bool(is_full_image_list[i]) if i < len(is_full_image_list) else False,
                crop_preds_raw=preds_before_nms[i],
                imgsz=imgsz
            )
                    
        return True

    def _process_single_crop(
        self,
        batch_idx: int,
        img_idx: int,
        slice_idx: int,
        slice_coord: Tuple[int, int, int, int],
        ori_shape: Tuple[int, int],
        resized_shape: Tuple[int, int],
        ratio_pad: Optional[Tuple],
        is_full: bool,
        crop_preds_raw: torch.Tensor,
        imgsz: Tuple[int, int]
    ):
        """
        Process predictions from a single crop.
        
        Transforms box coordinates from model space to original image coordinates.
        
        Coordinate Transformation:
            1. Transpose predictions: [channels, anchors] -> [anchors, channels]
            2. Filter by confidence threshold
            3. Convert xywh to xyxy format
            4. For full image: use scale_boxes with letterbox parameters
            5. For crops: scale to crop size, then add crop offset
            6. Clip to original image bounds
            7. Filter degenerate boxes (zero area after clipping)
            8. Convert back to xywh for storage
        
        Args:
            batch_idx: Index within the current batch
            img_idx: Original image index in dataset
            slice_idx: Slice index within the image
            slice_coord: (x1, y1, x2, y2) crop coordinates in original image
            ori_shape: (h, w) of original image
            resized_shape: (h, w) of crop after resize
            ratio_pad: ((scale_x, scale_y), (pad_w, pad_h)) from letterbox
            is_full: Whether this is the full image slice
            crop_preds_raw: Raw predictions tensor [channels, anchors]
            imgsz: Model input size (h, w)
        """
        img_key = str(img_idx)
        
        # Normalize shapes to tuples of ints
        ori_shape = tuple(int(x) for x in ori_shape)
        resized_shape = tuple(int(x) for x in resized_shape)
        slice_coord = tuple(int(c) for c in slice_coord)
        imgsz = tuple(int(x) for x in imgsz)
        
        # Initialize image data on first crop
        if self.image_crops[img_key]['original_shape'] is None:
            self.image_crops[img_key]['original_shape'] = ori_shape
            self.image_crops[img_key]['original_img_idx'] = img_idx
        
        self.image_crops[img_key]['processed_crops'].add(slice_idx)
        
        # Transpose: [channels, anchors] -> [anchors, channels]
        # Format: [x, y, w, h, class_scores..., optional_mask_coeffs...]
        crop_preds = crop_preds_raw.T
        
        # Get number of classes (sum of all tasks)
        nc_total = sum(self.validator.nc)
        
        # Compute confidence as max class score (YOLOv8 has no objectness)
        class_scores = crop_preds[:, 4:4+nc_total]
        conf = class_scores.max(dim=1)[0]
        
        # Filter by confidence (low threshold - NMS will filter more later)
        conf_threshold = 0.001
        valid_mask = conf > conf_threshold
        crop_preds_filtered = crop_preds[valid_mask]
        
        if len(crop_preds_filtered) == 0:
            return
        
        # Transform boxes from model space (xywh) to original image coordinates (xyxy)
        boxes_xywh = crop_preds_filtered[:, :4].clone()
        boxes_xyxy = ops.xywh2xyxy(boxes_xywh)
        
        if is_full:
            # Full letterboxed image: use scale_boxes to transform from letterbox to original
            ops.scale_boxes(imgsz, boxes_xyxy, ori_shape, ratio_pad=ratio_pad)
        else:
            # Crop: first scale from model space to crop space, then add offset
            ops.scale_boxes(imgsz, boxes_xyxy, resized_shape, ratio_pad=ratio_pad)
            x_min, y_min = slice_coord[0], slice_coord[1]
            boxes_xyxy[:, 0] += x_min
            boxes_xyxy[:, 1] += y_min
            boxes_xyxy[:, 2] += x_min
            boxes_xyxy[:, 3] += y_min
        
        # Clip to original image bounds
        h, w = ori_shape
        boxes_xyxy[:, 0].clamp_(0, w)
        boxes_xyxy[:, 1].clamp_(0, h)
        boxes_xyxy[:, 2].clamp_(0, w)
        boxes_xyxy[:, 3].clamp_(0, h)
        
        # Filter out degenerate boxes (zero or negative area after clipping)
        box_w = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
        box_h = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]
        valid_boxes = (box_w > 1) & (box_h > 1)  # At least 1 pixel
        boxes_xyxy = boxes_xyxy[valid_boxes]
        crop_preds_filtered = crop_preds_filtered[valid_boxes]
        
        if len(crop_preds_filtered) == 0:
            return
        
        # Convert back to xywh for NMS (NMS expects xywh in first 4 columns)
        boxes_xywh_transformed = ops.xyxy2xywh(boxes_xyxy)
        
        # Build output tensor with transformed coordinates
        crop_preds_transformed = crop_preds_filtered.clone()
        crop_preds_transformed[:, :4] = boxes_xywh_transformed
        
        self.image_crops[img_key]['predictions'].append(crop_preds_transformed.detach())
    
    # -------------------- Prediction Retrieval --------------------
    
    def get_aggregated_predictions(self, img_key: str) -> torch.Tensor:
        """
        Get concatenated predictions for a complete image.
        
        Args:
            img_key: Image key (string of image index)
            
        Returns:
            Tensor of shape [N, num_outputs] with xywh boxes in original image coords
        """
        data = self.image_crops[img_key]
        
        if not data['predictions']:
            nc_total = sum(self.validator.nc) if hasattr(self.validator, 'nc') else 80
            num_cols = 4 + nc_total  # xywh + class scores
            return torch.empty((0, num_cols), device=self.validator.device)
        
        return torch.cat(data['predictions'], dim=0)
    
    # -------------------- Completion Tracking --------------------
    
    def is_image_complete(self, img_key: str) -> bool:
        """
        Check if all crops for an image have been processed.
        
        Args:
            img_key: Image key (string of image index)
            
        Returns:
            True if all expected crops have been processed
        """
        img_idx = self.image_crops[img_key].get('original_img_idx')
        if img_idx is None:
            return False
            
        expected = self.expected_crops_per_image.get(img_idx, 0)
        processed = len(self.image_crops[img_key]['processed_crops'])
        
        return expected > 0 and processed >= expected
    
    def get_completed_images(self) -> List[str]:
        """
        Get list of image keys that have all crops processed.
        
        Returns:
            List of image keys ready for final NMS and metrics
        """
        return [img_key for img_key in list(self.image_crops.keys()) if self.is_image_complete(img_key)]
    
    # -------------------- Memory Management --------------------
    
    def cleanup_image(self, img_key: str):
        """
        Clean up stored data for a processed image to free memory.
        
        Should be called after an image has been fully processed and
        metrics have been computed.
        
        Args:
            img_key: Image key to clean up
        """
        if img_key in self.image_crops:
            data = self.image_crops[img_key]
            
            # Clear predictions list
            if 'predictions' in data:
                data['predictions'].clear()
            
            # Remove the image entry
            del self.image_crops[img_key]
    
    def get_memory_stats(self) -> dict:
        """
        Get memory statistics for debugging.
        
        Returns:
            Dict with num_incomplete_images and num_stored_predictions
        """
        num_images = len(self.image_crops)
        num_preds = sum(sum(len(p) for p in data['predictions']) for data in self.image_crops.values())
        
        return {
            'num_incomplete_images': num_images,
            'num_stored_predictions': num_preds,
        }
