"""
SAHI Mask Aggregator for Segmentation Validation.

This module extends the detection aggregator with mask support for instance segmentation.

Key Design Principle:
    Store BOTH model-space boxes (for mask generation) AND original-image-space boxes
    (for NMS and metrics). This is necessary because:
    
    - process_mask() expects boxes in MODEL space (with letterbox padding)
    - NMS and metrics expect boxes in ORIGINAL image space
    
SAHI Segmentation Workflow:
    1. Collect predictions + mask coefficients from each crop
    2. Store proto features for each crop (needed for mask generation)
    3. Store model-space boxes alongside transformed original-space boxes
    4. After NMS, generate masks using stored protos and model-space boxes
    5. Transform masks from crop space to original image coordinates
"""

import torch
import torch.nn.functional as F
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from ultralytics.utils import LOGGER, ops
from ultralytics.models.yolo.detect.sahi_val import _to_tuple, _fix_ratio_pad


class SAHISegmentAggregator:
    """
    Aggregates predictions from SAHI crops with mask support for instance segmentation.
    
    Extends detection aggregation with:
    - Mask coefficient storage
    - Proto feature storage (per-crop)
    - Model-space box preservation for mask generation
    - Mask transformation to original image coordinates
    
    Attributes:
        validator: Reference to SegmentationValidator
        image_crops: Dict mapping image keys to their aggregated data
        expected_crops_per_image: Dict mapping image index to expected crop count
        
    Data Structure (per image):
        ```
        image_crops[img_key] = {
            'predictions': [...],      # Predictions in ORIGINAL image coords
            'crop_metadata': [...],    # Proto, model-space boxes, crop info
            'processed_crops': set(),  # Processed slice indices
            'original_shape': (h, w),  # Original image dimensions
            'original_img_idx': int,   # Dataset index
        }
        ```
        
    Example:
        ```python
        aggregator = SAHISegmentAggregator(validator)
        aggregator.calculate_expected_crops(dataset)
        
        # During validation:
        aggregator.add_crop_predictions(batch, raw_preds, nms_preds, proto)
        
        # Process completed images:
        for img_key in aggregator.get_completed_images():
            preds = aggregator.get_aggregated_predictions(img_key)
            # Apply NMS, generate masks, compute metrics...
            aggregator.cleanup_image(img_key)
        ```
    """
    
    def __init__(self, validator):
        """
        Initialize SAHI segment aggregator.
        
        Args:
            validator: Reference to SegmentationValidator instance
        """
        self.validator = validator
        self._batch_count = 0
        self.reset()
    
    def reset(self):
        """Reset aggregator state for a new validation run."""
        self.image_crops = defaultdict(lambda: {
            'predictions': [],        # Predictions in ORIGINAL image coords
            'crop_metadata': [],      # List of crop metadata dicts
            'processed_crops': set(), # Set of processed slice indices
            'original_shape': None,   # (h, w) of original image
            'original_img_idx': None, # Index in dataset
        })
        self.expected_crops_per_image = {}
        self._batch_count = 0
    
    def calculate_expected_crops(self, dataset):
        """
        Pre-calculate expected crop count per image from dataset.
        
        Args:
            dataset: SAHIDataset with slice_indices attribute
        """
        if not hasattr(dataset, 'slice_indices'):
            return
            
        self.expected_crops_per_image.clear()
        for img_idx, _, _ in dataset.slice_indices:
            self.expected_crops_per_image[img_idx] = self.expected_crops_per_image.get(img_idx, 0) + 1
    
    def add_crop_predictions(
        self, 
        batch: Dict[str, Any], 
        preds_before_nms: torch.Tensor,
        preds_after_nms: List[torch.Tensor],
        proto: Optional[torch.Tensor] = None
    ) -> bool:
        """
        Add predictions from a batch of crops including proto features.
        
        Args:
            batch: Batch dict with SAHI metadata
            preds_before_nms: Raw model predictions [batch, channels, anchors]
            preds_after_nms: Post-NMS predictions (not used)
            proto: Prototype features [batch, 32, h, w]
            
        Returns:
            True if successful, False if metadata missing
        """
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

        imgsz = batch['img'].shape[2:]

        if isinstance(preds_before_nms, tuple):
            preds_before_nms = preds_before_nms[0] if preds_before_nms else None

        if preds_before_nms is None or not hasattr(preds_before_nms, 'shape'):
            return True

        if len(preds_before_nms.shape) != 3:
            LOGGER.debug(f"Unexpected preds_before_nms shape: {preds_before_nms.shape}")
            return False
        
        batch_size = min(preds_before_nms.shape[0], len(original_img_idx))
        
        for i in range(batch_size):
            crop_proto = proto[i] if proto is not None and i < len(proto) else None
            
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
                imgsz=imgsz,
                proto=crop_proto
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
        imgsz: Tuple[int, int],
        proto: Optional[torch.Tensor] = None
    ):
        """
        Process predictions from a single crop including mask coefficients.
        
        Stores both:
        - Model-space boxes (for mask generation via process_mask)
        - Original-image-space boxes (for NMS and metrics)
        
        Args:
            batch_idx: Index within current batch
            img_idx: Original image index in dataset
            slice_idx: Slice index within the image
            slice_coord: (x1, y1, x2, y2) crop coordinates
            ori_shape: (h, w) of original image
            resized_shape: (h, w) of crop after resize
            ratio_pad: Letterbox parameters
            is_full: Whether this is the full image slice
            crop_preds_raw: Raw predictions [channels, anchors]
            imgsz: Model input size (h, w)
            proto: Prototype features for this crop [32, h, w]
        """
        img_key = str(img_idx)
        
        # Normalize shapes
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
        crop_preds = crop_preds_raw.T
        
        # Get number of classes
        nc_total = sum(self.validator.nc)
        
        # Compute confidence
        class_scores = crop_preds[:, 4:4+nc_total]
        conf = class_scores.max(dim=1)[0]
        
        # Filter by confidence
        conf_threshold = 0.001
        valid_mask = conf > conf_threshold
        crop_preds_filtered = crop_preds[valid_mask]
        
        if len(crop_preds_filtered) == 0:
            return
        
        # Store ORIGINAL model-space boxes BEFORE transformation
        # These are needed for process_mask() later
        boxes_xywh_model = crop_preds_filtered[:, :4].clone()
        boxes_xyxy_model = ops.xywh2xyxy(boxes_xywh_model)
        
        # Transform boxes to original image coordinates
        boxes_xyxy_orig = boxes_xyxy_model.clone()
        
        if is_full:
            ops.scale_boxes(imgsz, boxes_xyxy_orig, ori_shape, ratio_pad=ratio_pad)
        else:
            ops.scale_boxes(imgsz, boxes_xyxy_orig, resized_shape, ratio_pad=ratio_pad)
            x_min, y_min = slice_coord[0], slice_coord[1]
            boxes_xyxy_orig[:, 0] += x_min
            boxes_xyxy_orig[:, 1] += y_min
            boxes_xyxy_orig[:, 2] += x_min
            boxes_xyxy_orig[:, 3] += y_min
        
        # Clip to original image bounds
        h, w = ori_shape
        boxes_xyxy_orig[:, 0].clamp_(0, w)
        boxes_xyxy_orig[:, 1].clamp_(0, h)
        boxes_xyxy_orig[:, 2].clamp_(0, w)
        boxes_xyxy_orig[:, 3].clamp_(0, h)
        
        # Filter degenerate boxes
        box_w = boxes_xyxy_orig[:, 2] - boxes_xyxy_orig[:, 0]
        box_h = boxes_xyxy_orig[:, 3] - boxes_xyxy_orig[:, 1]
        valid_boxes = (box_w > 1) & (box_h > 1)
        boxes_xyxy_orig = boxes_xyxy_orig[valid_boxes]
        boxes_xyxy_model = boxes_xyxy_model[valid_boxes]
        crop_preds_filtered = crop_preds_filtered[valid_boxes]
        
        if len(crop_preds_filtered) == 0:
            return
        
        # Convert to xywh for storage
        boxes_xywh_orig = ops.xyxy2xywh(boxes_xyxy_orig)
        
        # Build output tensor with transformed coordinates
        crop_preds_transformed = crop_preds_filtered.clone()
        crop_preds_transformed[:, :4] = boxes_xywh_orig
        
        # Store crop metadata for mask generation
        self.image_crops[img_key]['crop_metadata'].append({
            'proto': proto.detach() if proto is not None else None,
            'boxes_model': boxes_xyxy_model.detach(),
            'imgsz': imgsz,
            'slice_coord': slice_coord,
            'ratio_pad': ratio_pad,
            'is_full': is_full,
            'pred_start_idx': sum(len(p) for p in self.image_crops[img_key]['predictions']),
            'pred_count': len(crop_preds_transformed),
        })
        
        self.image_crops[img_key]['predictions'].append(crop_preds_transformed.detach())

    def get_aggregated_predictions(self, img_key: str) -> torch.Tensor:
        """
        Get concatenated predictions for a complete image.
        
        Returns:
            Tensor [N, num_outputs] with xywh boxes in original image coords
        """
        data = self.image_crops[img_key]
        
        if not data['predictions']:
            nc_total = sum(self.validator.nc) if hasattr(self.validator, 'nc') else 80
            num_cols = 4 + nc_total + 32  # xywh + class scores + mask coeffs
            return torch.empty((0, num_cols), device=self.validator.device)
        
        return torch.cat(data['predictions'], dim=0)
    
    def get_crop_metadata_for_index(self, img_key: str, pred_idx: int) -> Tuple[Optional[dict], int]:
        """
        Get crop metadata and local index for a given prediction index.
        
        Used during mask generation to find which crop a prediction came from.
        
        Args:
            img_key: Image key
            pred_idx: Index in the concatenated predictions tensor
            
        Returns:
            (metadata_dict, local_idx) where local_idx is within that crop
        """
        data = self.image_crops[img_key]
        for meta in data['crop_metadata']:
            start = meta['pred_start_idx']
            end = start + meta['pred_count']
            if start <= pred_idx < end:
                return meta, pred_idx - start
        return None, -1
    
    def is_image_complete(self, img_key: str) -> bool:
        """Check if all crops for an image have been processed."""
        img_idx = self.image_crops[img_key].get('original_img_idx')
        if img_idx is None:
            return False
            
        expected = self.expected_crops_per_image.get(img_idx, 0)
        processed = len(self.image_crops[img_key]['processed_crops'])
        
        return expected > 0 and processed >= expected
    
    def get_completed_images(self) -> List[str]:
        """Get list of image keys that have all crops processed."""
        return [img_key for img_key in list(self.image_crops.keys()) if self.is_image_complete(img_key)]

    
    def cleanup_image(self, img_key: str):
        """
        Clean up stored data for a processed image.
        
        Args:
            img_key: Image key to clean up
        """
        if img_key in self.image_crops:
            data = self.image_crops[img_key]
            
            if 'predictions' in data:
                data['predictions'].clear()
            
            if 'crop_metadata' in data:
                data['crop_metadata'].clear()
            
            del self.image_crops[img_key]
    
    def get_memory_stats(self) -> dict:
        """Get memory statistics for debugging."""
        num_images = len(self.image_crops)
        num_crops = sum(len(data['crop_metadata']) for data in self.image_crops.values())
        num_preds = sum(sum(len(p) for p in data['predictions']) for data in self.image_crops.values())
        
        return {
            'num_incomplete_images': num_images,
            'num_stored_crops': num_crops,
            'num_stored_predictions': num_preds,
        }
