import torch
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from ultralytics.utils import LOGGER, ops


def _to_tuple(val) -> tuple:
    """Recursively convert lists/arrays to tuples."""
    if isinstance(val, (list, tuple)):
        return tuple(_to_tuple(v) for v in val)
    if hasattr(val, 'tolist'):  # numpy array
        return tuple(_to_tuple(v) for v in val.tolist()) if val.ndim > 0 else val.item()
    return val


def _fix_ratio_pad(ratio_pad: tuple) -> Optional[tuple]:
    """
    Fix ratio_pad format from collate_fn artifacts.
    
    Sometimes ratio_pad gets extra nesting: (((scale, scale), (pad_w, pad_h)), (0, 0))
    This function extracts the correct format: ((scale, scale), (pad_w, pad_h))
    """
    if ratio_pad is None:
        return None
    
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
    
    For each image:
    1. Collects predictions from all crops (grid + full image)
    2. Transforms crop coordinates to original image coordinates
    3. Concatenates all predictions for final NMS
    
    Args:
        validator: Reference to DetectionValidator for device and nc info
    """
    
    def __init__(self, validator):
        self.validator = validator
        self.reset()
        
    def reset(self):
        """Reset aggregator for new validation run."""
        self.image_crops = defaultdict(lambda: {
            'predictions': [],       # List of prediction tensors
            'crop_coords': [],       # Corresponding crop coordinates
            'processed_crops': set(), # Set of processed slice indices
            'original_shape': None,   # (h, w) of original image
            'original_img_idx': None, # Index in dataset
        })
        self.expected_crops_per_image = {}
        
    def calculate_expected_crops(self, dataset):
        """Pre-calculate expected crop count per image from dataset."""
        if not hasattr(dataset, 'slice_indices'):
            return
            
        self.expected_crops_per_image.clear()
        for img_idx, _, _ in dataset.slice_indices:
            self.expected_crops_per_image[img_idx] = self.expected_crops_per_image.get(img_idx, 0) + 1
        
    def add_crop_predictions(self, batch: Dict[str, Any], preds_before_nms, preds_after_nms) -> bool:
        """
        Add predictions from a batch of crops.
        
        Args:
            batch: Batch dict with SAHI metadata (original_img_idx, slice_coords, etc.)
            preds_before_nms: Raw model predictions [batch, channels, anchors]
            preds_after_nms: Not used (NMS done after aggregation)
            
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
            LOGGER.error(f"Unexpected preds_before_nms shape: {preds_before_nms.shape}")
            return False

        # Process each crop in batch
        for i in range(len(original_img_idx)):
            self._process_single_crop(
                i, original_img_idx[i], slice_idx[i],
                _to_tuple(slice_coords[i]),
                _to_tuple(ori_shapes[i]),
                _to_tuple(resized_shapes[i]),
                _fix_ratio_pad(_to_tuple(ratio_pads[i])),
                bool(is_full_image_list[i]) if i < len(is_full_image_list) else False,
                preds_before_nms[i],
                imgsz
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
        """Process predictions from a single crop."""
        img_key = str(img_idx)
        
        # Initialize image data
        if self.image_crops[img_key]['original_shape'] is None:
            self.image_crops[img_key]['original_shape'] = ori_shape
            self.image_crops[img_key]['original_img_idx'] = img_idx
        
        self.image_crops[img_key]['processed_crops'].add(slice_idx)
        
        # Transpose: [channels, anchors] -> [anchors, channels]
        crop_preds = crop_preds_raw.T
        
        # Filter by confidence
        conf_threshold = 0.001
        valid_mask = crop_preds[:, 4] > conf_threshold
        crop_preds_filtered = crop_preds[valid_mask]
        
        if len(crop_preds_filtered) == 0:
            return
        
        # Transform boxes to original image coordinates
        boxes_xyxy = ops.xywh2xyxy(crop_preds_filtered[:, :4].clone())
        
        if is_full:
            # Full image: scale_boxes handles letterbox -> original transform
            ops.scale_boxes(imgsz, boxes_xyxy, ori_shape, ratio_pad=ratio_pad)
        else:
            # Crop: scale_boxes does nothing (ratio_pad=((1,1),(0,0))), then add offset
            ops.scale_boxes(imgsz, boxes_xyxy, resized_shape, ratio_pad=ratio_pad)
            x_min, y_min = slice_coord[0], slice_coord[1]
            boxes_xyxy[:, 0] += x_min
            boxes_xyxy[:, 1] += y_min
            boxes_xyxy[:, 2] += x_min
            boxes_xyxy[:, 3] += y_min
        
        # Clip to image bounds
        h, w = ori_shape
        boxes_xyxy[:, 0].clamp_(0, w)
        boxes_xyxy[:, 1].clamp_(0, h)
        boxes_xyxy[:, 2].clamp_(0, w)
        boxes_xyxy[:, 3].clamp_(0, h)
        
        # Store as xywh (required for NMS later)
        crop_preds_transformed = crop_preds_filtered.clone()
        crop_preds_transformed[:, :4] = ops.xyxy2xywh(boxes_xyxy)
        
        self.image_crops[img_key]['predictions'].append(crop_preds_transformed)
        self.image_crops[img_key]['crop_coords'].append(slice_coord)
    
    def get_aggregated_predictions(self, img_key: str) -> torch.Tensor:
        """
        Get concatenated predictions for a complete image.
        
        Returns:
            Tensor of shape [N, 4 + num_classes*2] with xywh boxes and class predictions
        """
        data = self.image_crops[img_key]
        
        if not data['predictions']:
            # Return empty tensor with correct number of columns
            num_cols = 4
            if hasattr(self.validator, 'nc'):
                num_cols += sum(nc + 1 for nc in self.validator.nc)
            return torch.empty((0, num_cols), device=self.validator.device)
        
        return torch.cat(data['predictions'], dim=0)
    
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
        return [img_key for img_key in self.image_crops if self.is_image_complete(img_key)]
