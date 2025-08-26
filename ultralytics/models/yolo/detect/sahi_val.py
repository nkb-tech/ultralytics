import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
from ultralytics.utils import LOGGER, ops


class SAHICropAggregator:
    """Aggregates predictions from crops back to full images for SAHI validation."""
    
    def __init__(self, validator):
        """
        Initialize SAHI crop aggregator.
        
        Args:
            validator: Reference to the DetectionValidator instance
        """
        self.validator = validator
        self.reset()
        
    def reset(self):
        """Reset aggregator for new validation run."""
        self.image_crops = defaultdict(lambda: {
            'predictions': [],  # List of predictions from all crops (before NMS)
            'crop_coords': [],  # Coordinates of each crop
            'processed_crops': set(),  # Track which crops we've seen
            'original_shape': None,
            'original_img_idx': None,  # Original image index in dataset
        })
        # Track total expected crops per image (calculated from dataset info)
        self.expected_crops_per_image = {}
        
    def calculate_expected_crops(self, dataset):
        """Calculate expected number of crops for each image based on dataset."""
        if not hasattr(dataset, 'slice_indices'):
            return
            
        # Group slice_indices by original image index
        for slice_info in dataset.slice_indices:
            if len(slice_info) >= 3:  # (img_idx, slice_idx, coords)
                img_idx = slice_info[0]
                if img_idx not in self.expected_crops_per_image:
                    self.expected_crops_per_image[img_idx] = 0
                self.expected_crops_per_image[img_idx] += 1
        
    def add_crop_predictions(self, batch, preds_before_nms, preds_after_nms):
        """
        Add predictions from a batch of crops.
        """
        original_img_idx = batch.get('original_img_idx', [])
        slice_idx = batch.get('slice_idx', [])
        slice_coords = batch.get('slice_coords', [])
        ori_shapes = batch.get('ori_shape', [])
        if preds_before_nms is not None:
            if isinstance(preds_before_nms, tuple):
                preds_before_nms = preds_before_nms[0] if len(preds_before_nms) > 0 else None

        if not original_img_idx:
            LOGGER.warning("SAHI metadata missing in batch, skipping aggregation")
            return False
            
        # Process each crop in the batch
        for i in range(len(original_img_idx)):
            img_idx = original_img_idx[i]
            img_key = str(img_idx)
            
            # Store original shape
            if self.image_crops[img_key]['original_shape'] is None:
                self.image_crops[img_key]['original_shape'] = ori_shapes[i]
                self.image_crops[img_key]['original_img_idx'] = img_idx
            
            # Track this crop
            self.image_crops[img_key]['processed_crops'].add(slice_idx[i])
            
            # Store raw predictions (before NMS) with crop coordinates
            if preds_before_nms is not None and hasattr(preds_before_nms, 'shape'):
                if len(preds_before_nms.shape) == 3:
                    # Format: [batch, outputs, anchors]
                    crop_preds = preds_before_nms[i]  # [outputs, anchors]
                    
                    # Transpose to [anchors, outputs]
                    crop_preds = crop_preds.T
                    conf_threshold = 0.001
                    valid_mask = crop_preds[:, 4] > conf_threshold
                    
                    crop_preds_filtered = crop_preds[valid_mask]
                    
                    if len(crop_preds_filtered) > 0:
                        x_min, y_min, _, _ = slice_coords[i]
                        
                        # Transform box coordinates from crop to original image
                        crop_preds_transformed = crop_preds_filtered.clone()
                        crop_preds_transformed[:, 0] += x_min  # x_center
                        crop_preds_transformed[:, 1] += y_min  # y_center
                
                        
                        self.image_crops[img_key]['predictions'].append(crop_preds_transformed)
                        self.image_crops[img_key]['crop_coords'].append(slice_coords[i])
                else:
                    LOGGER.error(f"Unexpected preds_before_nms shape: {preds_before_nms.shape}")
                    continue
                    
        return True
    
    def get_aggregated_predictions(self, img_key):
        """
        Get aggregated predictions for a complete image.
        
        Returns:
            Tensor: Concatenated predictions from all crops (before NMS)
        """
        data = self.image_crops[img_key]
        if not data['predictions']:
            num_cols = 4  # boxes
            if hasattr(self.validator, 'nc'):
                for nc in self.validator.nc:
                    num_cols += nc + 1  # classes + confidence
            return torch.empty((0, num_cols), device=self.validator.device)
        
        # Concatenate all predictions
        all_preds = torch.cat(data['predictions'], dim=0)  
        return all_preds
    
    def is_image_complete(self, img_key):
        """Check if all crops for an image have been processed."""
        img_idx = self.image_crops[img_key].get('original_img_idx')
        if img_idx is None:
            return False
            
        expected = self.expected_crops_per_image.get(img_idx, 0)
        processed = len(self.image_crops[img_key]['processed_crops'])
        
        return expected > 0 and processed >= expected
    
    def get_completed_images(self):
        """Get list of images that have all crops processed."""
        completed = []
        for img_key in list(self.image_crops.keys()):
            if self.is_image_complete(img_key):
                completed.append(img_key)
        return completed
