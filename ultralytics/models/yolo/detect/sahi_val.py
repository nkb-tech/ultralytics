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
                
        LOGGER.info(f"Expected crops per image calculated: {len(self.expected_crops_per_image)} images")
        
    def add_crop_predictions(self, batch, preds_before_nms, preds_after_nms):
        """
        Add predictions from a batch of crops.
        """
        # Extract metadata
        original_img_idx = batch.get('original_img_idx', [])
        slice_idx = batch.get('slice_idx', [])
        slice_coords = batch.get('slice_coords', [])
        ori_shapes = batch.get('ori_shape', [])
        
        LOGGER.info(f"add_crop_predictions called:")
        if preds_before_nms is not None:
            if isinstance(preds_before_nms, tuple):
                preds_before_nms = preds_before_nms[0] if len(preds_before_nms) > 0 else None
            if hasattr(preds_before_nms, 'shape'):
                LOGGER.info(f"  preds_before_nms shape: {preds_before_nms.shape}")
        
        # If metadata is missing, skip SAHI aggregation
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
                    
                    LOGGER.info(f"  Image {i}: transposed crop_preds shape: {crop_preds.shape}")
                    
                    # Log structure for first image
                    if i == 0 and len(crop_preds) > 0:
                        LOGGER.info(f"    Output structure analysis:")
                        LOGGER.info(f"    Objectness range: [{crop_preds[:, 4].min():.4f}, {crop_preds[:, 4].max():.4f}]")
                        LOGGER.info(f"    Box coord ranges before transform:")
                        LOGGER.info(f"      x: [{crop_preds[:, 0].min():.2f}, {crop_preds[:, 0].max():.2f}]")
                        LOGGER.info(f"      y: [{crop_preds[:, 1].min():.2f}, {crop_preds[:, 1].max():.2f}]")
                    
                    # Filter by objectness confidence
                    conf_threshold = 0.001
                    valid_mask = crop_preds[:, 4] > conf_threshold
                    
                    crop_preds_filtered = crop_preds[valid_mask]
                    
                    LOGGER.info(f"  Image {i}: filtered predictions: {len(crop_preds_filtered)}/{len(crop_preds)}")
                    
                    if len(crop_preds_filtered) > 0:
                        # Get crop info
                        x_min, y_min, x_max, y_max = slice_coords[i]
                        crop_width = x_max - x_min
                        crop_height = y_max - y_min
                        
                        # Transform box coordinates from crop to original image
                        # The predictions are in pixel coordinates relative to the crop
                        crop_preds_transformed = crop_preds_filtered.clone()
                        
                        # Offset coordinates to original image space
                        crop_preds_transformed[:, 0] += x_min  # x_center
                        crop_preds_transformed[:, 1] += y_min  # y_center
                        # Width and height stay the same (they're already in pixels)
                        
                        if i == 0:  # Debug first image
                            LOGGER.info(f"    After transform to original image:")
                            LOGGER.info(f"      x: [{crop_preds_transformed[:, 0].min():.2f}, {crop_preds_transformed[:, 0].max():.2f}]")
                            LOGGER.info(f"      y: [{crop_preds_transformed[:, 1].min():.2f}, {crop_preds_transformed[:, 1].max():.2f}]")
                            LOGGER.info(f"      Crop coords: {slice_coords[i]}")
                            LOGGER.info(f"      Original shape: {self.image_crops[img_key]['original_shape']}")
                        
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
        
        LOGGER.info(f"  get_aggregated_predictions for image {img_key}:")
        LOGGER.info(f"    Number of prediction lists: {len(data['predictions'])}")
        
        if data['predictions']:
            for i, pred in enumerate(data['predictions']):
                LOGGER.info(f"    Predictions[{i}] shape: {pred.shape if hasattr(pred, 'shape') else type(pred)}")
        
        if not data['predictions']:
            # Return empty tensor with correct number of columns
            # Get number of columns from validator
            num_cols = 4  # boxes
            if hasattr(self.validator, 'nc'):
                for nc in self.validator.nc:
                    num_cols += nc + 1  # classes + confidence
            return torch.empty((0, num_cols), device=self.validator.device)
        
        # Concatenate all predictions
        all_preds = torch.cat(data['predictions'], dim=0)
        LOGGER.info(f"    Concatenated predictions shape: {all_preds.shape}")
        
        # Проверим содержимое
        if len(all_preds) > 0:
            LOGGER.info(f"    Sample prediction (first 5 values): {all_preds[0][:5] if len(all_preds[0]) >= 5 else all_preds[0]}")
            LOGGER.info(f"    Box values range: x=[{all_preds[:, 0].min():.2f}, {all_preds[:, 0].max():.2f}], "
                        f"y=[{all_preds[:, 1].min():.2f}, {all_preds[:, 1].max():.2f}]")
            LOGGER.info(f"    Confidence values range: [{all_preds[:, 4].min():.4f}, {all_preds[:, 4].max():.4f}]")
        
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
