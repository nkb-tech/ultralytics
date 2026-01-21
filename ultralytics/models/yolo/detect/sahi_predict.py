"""
SAHI (Slicing Aided Hyper Inference) support for prediction/inference.

This module provides utilities for slicing images into crops during inference
and aggregating predictions back to full images.
"""

import numpy as np
import torch
from typing import List, Tuple, Dict, Any
from collections import defaultdict
from ultralytics.utils import LOGGER, ops


def calculate_slice_coordinates(
    img_h: int,
    img_w: int,
    crop_size: int,
    overlap_ratio: float = 0.2
) -> List[Tuple[int, int, int, int]]:
    """
    Calculate coordinates for slicing an image into crops using grid strategy.
    
    Args:
        img_h: Image height
        img_w: Image width
        crop_size: Size of each crop (square)
        overlap_ratio: Overlap ratio between adjacent crops (0 to 1)
    
    Returns:
        List of (x1, y1, x2, y2) coordinates for each crop
    """
    if img_h <= crop_size and img_w <= crop_size:
        return [(0, 0, img_w, img_h)]
    
    overlap = int(overlap_ratio * crop_size)
    step = crop_size - overlap
    
    # Calculate number of steps
    n_steps_h = max(1, (img_h - crop_size + step - 1) // step + 1) if img_h > crop_size else 1
    n_steps_w = max(1, (img_w - crop_size + step - 1) // step + 1) if img_w > crop_size else 1
    
    # Generate base coordinates
    y1_base = np.arange(0, n_steps_h) * step
    x1_base = np.arange(0, n_steps_w) * step
    
    # Adjust last step to align with image edge
    if img_h > crop_size and n_steps_h > 1:
        y1_base[-1] = img_h - crop_size
    if img_w > crop_size and n_steps_w > 1:
        x1_base[-1] = img_w - crop_size
    
    # Generate all crop coordinates
    slices = []
    for y1 in y1_base:
        for x1 in x1_base:
            x2 = min(x1 + crop_size, img_w)
            y2 = min(y1 + crop_size, img_h)
            slices.append((int(x1), int(y1), int(x2), int(y2)))
    
    return slices


def slice_image(im: np.ndarray, crop_size: int, overlap_ratio: float = 0.2) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """
    Slice an image into crops.
    
    Args:
        im: Input image as numpy array (H, W, C)
        crop_size: Size of each crop (square)
        overlap_ratio: Overlap ratio between adjacent crops
    
    Returns:
        List of (crop_image, (x1, y1, x2, y2)) tuples
    """
    h, w = im.shape[:2]
    coords = calculate_slice_coordinates(h, w, crop_size, overlap_ratio)
    
    crops = []
    for x1, y1, x2, y2 in coords:
        crop = im[y1:y2, x1:x2]
        crops.append((crop, (x1, y1, x2, y2)))
    
    return crops


class SAHIPredictAggregator:
    """Aggregates predictions from crops back to full images for SAHI inference."""
    
    def __init__(self, crop_size: int, overlap_ratio: float = 0.2):
        """
        Initialize SAHI prediction aggregator.
        
        Args:
            crop_size: Size of crops used for slicing
            overlap_ratio: Overlap ratio between crops
        """
        self.crop_size = crop_size
        self.overlap_ratio = overlap_ratio
        self.image_predictions = defaultdict(list)  # {img_key: [(preds, coords), ...]}
        
    def reset(self):
        """Reset aggregator for new inference run."""
        self.image_predictions.clear()
    
    def add_crop_predictions(self, img_key: str, preds: torch.Tensor, crop_coords: Tuple[int, int, int, int]):
        """
        Add predictions from a crop.
        
        Args:
            img_key: Unique identifier for the original image
            preds: Predictions tensor (N, 6) where columns are [x, y, w, h, conf, cls]
            crop_coords: (x1, y1, x2, y2) coordinates of the crop in original image
        """
        # Transform boxes from crop coordinates to original image coordinates
        x_min, y_min, _, _ = crop_coords
        
        # Copy predictions to avoid modifying original
        transformed_preds = preds.clone()
        
        # Transform boxes from crop space to original image space
        # preds format: [x_center, y_center, width, height, conf, cls]
        transformed_preds[:, 0] = transformed_preds[:, 0] + x_min  # x_center
        transformed_preds[:, 1] = transformed_preds[:, 1] + y_min  # y_center
        
        self.image_predictions[img_key].append((transformed_preds, crop_coords))
    
    def aggregate_predictions(self, img_key: str, orig_shape: Tuple[int, int], conf_threshold: float = 0.001) -> torch.Tensor:
        """
        Aggregate predictions from all crops for a complete image.
        
        Args:
            img_key: Unique identifier for the original image
            orig_shape: (height, width) of the original image
            conf_threshold: Confidence threshold for filtering predictions
        
        Returns:
            Aggregated predictions tensor (N, 6) in original image coordinates
        """
        if img_key not in self.image_predictions or not self.image_predictions[img_key]:
            # Return empty tensor with correct shape
            return torch.empty((0, 6), dtype=torch.float32)
        
        # Collect all predictions
        all_preds = []
        for preds, _ in self.image_predictions[img_key]:
            # Filter by confidence
            if conf_threshold > 0:
                mask = preds[:, 4] >= conf_threshold
                preds = preds[mask]
            
            if len(preds) > 0:
                all_preds.append(preds)
        
        if not all_preds:
            return torch.empty((0, 6), dtype=torch.float32)
        
        # Concatenate all predictions
        aggregated = torch.cat(all_preds, dim=0)
        
        # Clip boxes to image boundaries
        h, w = orig_shape
        aggregated[:, 0] = torch.clamp(aggregated[:, 0], 0, w)  # x_center
        aggregated[:, 1] = torch.clamp(aggregated[:, 1], 0, h)  # y_center
        aggregated[:, 2] = torch.clamp(aggregated[:, 2], 0, w)  # width
        aggregated[:, 3] = torch.clamp(aggregated[:, 3], 0, h)  # height
        
        return aggregated
    
    def is_image_complete(self, img_key: str, expected_crops: int) -> bool:
        """Check if all crops for an image have been processed."""
        return len(self.image_predictions.get(img_key, [])) >= expected_crops


