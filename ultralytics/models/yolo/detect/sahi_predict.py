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
        List of (x1, y1, x2, y2) coordinates for each crop, where coordinates are in the original image space.
    """
    # Early return for images smaller than crop size
    if img_h <= crop_size and img_w <= crop_size:
        return [(0, 0, img_w, img_h)]
    
    # Calculate overlap in pixels and step size (non-overlapping portion)
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
    
    # Extract actual image crops using the calculated coordinates
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
        # Dictionary mapping image keys to lists of (predictions, crop_coords) tuples
        self.image_predictions = defaultdict(list)  # {img_key: [(preds, coords), ...]}
        self.device = None
        
    def reset(self):
        """Reset aggregator for new inference run."""
        self.image_predictions.clear()
        self.device = None
    
    def add_crop_predictions(self, img_key: str, preds: torch.Tensor, crop_coords: Tuple[int, int, int, int]):
        """
        Add predictions from a single crop and transform coordinates to full image space.
        
        This method takes predictions made on a crop (in crop-local coordinates) and
        transforms them to the coordinate system of the original full image by adding
        the crop's offset.
        
        Args:
            img_key: Unique identifier for the original image
            preds: Predictions tensor
            crop_coords: (x1, y1, x2, y2) coordinates of the crop in original image
        """
        # Extract crop offset in the original image
        x_min, y_min = crop_coords[0], crop_coords[1]
        # Clone to avoid modifying the original predictions tensor
        transformed_preds = preds.clone()
        transformed_preds[:, 0] += x_min  # x1
        transformed_preds[:, 2] += x_min  # x2
        transformed_preds[:, 1] += y_min  # y1
        transformed_preds[:, 3] += y_min  # y2
        
        if self.device is None:
            self.device = transformed_preds.device
        
        # Store transformed predictions with their crop coordinates for later aggregation
        self.image_predictions[img_key].append((transformed_preds, crop_coords))
    
    def aggregate_predictions(self, img_key: str, orig_shape: Tuple[int, int], conf_threshold: float = 0.1) -> torch.Tensor:
        """
        Aggregate predictions from all crops for a complete image.
        
        Args:
            img_key: Unique identifier for the original image
            orig_shape: (height, width) of the original image
        
        Returns:
            Aggregated predictions tensor in original image coordinates (xyxy)
        """
        # Check if we have any predictions for this image
        if img_key not in self.image_predictions or not self.image_predictions[img_key]:
            device = self.device or torch.device("cpu")
            return torch.empty((0, 6), dtype=torch.float32, device=device)
        
        # Collect all predictions from crops
        all_preds = [preds for preds, _ in self.image_predictions[img_key]]
        
        # Clean up to prevent memory accumulation (remove predictions after processing)
        self.image_predictions.pop(img_key, None)
        
        # Return empty tensor if no predictions
        if not all_preds:
            device = self.device or torch.device("cpu")
            return torch.empty((0, 6), dtype=torch.float32, device=device)
        
        # Concatenate all predictions
        aggregated = torch.cat(all_preds, dim=0)
        
        # Clip boxes to image boundaries (in-place for efficiency)
        h, w = orig_shape
        aggregated[:, 0].clamp_(0, w)  # x1
        aggregated[:, 2].clamp_(0, w)  # x2
        aggregated[:, 1].clamp_(0, h)  # y1
        aggregated[:, 3].clamp_(0, h)  # y2

        return aggregated
    


