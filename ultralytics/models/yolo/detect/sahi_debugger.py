# ultralytics/models/yolo/detect/sahi_debugger.py
import torch
from ultralytics.utils import LOGGER, colorstr

class SAHIValidationDebugger:
    """Helper class to debug SAHI validation process"""
    
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.image_crops = {}
        self.current_image_idx = None
        
    def log_batch_info(self, batch, batch_i):
        """Log information about current batch"""
        if not self.enabled:
            return
            
        LOGGER.info(colorstr('blue', 'bold', f'\n=== Batch {batch_i} Info ==='))
        
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                LOGGER.info(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            elif isinstance(value, list):
                LOGGER.info(f"  {key}: list of length {len(value)}")
                if value and hasattr(value[0], 'shape'):
                    LOGGER.info(f"    First element shape: {value[0].shape}")
            else:
                LOGGER.info(f"  {key}: type={type(value)}")
        
        if 'im_file' in batch:
            LOGGER.info(f"  Image files in batch: {batch['im_file']}")
            
        if 'ori_shape' in batch:
            LOGGER.info(f"  Original shapes: {batch['ori_shape']}")
            
    def track_crop_mapping(self, batch, batch_i):
        """Track which crops belong to which original image"""
        if not self.enabled or 'im_file' not in batch:
            return
            
        for i, im_file in enumerate(batch['im_file']):
            LOGGER.info(f"  Crop {i}: from file {im_file}")
            
    def log_predictions(self, preds, batch, batch_i):
        """Log prediction information"""
        if not self.enabled:
            return
            
        LOGGER.info(colorstr('green', f'  Predictions for batch {batch_i}:'))
        
        if isinstance(preds, list):
            for i, pred in enumerate(preds):
                if pred is not None and len(pred) > 0:
                    LOGGER.info(f"    Image {i}: {len(pred)} detections, shape={pred.shape}")
                else:
                    LOGGER.info(f"    Image {i}: No detections")
        else:
            LOGGER.info(f"    Predictions shape: {preds.shape if hasattr(preds, 'shape') else type(preds)}")
