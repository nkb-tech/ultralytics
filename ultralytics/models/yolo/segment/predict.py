# Ultralytics YOLO 🚀, AGPL-3.0 license

import numpy as np

from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, nms, ops


class SegmentationPredictor(DetectionPredictor):
    """
    A class extending the DetectionPredictor class for prediction based on a segmentation model.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.segment import SegmentationPredictor

        args = dict(model="yolov8n-seg.pt", source=ASSETS)
        predictor = SegmentationPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initializes the SegmentationPredictor with the provided configuration, overrides, and callbacks."""
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "segment"

    def postprocess(self, preds, img, orig_imgs):
        """Applies non-max suppression and processes detections for each image in an input batch."""
        # PyTorch Segment.forward returns ((detections, proto), raw_dict) for inference.
        # Exported models may return the older (detections, proto) layout.
        if isinstance(preds, (list, tuple)) and len(preds) == 2 and isinstance(preds[0], (list, tuple)):
            det_preds, proto = preds[0]
        else:
            det_preds = preds[0]
            proto = preds[1][-1] if isinstance(preds[1], tuple) else preds[1]

        # nc must be a list for multi-task NMS compatibility (fork modification)
        # Get nc from model architecture - search in multiple places
        nc = None
        
        # Try self.model.nc
        if hasattr(self.model, 'nc') and self.model.nc is not None:
            nc = self.model.nc
        # Try self.model.model.nc (underlying model)
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'nc'):
            nc = self.model.model.nc
        # Try detection head (last layer)
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'model'):
            # Get the Detect/Segment head
            for m in self.model.model.model:
                if hasattr(m, 'nc'):
                    nc = m.nc
                    break
        
        # Fallback: calculate from prediction shape
        # preds[0] shape: [batch, channels, anchors] where channels = 4 + nc + 32 (mask coeffs)
        if nc is None:
            pred_channels = det_preds.shape[1]
            nc = pred_channels - 4 - 32  # bbox(4) + classes(nc) + masks(32)
            if nc < 1:
                nc = len(self.model.names)  # last resort
        
        # Ensure nc is a list for multi-task NMS
        if isinstance(nc, int):
            nc_list = [nc]
        elif isinstance(nc, (list, tuple)):
            nc_list = list(nc)
        else:
            nc_list = [nc]
        p = nms.non_max_suppression(
            det_preds,
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=nc_list,
            classes=self.args.classes,
        )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []

        # Calculate mask coefficient start column based on proto channels
        # Fork's NMS output: [x1,y1,x2,y2, conf0,cls0, conf1,cls1, ..., mask_coeffs]
        # Mask coefficients are always the last `proto.shape[0]` columns (typically 32)
        n_mask_coeffs = proto.shape[1] if proto.dim() == 3 else 32  # proto shape: (batch, channels, h, w) or (channels, h, w)
        
        for i, (pred, orig_img, img_path) in enumerate(zip(p, orig_imgs, self.batch[0])):
            if not len(pred):  # save empty boxes
                masks = None
            else:
                # Dynamically find mask coefficient start: last n_mask_coeffs columns
                mask_start_col = pred.shape[1] - n_mask_coeffs
                mask_coeffs = pred[:, mask_start_col:]
                
                if self.args.retina_masks:
                    pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
                    masks = ops.process_mask_native(proto[i], mask_coeffs, pred[:, :4], orig_img.shape[:2])  # HWC
                else:
                    masks = ops.process_mask(proto[i], mask_coeffs, pred[:, :4], img.shape[2:], upsample=True)  # HWC
                    pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            
            # Handle 16-bit single-channel images (e.g., X-ray images)
            # Convert 16-bit to 8-bit and triple single channel to 3 channels
            if orig_img.dtype == np.uint16:
                orig_img = (orig_img >> 8).astype(np.uint8)
            if orig_img.ndim == 2:
                orig_img = orig_img[..., None]
            if orig_img.shape[2] == 1:
                orig_img = np.repeat(orig_img, 3, axis=2)
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], masks=masks))
        return results
