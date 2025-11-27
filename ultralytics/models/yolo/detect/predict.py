# Ultralytics YOLO 🚀, AGPL-3.0 license

import numpy as np
import torch
import cv2
from pathlib import Path
from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import DEFAULT_CFG, ops, LOGGER, colorstr

from ultralytics.models.yolo.detect.sahi_predict import slice_image, SAHIPredictAggregator


class DetectionPredictor(BasePredictor):
    """
    A class extending the BasePredictor class for prediction based on a detection model.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.detect import DetectionPredictor

        args = dict(model="yolov8n.pt", source=ASSETS)
        predictor = DetectionPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """
    
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize DetectionPredictor with SAHI support."""
        super().__init__(cfg, overrides, _callbacks)
        
        # SAHI support
        self.sahi_enabled = getattr(self.args, 'sahi', False)
        self.sahi_aggregator = None
        if self.sahi_enabled:
            crop_size = getattr(self.args, 'crop_size', 640)
            overlap_ratio = getattr(self.args, 'overlap_ratio', 0.2)
            self.sahi_aggregator = SAHIPredictAggregator(crop_size=crop_size, overlap_ratio=overlap_ratio)
            LOGGER.info(f"SAHI inference enabled: crop_size={crop_size}, overlap_ratio={overlap_ratio}")

    def stream_inference(self, source=None, model=None, *args, **kwargs):
        """Streams real-time inference with SAHI support."""
        if self.sahi_enabled:
            return self._sahi_stream_inference(source, model, *args, **kwargs)
        else:
            return super().stream_inference(source, model, *args, **kwargs)
    
    def _sahi_stream_inference(self, source=None, model=None, *args, **kwargs):
        """SAHI-enabled stream inference that slices images into crops."""
        
        if self.args.verbose:
            LOGGER.info("")
        
        with self._lock:
            self.setup_source(source if source is not None else self.args.source)
            
            # Check if save_dir/ label file exists
            if self.args.save or self.args.save_txt:
                (self.save_dir / "labels" if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)
            
            # Warmup model
            if not self.done_warmup:
                self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs, 3, *self.imgsz))
                self.done_warmup = True
            
            self.seen, self.windows, self.batch = 0, [], None
            profilers = (
                ops.Profile(device=self.device),
                ops.Profile(device=self.device),
                ops.Profile(device=self.device),
            )
            self.run_callbacks("on_predict_start")
            
            for self.batch in self.dataset:
                self.run_callbacks("on_predict_batch_start")
                paths, im0s, s = self.batch
                
                # Reset aggregator for new batch
                self.sahi_aggregator.reset()
                
                # Process each image in the batch with SAHI
                aggregated_preds_list = []
                for img_idx, (im0, img_path) in enumerate(zip(im0s, paths)):
                    orig_shape = im0.shape[:2]
                    
                    # Slice image into crops
                    with profilers[0]:
                        crops = slice_image(im0, self.sahi_aggregator.crop_size, self.sahi_aggregator.overlap_ratio)
                        img_key = str(self.seen + img_idx)
                    
                    # Process each crop
                    for crop_im, crop_coords in crops:
                        # Preprocess crop
                        crop_im_batch = [crop_im]  # Single image batch
                        im = self.preprocess(crop_im_batch)
                        
                        # Inference on crop
                        with profilers[1]:
                            preds = self.inference(im, *args, **kwargs)
                            if self.args.embed:
                                continue
                        
                        # Postprocess crop predictions
                        with profilers[2]:
                            # Apply NMS on crop
                            if not self.nms:
                                m = self.model.model.model[-1]
                                is_multitask = isinstance(m.nc, (list, tuple)) and len(m.nc) > 1
                                agnostic = self.args.agnostic_nms or is_multitask
                                
                                crop_preds_nms = ops.non_max_suppression(
                                    preds,
                                    self.args.conf,
                                    self.args.iou,
                                    agnostic=agnostic,
                                    max_det=self.args.max_det,
                                    classes=self.args.classes,
                                    nc=m.nc,
                                )[0]  # First (and only) image in batch
                                
                                if len(crop_preds_nms) > 0:
                                    # Scale boxes from model output size to crop size
                                    crop_h, crop_w = crop_im.shape[:2]
                                    model_h, model_w = im.shape[2:]
                                    crop_preds_nms[:, :4] = ops.scale_boxes(
                                        (model_h, model_w),
                                        crop_preds_nms[:, :4],
                                        (crop_h, crop_w)
                                    )
                                    
                                    # Convert from xyxy to xywh for aggregation
                                    crop_preds_xywh = crop_preds_nms.clone()
                                    crop_preds_xywh[:, :4] = ops.xyxy2xywh(crop_preds_nms[:, :4])
                                    
                                    # Add to aggregator
                                    self.sahi_aggregator.add_crop_predictions(
                                        img_key, crop_preds_xywh, crop_coords
                                    )
                
                    # Aggregate predictions for this image
                    with profilers[2]:
                        aggregated_preds = self.sahi_aggregator.aggregate_predictions(
                            img_key, orig_shape, conf_threshold=0.001
                        )
                        
                        # Apply final NMS on aggregated predictions to remove duplicates from overlapping crops
                        if len(aggregated_preds) > 0:
                            # Convert back to xyxy for NMS
                            aggregated_xyxy = ops.xywh2xyxy(aggregated_preds[:, :4])
                            aggregated_for_nms = torch.cat([aggregated_xyxy, aggregated_preds[:, 4:]], dim=1)
                            
                            m = self.model.model.model[-1]
                            is_multitask = isinstance(m.nc, (list, tuple)) and len(m.nc) > 1
                            agnostic = self.args.agnostic_nms or is_multitask
                            
                            # Reshape for NMS: [1, num_detections, 6]
                            aggregated_for_nms = aggregated_for_nms.unsqueeze(0)
                            
                            final_preds = ops.non_max_suppression(
                                aggregated_for_nms,
                                self.args.conf,  # Slightly lower conf for final NMS
                                self.args.iou,
                                agnostic=agnostic,
                                max_det=self.args.max_det,
                                classes=self.args.classes,
                                nc=m.nc,
                            )[0]  # First (and only) image
                        else:
                            final_preds = aggregated_preds
                        
                        aggregated_preds_list.append(final_preds)
                
                # Create Results objects
                if not isinstance(im0s, list):
                    im0s_list = ops.convert_torch2numpy_batch(im0s)
                else:
                    im0s_list = im0s
                
                # Store original images for write_results
                self._sahi_orig_imgs = im0s_list
                
                with profilers[2]:
                    self.results = []
                    for pred, orig_img, img_path in zip(aggregated_preds_list, im0s_list, paths):
                        self.results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
                
                self.run_callbacks("on_predict_postprocess_end")
                
                # Visualize, save, write results
                n = len(im0s)
                for i in range(n):
                    self.seen += 1
                    self.results[i].speed = {
                        "preprocess": profilers[0].dt * 1e3 / n,
                        "inference": profilers[1].dt * 1e3 / n,
                        "postprocess": profilers[2].dt * 1e3 / n,
                    }
                    if self.args.verbose or self.args.save or self.args.save_txt or self.args.show:
                        # Convert original image to tensor format for write_results
                        orig_img = self._sahi_orig_imgs[i]
                        orig_img_tensor = torch.from_numpy(orig_img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                        orig_img_tensor = orig_img_tensor.to(self.device)
                        s[i] += self.write_results(i, Path(paths[i]), orig_img_tensor, s)
                
                # Print batch results
                if self.args.verbose:
                    LOGGER.info("\n".join(s))
                
                self.run_callbacks("on_predict_batch_end")
                yield from self.results
        
        # Release assets
        for v in self.vid_writer.values():
            if isinstance(v, cv2.VideoWriter):
                v.release()
        
        # Print final results
        if self.args.verbose and self.seen:
            t = tuple(x.t / self.seen * 1e3 for x in profilers)
            LOGGER.info(
                f"Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess per image at shape "
                f"{(min(self.args.batch, self.seen), 3, *self.imgsz)}" % t
            )
        if self.args.save or self.args.save_txt or self.args.save_crop:
            nl = len(list(self.save_dir.glob("labels/*.txt")))
            s = f"\n{nl} label{'s' * (nl > 1)} saved to {self.save_dir / 'labels'}" if self.args.save_txt else ""
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}{s}")
        self.run_callbacks("on_predict_end")

    def postprocess(self, preds, img, orig_imgs):
        """Post-processes predictions and returns a list of Results objects."""
        # Standard postprocessing
        if not self.nms:
            m = self.model.model.model[-1]  # detect head
            is_multitask = isinstance(m.nc, (list, tuple)) and len(m.nc) > 1
            agnostic = self.args.agnostic_nms or is_multitask
            preds = ops.non_max_suppression(
                preds,
                self.args.conf,
                self.args.iou,
                agnostic=agnostic,
                max_det=self.args.max_det,
                classes=self.args.classes,
                nc=m.nc,
            )
        elif self.engine:
            preds = ops.process_nms_trt_results(preds, self.output_names)
        elif self.onnx:
            preds = ops.process_nms_onnx_results(preds)

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0]):
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results
