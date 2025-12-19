# Ultralytics YOLO 🚀, AGPL-3.0 license

"""
YOLO Segmentation Validator.

This module provides the SegmentationValidator class for validating instance
segmentation models. Extends DetectionValidator with mask support.

Features:
    - Instance segmentation with polygon masks
    - SAHI integration for large images
    - Mask IoU computation
    - Both box and mask metrics
    - Multiple output formats (COCO JSON, TXT)

SAHI Mask Generation:
    For SAHI mode, masks are generated after NMS by:
    1. Looking up which crop each prediction came from
    2. Using the stored proto features and model-space boxes
    3. Generating masks via process_mask()
    4. Transforming masks to original image coordinates
"""

from multiprocessing.pool import ThreadPool
from pathlib import Path
import cv2

import numpy as np
import torch
import torch.nn.functional as F

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.models.yolo.detect.sahi_val import _fix_ratio_pad, _to_tuple
from ultralytics.utils import LOGGER, NUM_THREADS, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.metrics import SegmentMetrics, box_iou, mask_iou
from ultralytics.utils.plotting import output_to_target, plot_images


class SegmentationValidator(DetectionValidator):
    """
    Segmentation validator extending DetectionValidator with mask support.
    
    Supports:
        - Standard instance segmentation
        - SAHI mode for high-resolution images
        - Both box and mask metrics
    
    Attributes:
        plot_masks: List of masks for plotting
        process: Mask processing function (native or standard)
        metrics: List of SegmentMetrics for each task
    
    Example:
        ```python
        from ultralytics.models.yolo.segment import SegmentationValidator
        
        args = dict(model="yolov8n-seg.pt", data="coco8-seg.yaml")
        validator = SegmentationValidator(args=args)
        validator()
        ```
    """

    # ==================== Initialization ====================

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """
        Initialize SegmentationValidator.
        
        Args:
            dataloader: Validation dataloader
            save_dir: Directory to save results
            pbar: Progress bar
            args: Validation arguments
            _callbacks: Callback functions
        """
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.plot_masks = None
        self.process = None
        self.args.task = "segment"
        self.metrics = [SegmentMetrics(save_dir=self.save_dir, on_plot=self.on_plot)]

    def init_metrics(self, model):
        """
        Initialize metrics for segmentation validation.
        
        Sets up:
            - Mask processing function
            - SegmentMetrics with mask support
            - SAHI aggregator if enabled
        
        Args:
            model: YOLO segmentation model
        """
        super().init_metrics(model)
        self.plot_masks = []
        
        # Choose mask processing function
        if self.args.save_json:
            check_requirements("pycocotools>=2.0.6")
        self.process = ops.process_mask_native if self.args.save_json or self.args.save_txt else ops.process_mask
        
        # Override stats for segmentation (add tp_m for mask metrics)
        self.stats = [
            dict(tp_m=[], tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[]) 
            for _ in self.nc
        ]
        
        # Override metrics with SegmentMetrics
        self.metrics = [
            SegmentMetrics(save_dir=self.save_dir, on_plot=self.on_plot, names=names)
            for names in self.names
        ]
        
        # Setup SAHI segmentation aggregator
        if self.sahi_enabled:
            self._setup_sahi_segmentation()

    def _setup_sahi_segmentation(self):
        """
        Setup SAHI mask aggregator for segmentation.
        
        Note: The aggregator is typically already set up in get_dataloader().
        This method serves as a fallback or re-initialization if needed.
        """
        # Check if aggregator is already correctly set up
        from ultralytics.models.yolo.segment.sahi_val import SAHISegmentAggregator
        if isinstance(self.sahi_aggregator, SAHISegmentAggregator):
            # Already correctly set up, just initialize tracking variables
            self._last_raw_preds = None
            self._last_proto = None
            return
        
        # Fallback: set up aggregator if it wasn't set up correctly
        try:
            self.sahi_aggregator = SAHISegmentAggregator(self)
            if hasattr(self.dataloader, 'dataset'):
                self.sahi_aggregator.calculate_expected_crops(self.dataloader.dataset)
            self._last_raw_preds = None
            self._last_proto = None
        except ImportError as e:
            LOGGER.warning(f"Could not import SAHISegmentAggregator: {e}")
            self.sahi_enabled = False

    def get_desc(self):
        """Return formatted description of evaluation metrics."""
        return ("%22s" + "%11s" * 10) % (
            "Class", "Images", "Instances",
            "Box(P", "R", "mAP50", "mAP50-95)",
            "Mask(P", "R", "mAP50", "mAP50-95)",
        )

    # ==================== Dataset & Dataloader ====================

    def get_dataloader(self, dataset_path, batch_size):
        """
        Construct and return dataloader for segmentation validation.
        
        Overrides DetectionValidator.get_dataloader to use SAHISegmentAggregator
        instead of SAHICropAggregator for proper mask handling.
        
        Args:
            dataset_path: Path to dataset
            batch_size: Batch size
            
        Returns:
            DataLoader instance
        """
        from ultralytics.data import build_dataloader
        from ultralytics.data.sahi_dataset import SAHIDataset
        from ultralytics.models.yolo.segment.sahi_val import SAHISegmentAggregator
        
        dataset = self.build_dataset(dataset_path, batch=batch_size, mode="val")
        if isinstance(dataset, SAHIDataset):
            self.sahi_enabled = True
            self.sahi_aggregator = SAHISegmentAggregator(self)
            self.sahi_aggregator.calculate_expected_crops(dataset)
            LOGGER.info(f"SAHI validation enabled: {len(dataset.im_files)} images, {len(dataset)} crops")
        else:
            self.sahi_enabled = False
            self.sahi_aggregator = None
        return build_dataloader(dataset, batch_size, self.args.workers, shuffle=False, rank=-1, drop_last=False)

    # ==================== Data Processing ====================

    def preprocess(self, batch):
        """
        Preprocess batch for segmentation.
        
        Extends parent preprocessing with mask handling.
        
        Args:
            batch: Batch dict from dataloader
            
        Returns:
            Preprocessed batch with masks on device
        """
        batch = super().preprocess(batch)
        batch["masks"] = batch["masks"].to(self.device).float()
        return batch

    def postprocess(self, preds):
        """
        Post-process YOLO predictions for segmentation.
        
        Returns both detections and proto features.
        
        Args:
            preds: Model predictions (detections + proto)
            
        Returns:
            Tuple of (post-NMS detections, proto tensor)
        """
        sahi_enabled = getattr(self, 'sahi_enabled', False)
        
        # Clone raw predictions BEFORE NMS modifies them in-place
        if sahi_enabled:
            self._last_raw_preds = preds[0].clone()
        
        # Apply NMS
        p = ops.non_max_suppression(
            preds[0],
            self.args.conf,
            self.args.iou,
            labels=self.lb,
            multi_label=True,
            agnostic=self.args.single_cls or self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=self.nc,
        )
        
        # Extract proto features
        proto = preds[1][-1] if isinstance(preds[1], (list, tuple)) and len(preds[1]) == 3 else preds[1]
        
        if sahi_enabled:
            self._last_proto = proto
        
        return p, proto

    def _prepare_batch(self, si, batch):
        """
        Prepare a single sample for segmentation validation.
        
        Adds masks to the prepared batch from parent.
        
        Args:
            si: Sample index in batch
            batch: Full batch dict
            
        Returns:
            Prepared batch dict with masks
        """
        pbatch = super()._prepare_batch(si, batch)
        
        # Fix ratio_pad format
        if "ratio_pad" in pbatch:
            pbatch["ratio_pad"] = _fix_ratio_pad(_to_tuple(pbatch["ratio_pad"]))
        
        # Add masks
        midx = [si] if self.args.overlap_mask else batch["batch_idx"] == si
        pbatch["masks"] = batch["masks"][midx]
        pbatch["batch_idx_single"] = si
        
        return pbatch

    def _prepare_pred(self, pred, pbatch, proto):
        """
        Prepare predictions with masks for validation.
        
        Generates instance masks using proto features.
        
        Args:
            pred: Predictions tensor
            pbatch: Prepared batch dict
            proto: Proto features tensor
            
        Returns:
            Tuple of (scaled predictions, prediction masks)
        """
        predn = pred.clone()
        ops.scale_boxes(
            pbatch["imgsz"], predn[:, :4], pbatch["ori_shape"], 
            ratio_pad=pbatch["ratio_pad"]
        )
        
        # Handle proto shape
        proto_in = proto
        if isinstance(proto, (list, tuple)):
            proto_in = proto[-1]
        if isinstance(proto_in, torch.Tensor) and proto_in.dim() == 4:
            proto_in = proto_in[pbatch.get("batch_idx_single", 0)]
        
        # Calculate mask coefficient start column
        # Format: [x1, y1, x2, y2, conf0, cls0, conf1, cls1, ..., mask_coeffs(32)]
        num_tasks = len(self.nc) if hasattr(self, 'nc') else 1
        mask_start_col = 4 + 2 * num_tasks
        
        pred_masks = self.process(proto_in, pred[:, mask_start_col:], pred[:, :4], shape=pbatch["imgsz"])
        return predn, pred_masks

    # ==================== Metrics Update ====================

    def update_metrics(self, preds, batch):
        """
        Update metrics with predictions.
        
        Routes to SAHI or standard processing based on mode.
        
        Args:
            preds: Tuple of (detections, proto)
            batch: Current batch dict
        """
        if self.sahi_enabled and self.sahi_aggregator is not None:
            self._update_metrics_sahi(preds, batch)
        else:
            self._update_metrics_standard(preds, batch)

    def _update_metrics_standard(self, preds, batch):
        """
        Standard segmentation metrics update.
        
        Computes both box and mask metrics for each prediction.
        """
        for si, pred in enumerate(preds[0]):
            self.seen += 1
            npr = len(pred)
            
            # Initialize statistics for all tasks
            stat = [
                dict(
                    conf=torch.zeros(0, device=self.device),
                    pred_cls=torch.zeros(0, device=self.device),
                    tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
                    tp_m=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
                )
                for _ in range(self.num_tasks)
            ]
            
            pbatch = self._prepare_batch(si, batch)
            cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")
            nl = len(cls)
            
            # Store target info for all tasks
            for t in range(self.num_tasks):
                gt_cls = cls[:, t] if cls.dim() > 1 else cls
                stat[t]["target_cls"] = gt_cls
                stat[t]["target_img"] = gt_cls.unique()
            
            if npr == 0:
                if nl:
                    for t in range(self.num_tasks):
                        for k in self.stats[t].keys():
                            self.stats[t][k].append(stat[t][k])
                        if self.args.plots:
                            self.confusion_matrices[t].process_batch(
                                detections=None, gt_bboxes=bbox, gt_cls=cls[:, t] if cls.dim() > 1 else cls
                            )
                continue

            gt_masks = pbatch.pop("masks")
            
            if self.args.single_cls:
                pred[:, 5] = 0
            
            predn, pred_masks = self._prepare_pred(pred, pbatch, preds[1])
            
            # Process each task separately
            for t in range(self.num_tasks):
                # Extract conf/cls for task t (format: [x1,y1,x2,y2, conf0,cls0, conf1,cls1, ..., mask_coeffs])
                stat[t]["conf"] = predn[..., 4 + 2 * t]
                stat[t]["pred_cls"] = predn[..., 5 + 2 * t]
                
                gt_cls = cls[:, t] if cls.dim() > 1 else cls
                
                if nl:
                    stat[t]["tp"] = self._process_batch(predn, bbox, gt_cls)
                    stat[t]["tp_m"] = self._process_batch(
                        predn, bbox, gt_cls, pred_masks, gt_masks, 
                        self.args.overlap_mask, masks=True
                    )
                    if self.args.plots:
                        det = predn[..., [0, 1, 2, 3, 4 + 2 * t, 5 + 2 * t]]
                        self.confusion_matrices[t].process_batch(det, bbox, gt_cls)
                
                for k in self.stats[t].keys():
                    self.stats[t][k].append(stat[t][k])

            # Store masks for plotting
            pred_masks = torch.as_tensor(pred_masks, dtype=torch.uint8)
            if self.args.plots and self.batch_i < 3:
                self.plot_masks.append(pred_masks[:15].cpu())

            # Save outputs
            if self.args.save_json:
                self.pred_to_json(
                    predn,
                    batch["im_file"][si],
                    ops.scale_image(
                        pred_masks.permute(1, 2, 0).contiguous().cpu().numpy(),
                        pbatch["ori_shape"],
                        ratio_pad=batch["ratio_pad"][si],
                    ),
                )
            if self.args.save_txt:
                self.save_one_txt(
                    predn,
                    pred_masks,
                    self.args.save_conf,
                    pbatch["ori_shape"],
                    self.save_dir / "labels" / f'{Path(batch["im_file"][si]).stem}.txt',
                )

    def _update_metrics_sahi(self, preds, batch):
        """
        SAHI segmentation metrics update.
        
        Collects predictions and proto features, processes complete images.
        """
        has_raw = hasattr(self, '_last_raw_preds') and self._last_raw_preds is not None
        has_proto = hasattr(self, '_last_proto') and self._last_proto is not None
        
        if not (has_raw and has_proto):
            LOGGER.warning(f"Raw predictions not available for SAHI (raw={has_raw}, proto={has_proto})")
            return
        
        raw_preds = self._last_raw_preds
        proto = self._last_proto
        
        success = self.sahi_aggregator.add_crop_predictions(batch, raw_preds, preds[0], proto)
        
        # Clear references
        self._last_raw_preds = None
        self._last_proto = None
        
        if not success:
            LOGGER.warning("Cannot add_crop_predictions for SAHI segmentation")
            return
        
        # Process completed images
        for img_key in self.sahi_aggregator.get_completed_images():
            try:
                self._process_complete_image_segment(img_key)
            except Exception as e:
                LOGGER.error(f"Error processing complete image {img_key}: {e}")
                import traceback
                traceback.print_exc()
            finally:
                self.sahi_aggregator.cleanup_image(img_key)

    # ==================== SAHI Image Processing ====================

    def _process_complete_image_segment(self, img_key):
        """
        Process a complete SAHI image for segmentation.
        
        Steps:
            1. Get aggregated predictions
            2. Apply NMS with tracking indices
            3. Get ground truth and masks
            4. Generate prediction masks
            5. Update metrics
        """
        aggregated_preds_raw = self.sahi_aggregator.get_aggregated_predictions(img_key)
        original_shape = self.sahi_aggregator.image_crops[img_key]['original_shape']
        img_idx = self.sahi_aggregator.image_crops[img_key]['original_img_idx']
        
        mask_start_col = 4 + 2 * len(self.nc)
        
        if len(aggregated_preds_raw) == 0:
            aggregated_preds = torch.empty((0, mask_start_col), device=self.device)
            nms_indices = torch.zeros(0, device=self.device, dtype=torch.long)
        else:
            # Add tracking indices for NMS
            tracking_indices = torch.arange(len(aggregated_preds_raw), device=self.device, dtype=aggregated_preds_raw.dtype).unsqueeze(1)
            preds_with_tracking = torch.cat([aggregated_preds_raw, tracking_indices], dim=1)
            preds_for_nms = preds_with_tracking.unsqueeze(0).permute(0, 2, 1)
            
            nms_results = ops.non_max_suppression(
                preds_for_nms,
                self.args.conf,
                self.args.iou,
                labels=[],
                multi_label=True,
                agnostic=self.args.single_cls or self.args.agnostic_nms,
                max_det=self.args.max_det,
                nc=[1] if self.args.single_cls else self.nc,
            )
            
            if nms_results and len(nms_results[0]) > 0:
                preds_with_ids = nms_results[0]
                nms_indices = preds_with_ids[:, -1].long()
                aggregated_preds = preds_with_ids[:, :-1]
            else:
                aggregated_preds = torch.empty((0, mask_start_col), device=self.device)
                nms_indices = torch.zeros(0, device=self.device, dtype=torch.long)
        
        # Get ground truth
        original_labels = self.dataloader.dataset.labels[img_idx]
        
        if 'cls' in original_labels and len(original_labels['cls']) > 0:
            gt_cls = torch.tensor(original_labels['cls'], device=self.device, dtype=torch.float32)
            gt_bboxes = torch.tensor(original_labels['bboxes'], device=self.device, dtype=torch.float32)
        else:
            gt_cls = torch.empty((0, len(self.nc)), device=self.device, dtype=torch.float32)
            gt_bboxes = torch.empty((0, 4), device=self.device, dtype=torch.float32)
        
        # Ensure gt_cls is 2D with shape (N, n_tasks)
        if gt_cls.dim() == 1:
            gt_cls = gt_cls.unsqueeze(1)
        
        # Build GT masks
        gt_segments = original_labels.get("segments", [])
        gt_masks = self._build_gt_masks(gt_segments, original_shape)
        
        # Generate prediction masks
        pred_masks = self._generate_pred_masks_for_sahi(
            aggregated_preds, nms_indices, img_key, original_shape, mask_start_col
        )
        
        # Store for plotting (keep gt_cls in 2D format for proper plotting)
        if self.args.plots and len(self.plotter._sahi_plot_cache) < 16:
            if img_idx not in self.plotter._sahi_plot_cache:
                max_plot_items = 15
                
                preds_limited = aggregated_preds[:max_plot_items].clone().cpu() if len(aggregated_preds) else aggregated_preds.clone().cpu()
                pred_masks_plot = pred_masks[:max_plot_items].cpu().to(torch.uint8) if pred_masks is not None and len(pred_masks) > 0 else None
                gt_cls_limited = gt_cls[:max_plot_items].clone().cpu() if len(gt_cls) > 0 else gt_cls.clone().cpu()
                gt_bboxes_limited = gt_bboxes[:max_plot_items].clone().cpu() if len(gt_bboxes) > 0 else gt_bboxes.clone().cpu()
                gt_masks_plot = gt_masks[:max_plot_items].cpu().to(torch.uint8) if gt_masks is not None and len(gt_masks) > 0 else None
                
                self.plotter._sahi_plot_cache[img_idx] = {
                    'im_file': self.dataloader.dataset.im_files[img_idx],
                    'original_shape': original_shape,
                    'predictions': preds_limited,
                    'gt_cls': gt_cls_limited,  # 2D: (N, n_tasks)
                    'gt_bboxes': gt_bboxes_limited,
                    'pred_masks': pred_masks_plot,
                    'gt_masks': gt_masks_plot,
                }
        
        # Update metrics for all tasks
        self._update_sahi_image_metrics(
            aggregated_preds, gt_cls, gt_bboxes, gt_masks, pred_masks, original_shape
        )

    def _update_sahi_image_metrics(self, aggregated_preds, gt_cls, gt_bboxes, gt_masks, pred_masks, original_shape):
        """
        Update metrics for a single SAHI-processed image.
                
        Args:
            aggregated_preds: NMS'd predictions
            gt_cls: Ground truth classes (2D: [N, n_tasks])
            gt_bboxes: Ground truth boxes (normalized xywh)
            gt_masks: Ground truth masks
            pred_masks: Prediction masks
            original_shape: (h, w) of original image
        """
        self.seen += 1
        npr = len(aggregated_preds)
        nl = len(gt_cls)
        
        # Initialize statistics for all tasks
        stat = [
            dict(
                conf=torch.zeros(0, device=self.device),
                pred_cls=torch.zeros(0, device=self.device),
                tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
                tp_m=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
            )
            for _ in range(self.num_tasks)
        ]
        
        # Convert GT boxes to pixel xyxy
        h, w = original_shape
        gt_bboxes_xyxy = ops.xywh2xyxy(gt_bboxes) * torch.tensor([w, h, w, h], device=self.device) if nl > 0 else torch.empty((0, 4), device=self.device)
        
        # Ensure gt_cls is 2D
        if gt_cls.dim() == 1:
            gt_cls = gt_cls.unsqueeze(1)
        
        # Store target info for all tasks
        for t in range(self.num_tasks):
            gt_cls_task = gt_cls[:, t] if gt_cls.shape[1] > t else gt_cls[:, 0]
            stat[t]["target_cls"] = gt_cls_task
            stat[t]["target_img"] = gt_cls_task.unique()
        
        if npr == 0:
            if nl:
                for t in range(self.num_tasks):
                    for k in self.stats[t].keys():
                        self.stats[t][k].append(stat[t][k])
                    if self.args.plots:
                        gt_cls_task = gt_cls[:, t] if gt_cls.shape[1] > t else gt_cls[:, 0]
                        self.confusion_matrices[t].process_batch(
                            detections=None, gt_bboxes=gt_bboxes_xyxy, gt_cls=gt_cls_task
                        )
            return
        
        if self.args.single_cls:
            aggregated_preds[:, 5] = 0
        
        # Process each task separately
        for t in range(self.num_tasks):
            gt_cls_task = gt_cls[:, t] if gt_cls.shape[1] > t else gt_cls[:, 0]
            
            # Extract conf/cls for task t (format: [x1,y1,x2,y2, conf0,cls0, conf1,cls1, ..., mask_coeffs])
            stat[t]["conf"] = aggregated_preds[..., 4 + 2 * t]
            stat[t]["pred_cls"] = aggregated_preds[..., 5 + 2 * t]
            
            if nl > 0:
                stat[t]["tp"] = self._process_batch(aggregated_preds, gt_bboxes_xyxy, gt_cls_task)
                
                if pred_masks is not None and len(gt_masks):
                    stat[t]["tp_m"] = self._process_batch(
                        aggregated_preds, gt_bboxes_xyxy, gt_cls_task,
                        pred_masks, gt_masks, overlap=False, masks=True
                    )
                
                if self.args.plots:
                    det = aggregated_preds[..., [0, 1, 2, 3, 4 + 2 * t, 5 + 2 * t]]
                    self.confusion_matrices[t].process_batch(det, gt_bboxes_xyxy, gt_cls_task)
            
            for k in self.stats[t].keys():
                self.stats[t][k].append(stat[t][k])

    # ==================== Mask Generation ====================

    def _build_gt_masks(self, segments, original_shape):
        """
        Build ground truth masks from polygon segments.
        
        Args:
            segments: List of polygon segments (normalized coordinates)
            original_shape: (h, w) of image
            
        Returns:
            Tensor [N, H, W] of boolean masks
        """
        h, w = original_shape
        gt_masks_list = []
        
        if segments:
            for seg in segments:
                if seg is None or len(seg) < 3:
                    continue
                poly = np.array(seg, dtype=np.float32).copy()
                poly[:, 0] *= w
                poly[:, 1] *= h
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [poly.astype(np.int32)], 1)
                gt_masks_list.append(mask)
        
        if gt_masks_list:
            return torch.from_numpy(np.stack(gt_masks_list)).to(self.device).bool()
        else:
            return torch.zeros((0, h, w), device=self.device, dtype=torch.bool)

    def _generate_pred_masks_for_sahi(self, aggregated_preds, nms_indices, img_key, original_shape, mask_start_col):
        """
        Generate prediction masks for SAHI-aggregated detections.
        
        Uses stored proto features and model-space boxes to generate masks,
        then transforms them to original image coordinates.
        
        Groups detections by source crop for efficient batch processing.
        
        Args:
            aggregated_preds: Post-NMS predictions
            nms_indices: Indices mapping NMS results to original predictions
            img_key: Image key in aggregator
            original_shape: (h, w) of original image
            mask_start_col: Column index where mask coefficients start
            
        Returns:
            Tensor [N, H, W] of boolean masks, or None
        """
        if len(aggregated_preds) == 0:
            return None
        
        if aggregated_preds.shape[1] <= mask_start_col:
            return None
        
        h, w = original_shape
        pred_masks = torch.zeros((len(aggregated_preds), h, w), device=self.device, dtype=torch.bool)
        mask_coeffs_all = aggregated_preds[:, mask_start_col:]
        
        # Group detections by source crop for efficiency
        from collections import defaultdict
        crop_detections = defaultdict(list)
        
        for det_idx, orig_idx in enumerate(nms_indices):
            orig_idx_int = orig_idx.item()
            meta, local_idx = self.sahi_aggregator.get_crop_metadata_for_index(img_key, orig_idx_int)
            
            if meta is None or meta.get('proto') is None:
                continue
            
            boxes_model = meta['boxes_model']
            if local_idx >= len(boxes_model):
                continue
            
            crop_id = meta['pred_start_idx']
            crop_detections[crop_id].append((det_idx, local_idx, meta))
        
        # Process each crop's detections in batch
        for crop_id, detections in crop_detections.items():
            if not detections:
                continue
            
            _, _, meta = detections[0]
            
            proto = meta['proto']
            boxes_model = meta['boxes_model']
            imgsz = meta['imgsz']
            
            det_indices = [d[0] for d in detections]
            local_indices = [d[1] for d in detections]
            
            batch_boxes = boxes_model[local_indices]
            batch_coeffs = mask_coeffs_all[det_indices]
            
            try:
                masks_lb = ops.process_mask(proto, batch_coeffs, batch_boxes, shape=imgsz, upsample=True)
            except Exception as e:
                LOGGER.warning(f"process_mask failed: {e}, coeffs={batch_coeffs.shape}, proto={proto.shape}, boxes={batch_boxes.shape}")
                continue
            
            if masks_lb is None or masks_lb.numel() == 0:
                continue
            
            for i, det_idx in enumerate(det_indices):
                mask_lb = masks_lb[i].float()
                box_model = batch_boxes[i]
                box_orig = aggregated_preds[det_idx, :4]
                
                pred_mask = self._transform_mask_to_original(
                    mask_lb, box_model, box_orig, meta, original_shape
                )
                
                if pred_mask is not None:
                    pred_masks[det_idx] = pred_mask
        
        return pred_masks

    def _transform_mask_to_original(self, mask_lb, box_model, box_orig, meta, original_shape):
        """
        Transform a mask from letterbox/model space to original image coordinates.
        
        Args:
            mask_lb: Mask in letterbox space [H, W]
            box_model: Box in model space (xyxy)
            box_orig: Box in original image space (xyxy)
            meta: Crop metadata dict
            original_shape: (h, w) of original image
            
        Returns:
            Boolean mask in original image coordinates, or None
        """
        h_orig, w_orig = original_shape
        imgsz_h, imgsz_w = meta['imgsz']
        
        # Extract mask region from model space
        mx1, my1, mx2, my2 = box_model.int().tolist()
        mx1, my1 = max(0, mx1), max(0, my1)
        mx2, my2 = min(imgsz_w, mx2), min(imgsz_h, my2)
        
        if mx2 <= mx1 or my2 <= my1:
            return None
        
        mask_region = mask_lb[my1:my2, mx1:mx2]
        
        # Get target coordinates in original image
        ox1, oy1, ox2, oy2 = box_orig.int().tolist()
        ox1, oy1 = max(0, ox1), max(0, oy1)
        ox2, oy2 = min(w_orig, ox2), min(h_orig, oy2)
        
        if ox2 <= ox1 or oy2 <= oy1:
            return None
        
        target_h, target_w = oy2 - oy1, ox2 - ox1
        
        # Resize mask if needed
        if mask_region.shape[0] != target_h or mask_region.shape[1] != target_w:
            if mask_region.numel() > 0:
                mask_region = F.interpolate(
                    mask_region.unsqueeze(0).unsqueeze(0),
                    size=(target_h, target_w),
                    mode='bilinear',
                    align_corners=False
                )[0, 0]
            else:
                return None
        
        # Create output mask
        output_mask = torch.zeros((h_orig, w_orig), device=mask_lb.device, dtype=torch.bool)
        output_mask[oy1:oy2, ox1:ox2] = mask_region.ge(0.5)
        
        return output_mask

    # ==================== Batch Processing ====================

    def _process_batch(self, detections, gt_bboxes, gt_cls, pred_masks=None, gt_masks=None, overlap=False, masks=False):
        """
        Compute correct prediction matrix for a batch.
        
        Handles both box and mask IoU computation.
        
        Args:
            detections: Detection tensor
            gt_bboxes: Ground truth boxes
            gt_cls: Ground truth classes
            pred_masks: Prediction masks (optional)
            gt_masks: Ground truth masks (optional)
            overlap: Whether GT masks are overlapping
            masks: Whether to compute mask IoU
            
        Returns:
            Tensor [N, niou] of true positive flags
        """
        if masks:
            if gt_masks is None or pred_masks is None:
                return torch.zeros(len(detections), self.niou, dtype=torch.bool, device=self.device)
            
            pred_masks_f = pred_masks.float()
            gt_masks_f = gt_masks.float()
            nl = len(gt_cls)

            # Decode merged GT mask if needed
            if gt_masks_f.shape[0] == 1 and nl > 1:
                merged = gt_masks_f[0]
                decoded = []
                for cid in range(1, nl + 1):
                    decoded.append((merged == cid).float())
                gt_masks_f = torch.stack(decoded, dim=0)

            # Choose target shape and resize if needed
            if gt_masks_f.numel() and pred_masks_f.numel():
                target_shape = gt_masks_f.shape[1:]
            elif pred_masks_f.numel():
                target_shape = pred_masks_f.shape[1:]
            else:
                target_shape = None

            if target_shape is not None:
                if pred_masks_f.shape[1:] != target_shape:
                    pred_masks_f = F.interpolate(
                        pred_masks_f[None], target_shape, mode="bilinear", align_corners=False
                    )[0]
                if gt_masks_f.shape[1:] != target_shape:
                    gt_masks_f = F.interpolate(
                        gt_masks_f[None], target_shape, mode="bilinear", align_corners=False
                    )[0]

            # Handle overlap mode
            if overlap:
                if gt_masks_f.shape[0] == 1 and nl > 1:
                    try:
                        merged = gt_masks_f[0]
                        decoded = torch.zeros((nl, *merged.shape), device=gt_masks_f.device, dtype=gt_masks_f.dtype)
                        for idx in range(nl):
                            decoded[idx] = (merged == (idx + 1)).float()
                        gt_masks_f = decoded
                        overlap = False
                    except Exception:
                        overlap = False
                else:
                    overlap = False

                if overlap:
                    index = torch.arange(nl, device=gt_masks_f.device).view(nl, 1, 1) + 1
                    gt_masks_f = gt_masks_f.repeat(nl, 1, 1)
                    gt_masks_f = torch.where(gt_masks_f == index, 1.0, 0.0)

            iou = mask_iou(
                gt_masks_f.view(gt_masks_f.shape[0], -1),
                pred_masks_f.view(pred_masks_f.shape[0], -1),
            )
        else:
            iou = box_iou(gt_bboxes, detections[:, :4])

        return self.match_predictions(detections[:, 5], gt_cls, iou)

    # ==================== Results & Statistics ====================

    def get_stats(self):
        """
        Compute and return metrics statistics.
        
        Handles empty stats gracefully.
        
        Returns:
            Dict with all task metrics
        """
        results = {}
        self.nt_per_class, self.nt_per_image = [], []
        fitness_values = []
        
        for i, (m, st) in enumerate(zip(self.metrics, self.stats)):
            if not st or not any(len(v) > 0 for v in st.values()):
                stats = {
                    'tp': np.zeros((0, self.niou)),
                    'tp_m': np.zeros((0, self.niou)),
                    'conf': np.zeros(0),
                    'pred_cls': np.zeros(0),
                    'target_cls': np.zeros(0),
                }
                self.nt_per_class.append(np.zeros(self.nc[i], dtype=int))
                self.nt_per_image.append(np.zeros(self.nc[i], dtype=int))
            else:
                stats = {k: torch.cat(v, 0).cpu().numpy() if v else np.zeros(0) for k, v in st.items()}
                ntc = np.bincount(stats["target_cls"].astype(int), minlength=self.nc[i])
                nti = np.bincount(stats.get("target_img", stats["target_cls"]).astype(int), minlength=self.nc[i])
                self.nt_per_class.append(ntc)
                self.nt_per_image.append(nti)
            
            stats.pop("target_img", None)
            
            if len(stats.get('tp', [])) and stats["tp"].any():
                m.process(**stats)
            
            for k, v in m.results_dict.items():
                results[f"task{i}_{k}"] = v
            if f"task{i}_fitness" in results:
                fitness_values.append(results[f"task{i}_fitness"])
        
        if fitness_values:
            results["fitness"] = np.mean(fitness_values)

        return results

    # ==================== Finalization ====================

    def finalize_metrics(self, *args, **kwargs):
        """
        Finalize metrics after validation.
        
        Processes any remaining incomplete SAHI images.
        """
        if self.sahi_enabled and self.sahi_aggregator is not None:
            remaining_images = list(self.sahi_aggregator.image_crops.keys())
            
            if remaining_images:
                LOGGER.warning(f"Processing {len(remaining_images)} incomplete images at validation end")
                for img_key in remaining_images:
                    try:
                        self._process_complete_image_segment(img_key)
                    except Exception as e:
                        LOGGER.error(f"Error processing incomplete image {img_key}: {e}")
                    finally:
                        if not self.keep_sahi_images:
                            self.sahi_aggregator.cleanup_image(img_key)
        
        super().finalize_metrics(*args, **kwargs)

    # ==================== Plotting ====================

    def plot_val_samples(self, batch, ni):
        """Plot validation samples with masks and bounding boxes."""
        if self.sahi_enabled:
            self.plotter.plot_val_samples(batch, ni)
            return
            
        plot_images(
            batch["img"],
            batch["batch_idx"],
            batch["cls"],
            batch["bboxes"],
            masks=batch["masks"],
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def plot_predictions(self, batch, preds, ni):
        """Plot batch predictions with masks."""
        if self.sahi_enabled:
            self.plotter.plot_predictions(batch, preds[0], ni)
            return
            
        plot_images(
            batch["img"],
            *output_to_target(preds[0], max_det=15),
            torch.cat(self.plot_masks, dim=0) if len(self.plot_masks) else self.plot_masks,
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )
        self.plot_masks.clear()

    # ==================== Output Saving ====================

    def save_one_txt(self, predn, pred_masks, save_conf, shape, file):
        """Save YOLO detections to txt file."""
        from ultralytics.engine.results import Results
        
        names = self.names[0] if isinstance(self.names, list) else self.names

        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),
            path=None,
            names=names,
            boxes=predn[:, :6],
            masks=pred_masks,
        ).save_txt(file, save_conf=save_conf)

    def pred_to_json(self, predn, filename, pred_masks):
        """Save predictions to COCO JSON format with RLE masks."""
        from pycocotools.mask import encode

        def single_encode(x):
            rle = encode(np.asarray(x[:, :, None], order="F", dtype="uint8"))[0]
            rle["counts"] = rle["counts"].decode("utf-8")
            return rle

        stem = Path(filename).stem
        image_id = int(stem) if stem.isnumeric() else stem
        box = ops.xyxy2xywh(predn[:, :4])
        box[:, :2] -= box[:, 2:] / 2
        pred_masks = np.transpose(pred_masks, (2, 0, 1))
        
        with ThreadPool(NUM_THREADS) as pool:
            rles = pool.map(single_encode, pred_masks)
        
        for i, (p, b) in enumerate(zip(predn.tolist(), box.tolist())):
            self.jdict.append({
                "image_id": image_id,
                "category_id": self.class_map[int(p[5])],
                "bbox": [round(x, 3) for x in b],
                "score": round(p[4], 5),
                "segmentation": rles[i],
            })
