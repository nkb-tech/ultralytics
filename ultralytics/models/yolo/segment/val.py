# Ultralytics YOLO 🚀, AGPL-3.0 license

from __future__ import annotations

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

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, nms, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.metrics import SegmentMetrics, box_iou, mask_iou
from ultralytics.utils.plotting import plot_images


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

    def __init__(
        self,
        dataloader=None,
        save_dir=None,
        pbar=None,
        args=None,
        _callbacks=None,
    ) -> None:
        """Initialize SegmentationValidator and set task to 'segment', metrics to SegmentMetrics.

        Args:
            dataloader (torch.utils.data.DataLoader, optional): DataLoader to use for validation.
            save_dir (Path, optional): Directory to save results.
            pbar (tqdm, optional): Progress bar.
            args (dict, optional): Arguments for the validator.
            _callbacks (list, optional): List of callback functions.
        """
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.plot_masks = None
        self.process = None
        self.args.task = "segment"
        self.metrics = [SegmentMetrics(save_dir=self.save_dir, on_plot=self.on_plot)]
        
        if self.dataloader is not None and hasattr(self.dataloader, 'dataset'):
            from ultralytics.data.sahi_dataset import SAHIDataset
            if isinstance(self.dataloader.dataset, SAHIDataset):
                self.sahi_enabled = True
                from ultralytics.models.yolo.segment.sahi_val import SAHISegmentAggregator
                self.sahi_aggregator = SAHISegmentAggregator(self)
                self.sahi_aggregator.calculate_expected_crops(self.dataloader.dataset)

    def init_metrics(self, model: torch.nn.Module) -> None:
        """Initialize metrics for segmentation validation.

        Sets up mask processing function (process_mask_native vs process_mask),
        overrides stats with tp_m, overrides metrics with SegmentMetrics per task,
        and configures SAHI aggregator when enabled.

        Args:
            model (torch.nn.Module): Model to validate.
        """
        super().init_metrics(model)

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

    def _setup_sahi_segmentation(self) -> None:
        """Setup SAHI mask aggregator for segmentation.

        Verifies aggregator is SAHISegmentAggregator, initializes _last_raw_preds
        and _last_proto. Serves as fallback if get_dataloader did not set it up.
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

    def get_desc(self) -> str:
        """Return a formatted description of evaluation metrics."""
        return ("%22s" + "%11s" * 10) % (
            "Class", "Images", "Instances",
            "Box(P", "R", "mAP50", "mAP50-95)",
            "Mask(P", "R", "mAP50", "mAP50-95)",
        )

    def get_dataloader(self, dataset_path: str, batch_size: int = 16):
        """Construct and return dataloader for segmentation validation.

        Overrides DetectionValidator.get_dataloader to use SAHISegmentAggregator
        instead of SAHICropAggregator when dataset is SAHIDataset.

        Args:
            dataset_path (str): Path to dataset.
            batch_size (int): Batch size.

        Returns:
            (torch.utils.data.DataLoader): DataLoader instance.
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

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Preprocess batch of images for YOLO segmentation validation.

        Extends parent preprocessing with mask handling (move to device, float).

        Args:
            batch (dict[str, Any]): Batch containing images and annotations.

        Returns:
            (dict[str, Any]): Preprocessed batch with masks on device.
        """
        batch = super().preprocess(batch)
        batch["masks"] = batch["masks"].to(self.device).float()
        return batch

    def postprocess(self, preds: list[torch.Tensor] | tuple) -> list[dict[str, torch.Tensor]]:
        """Post-process YOLO predictions for segmentation and return detections with masks.

        Extracts proto from preds[0][1] (tuple) or preds[1], runs parent NMS, then generates
        instance masks from proto coefficients for each detection.

        Args:
            preds (list[torch.Tensor] | tuple): Raw predictions from the model. For segment,
                preds[0] may be (det_tensor, proto) tuple or preds[1] is proto.

        Returns:
            (list[dict[str, torch.Tensor]]): Processed detection predictions with 'bboxes',
                'conf_0', 'cls_0', 'masks' keys per image.
        """
        proto = preds[0][1] if isinstance(preds[0], tuple) else preds[1]
        if getattr(self, "sahi_enabled", False):
            self._last_proto = proto
        result = super().postprocess(preds[0])
        imgsz = [4 * x for x in proto.shape[2:]]
        for i, pred in enumerate(result):
            coefficient = pred.pop("extra")
            pred["masks"] = (
                self.process(proto[i], coefficient, pred["bboxes"], shape=imgsz)
                if coefficient.shape[0]
                else torch.zeros(
                    (0, *(imgsz if self.process is ops.process_mask_native else proto.shape[2:])),
                    dtype=torch.uint8,
                    device=pred["bboxes"].device,
                )
            )
        return result

    def _prepare_batch(self, si: int, batch: dict[str, Any]) -> dict[str, Any]:
        """Prepare a batch for validation by processing images and targets.

        Extends parent with mask preparation: overlap_mask handling, interpolation to
        mask_size, and binarization.

        Args:
            si (int): Sample index within the batch.
            batch (dict[str, Any]): Batch data containing images and annotations.

        Returns:
            (dict[str, Any]): Prepared batch with 'cls', 'bboxes', 'masks', 'ori_shape',
                'imgsz', 'ratio_pad', 'im_file'.
        """
        pbatch = super()._prepare_batch(si, batch)
        nl = pbatch["cls"].shape[0]
        if self.args.overlap_mask:
            masks = batch["masks"][si]
            index = torch.arange(1, nl + 1, device=masks.device).view(nl, 1, 1)
            masks = (masks == index).float()
        else:
            masks = batch["masks"][batch["batch_idx"] == si]
        if nl:
            mask_size = [s if self.process is ops.process_mask_native else s // 4 for s in pbatch["imgsz"]]
            if masks.shape[1:] != mask_size:
                masks = F.interpolate(masks[None], mask_size, mode="bilinear", align_corners=False)[0]
                masks = masks.gt_(0.5)
        pbatch["masks"] = masks
        return pbatch

    def update_metrics(self, preds: list[dict[str, torch.Tensor]], batch: dict[str, Any]) -> None:
        """Update metrics with new predictions and ground truth.

        Routes to SAHI aggregator when sahi_enabled, otherwise uses standard segment
        metrics (appends to self.stats).

        Args:
            preds (list[dict[str, torch.Tensor]]): List of predictions from the model.
            batch (dict[str, Any]): Batch data containing ground truth.
        """
        if self.sahi_enabled and self.sahi_aggregator is not None:
            self._update_metrics_sahi(preds, batch)
        else:
            self._update_metrics_standard(preds, batch)

    def _update_metrics_standard(self, preds: list[dict[str, torch.Tensor]], batch: dict[str, Any]) -> None:
        """Standard segmentation metrics update.

        Iterates over predictions, prepares batch and pred per image, computes tp and tp_m
        via _process_batch, appends to self.stats, handles confusion matrix, save_json,
        save_txt.

        Args:
            preds (list[dict[str, torch.Tensor]]): List of per-image prediction dicts.
            batch (dict[str, Any]): Batch data containing ground truth.
        """
        for si, pred in enumerate(preds):
            self.seen += 1
            pbatch = self._prepare_batch(si, batch)
            predn = self._prepare_pred(pred)

            cls = pbatch["cls"]
            nl = len(cls)
            no_pred = predn["cls_0"].shape[0] == 0

            for t in range(self.num_tasks):
                gt_cls = cls[:, t]
                stat = {"target_cls": gt_cls, "target_img": gt_cls.unique()}

                if no_pred:
                    stat["conf"] = torch.zeros(0, device=self.device)
                    stat["pred_cls"] = torch.zeros(0, device=self.device)
                    stat["tp"] = np.zeros((0, self.niou), dtype=bool)
                    stat["tp_m"] = np.zeros((0, self.niou), dtype=bool)
                else:
                    stat["conf"] = predn[f"conf_{t}"]
                    stat["pred_cls"] = predn[f"cls_{t}"]
                    if nl:
                        stat.update(self._process_batch(predn, pbatch, task=t))
                    else:
                        npr = predn["cls_0"].shape[0]
                        stat["tp"] = np.zeros((npr, self.niou), dtype=bool)
                        stat["tp_m"] = np.zeros((npr, self.niou), dtype=bool)

                for k in self.stats[t]:
                    self.stats[t][k].append(stat[k].cpu().numpy() if hasattr(stat[k], "cpu") else stat[k])

                if self.args.plots and nl:
                    det = torch.cat([predn["bboxes"], predn[f"conf_{t}"].unsqueeze(1), predn[f"cls_{t}"].unsqueeze(1)], 1) if not no_pred else None
                    self.confusion_matrices[t].process_batch(detections=det, gt_bboxes=pbatch["bboxes"], gt_cls=gt_cls)

            if no_pred:
                continue

            if self.args.save_json or self.args.save_txt:
                predn_scaled = self.scale_preds(predn, pbatch)
            if self.args.save_json:
                self.pred_to_json(predn_scaled, pbatch)
            if self.args.save_txt:
                self.save_one_txt(predn_scaled, self.args.save_conf, pbatch["ori_shape"], self.save_dir / "labels" / f"{Path(pbatch['im_file']).stem}.txt")

    def _prepare_pred(self, pred: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Prepare predictions for evaluation against ground truth.

        Creates a copy of pred dict (cloned tensors) and optionally zeros cls_0
        when single_cls is enabled.

        Args:
            pred (dict[str, torch.Tensor]): Post-processed predictions with 'bboxes',
                'conf_0', 'cls_0', 'masks', etc.

        Returns:
            (dict[str, torch.Tensor]): Prepared predictions (copy, optionally modified).
        """
        predn = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in pred.items()}
        if self.args.single_cls:
            predn["cls_0"] = predn["cls_0"] * 0
        return predn

    def _process_batch(
        self, preds: dict[str, torch.Tensor], batch: dict[str, Any], task: int = 0
    ) -> dict[str, np.ndarray]:
        """Compute correct prediction matrix for a batch based on bounding boxes and masks.

        Calls parent for box IoU (tp), then computes mask IoU (tp_m) via mask_iou and
        match_predictions. Aligns with upstream (preds, batch) -> dict signature.

        Args:
            preds (dict[str, torch.Tensor]): Predictions with 'bboxes', 'cls_{task}', 'masks'.
            batch (dict[str, Any]): Batch with 'bboxes', 'cls', 'masks'.
            task (int): Task index for multitask (default 0).

        Returns:
            (dict[str, np.ndarray]): Dictionary with 'tp' (box IoU) and 'tp_m' (mask IoU)
                matrices of shape (N, niou).

        Notes:
            - Overlapping masks are handled based on overlap_mask in batch preparation.
        """
        tp = super()._process_batch(preds, batch, task=task)
        gt_cls = batch["cls"][:, task] if batch["cls"].dim() > 1 else batch["cls"]
        if gt_cls.shape[0] == 0 or preds[f"cls_{task}"].shape[0] == 0:
            tp_m = np.zeros((preds[f"cls_{task}"].shape[0], self.niou), dtype=bool)
        else:
            gt_m = batch["masks"].flatten(1).float()
            pr_m = preds["masks"].flatten(1).float()
            iou = mask_iou(gt_m, pr_m)
            tp_m = self.match_predictions(preds[f"cls_{task}"], gt_cls, iou).cpu().numpy()
        tp["tp_m"] = tp_m
        return tp

    def _update_metrics_sahi(self, preds: list[dict[str, torch.Tensor]], batch: dict[str, Any]) -> None:
        """SAHI segmentation metrics update.

        Collects raw predictions and proto from _last_raw_preds/_last_proto (set in postprocess),
        passes to SAHI aggregator, processes completed images via _process_complete_image_segment.

        Args:
            preds (list[dict[str, torch.Tensor]]): List of predictions (unused; raw from postprocess).
            batch (dict[str, Any]): Batch data for crop metadata.
        """
        has_raw = hasattr(self, '_last_raw_preds') and self._last_raw_preds is not None
        has_proto = hasattr(self, '_last_proto') and self._last_proto is not None
        
        if not (has_raw and has_proto):
            LOGGER.warning(f"Raw predictions not available for SAHI (raw={has_raw}, proto={has_proto})")
            return
        
        raw_preds = self._last_raw_preds
        proto = self._last_proto
        
        success = self.sahi_aggregator.add_crop_predictions(batch, raw_preds, [], proto)
        
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

    def _process_complete_image_segment(self, img_key: str) -> None:
        """Process a complete SAHI image for segmentation.

        Steps:
            1. Get aggregated predictions from SAHI aggregator
            2. Apply NMS with tracking indices to preserve mask coefficients
            3. Load ground truth and build GT masks from segments
            4. Generate prediction masks via _generate_pred_masks_for_sahi
            5. Update metrics via _update_sahi_image_metrics

        Args:
            img_key (str): Key identifying the image in the SAHI aggregator.
        """
        aggregated_preds_raw = self.sahi_aggregator.get_aggregated_predictions(img_key)
        original_shape = self.sahi_aggregator.image_crops[img_key]['original_shape']
        img_idx = self.sahi_aggregator.image_crops[img_key]['original_img_idx']
        
        # For single task: mask_start_col = 6 (after xyxy, conf, cls)
        # NMS output format: [x1, y1, x2, y2, conf, cls, mask_coeffs...]
        mask_start_col = 4 + 2 * len(self.nc)
        
        # nc for NMS should be a list (the fork's NMS uses sum(nc))
        nc_for_nms = [1] if self.args.single_cls else self.nc
        
        if len(aggregated_preds_raw) == 0:
            aggregated_preds = torch.empty((0, mask_start_col), device=self.device)
            nms_indices = torch.zeros(0, device=self.device, dtype=torch.long)
        else:
            # Add tracking indices for NMS - will be preserved as last column of extra data
            tracking_indices = torch.arange(len(aggregated_preds_raw), device=self.device, dtype=aggregated_preds_raw.dtype).unsqueeze(1)
            preds_with_tracking = torch.cat([aggregated_preds_raw, tracking_indices], dim=1)
            preds_for_nms = preds_with_tracking.unsqueeze(0).permute(0, 2, 1)
            
            nms_results = nms.non_max_suppression(
                preds_for_nms,
                self.args.conf,
                self.args.iou,
                labels=[],
                multi_label=True,
                agnostic=self.args.single_cls or self.args.agnostic_nms,
                max_det=self.args.max_det,
                nc=nc_for_nms,  # Pass as list (fork's NMS uses sum(nc))
            )
            
            if nms_results and len(nms_results[0]) > 0:
                preds_with_ids = nms_results[0]
                
                # NMS output format: [x1,y1,x2,y2, conf, cls, extra_data...]
                # The tracking index should be the last column if mask data was preserved
                # Expected columns: 4 + 2*len(nc) + 32 (mask) + 1 (tracking) = 39 for single task
                expected_cols_with_tracking = mask_start_col + 32 + 1  # xyxy + conf/cls + mask + tracking
                
                if preds_with_ids.shape[1] >= expected_cols_with_tracking:
                    # Tracking index was preserved as the last column
                    nms_indices = preds_with_ids[:, -1].long()
                    aggregated_preds = preds_with_ids[:, :-1]  # Remove tracking index
                else:
                    # NMS didn't preserve mask/tracking data - use sequential matching
                    # This is normal when NMS strips extra columns
                    nms_indices = torch.arange(len(preds_with_ids), device=self.device, dtype=torch.long)
                    aggregated_preds = preds_with_ids
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
        plotter = getattr(self, "plotter", None)
        if self.args.plots and plotter is not None and len(plotter._sahi_plot_cache) < 16:
            has_gt = len(gt_cls) > 0 and len(gt_bboxes) > 0

            if has_gt and img_idx not in plotter._sahi_plot_cache:
                max_plot_items = 15
                
                if len(aggregated_preds) > 0:
                    preds_for_plot = aggregated_preds[:max_plot_items, :6].clone().cpu()
                else:
                    preds_for_plot = torch.empty((0, 6))
                
                pred_masks_plot = pred_masks[:max_plot_items].cpu().to(torch.uint8) if pred_masks is not None and len(pred_masks) > 0 else None
                gt_cls_limited = gt_cls[:max_plot_items].clone().cpu() if len(gt_cls) > 0 else gt_cls.clone().cpu()
                gt_bboxes_limited = gt_bboxes[:max_plot_items].clone().cpu() if len(gt_bboxes) > 0 else gt_bboxes.clone().cpu()
                gt_masks_plot = gt_masks[:max_plot_items].cpu().to(torch.uint8) if gt_masks is not None and len(gt_masks) > 0 else None
                
                plotter._sahi_plot_cache[img_idx] = {
                    'im_file': self.dataloader.dataset.im_files[img_idx],
                    'original_shape': original_shape,
                    'predictions': preds_for_plot,
                    'gt_cls': gt_cls_limited,
                    'gt_bboxes': gt_bboxes_limited,
                    'pred_masks': pred_masks_plot,
                    'gt_masks': gt_masks_plot,
                }
        
        # Update metrics for all tasks
        self._update_sahi_image_metrics(
            aggregated_preds, gt_cls, gt_bboxes, gt_masks, pred_masks, original_shape
        )

    def _update_sahi_image_metrics(
        self,
        aggregated_preds: torch.Tensor,
        gt_cls: torch.Tensor,
        gt_bboxes: torch.Tensor,
        gt_masks: torch.Tensor,
        pred_masks: torch.Tensor | None,
        original_shape: tuple[int, int],
    ) -> None:
        """Update metrics for a single SAHI-processed image.

        Builds preds_dict and batch_dict from tensors, calls _process_batch for tp/tp_m,
        appends to self.stats per task.

        Args:
            aggregated_preds (torch.Tensor): NMS'd predictions [M, cols] where cols =
                4 + 2*num_tasks + 32 (mask coefficients).
            gt_cls (torch.Tensor): Ground truth classes (2D: [N, n_tasks]).
            gt_bboxes (torch.Tensor): Ground truth boxes (normalized xywh).
            gt_masks (torch.Tensor): Ground truth masks [N, H, W].
            pred_masks (torch.Tensor | None): Prediction masks [M, H, W] or None.
            original_shape (tuple[int, int]): (h, w) of original image.
        """
        self.seen += 1
        npr = len(aggregated_preds)
        nl = len(gt_cls)
        
        # Validate prediction tensor shape
        # Expected format after NMS: [x1, y1, x2, y2, conf, cls, mask_coeffs...]
        # For single task, minimum columns = 6 (4 box + 1 conf + 1 cls)
        min_cols_required = 4 + 2 * self.num_tasks  # xyxy + (conf, cls) per task
        
        if npr > 0 and aggregated_preds.shape[1] < min_cols_required:
            LOGGER.warning(f"SAHI: aggregated_preds has {aggregated_preds.shape[1]} columns, "
                         f"expected at least {min_cols_required}. Skipping this image for metrics.")
            # Still count as seen but don't update stats
            return
        
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
            conf_idx = 4 + 2 * t
            cls_idx = 5 + 2 * t
            
            # Safety check for column indices
            if cls_idx >= aggregated_preds.shape[1]:
                LOGGER.warning(f"SAHI: Cannot access column {cls_idx} in tensor with {aggregated_preds.shape[1]} columns")
                continue
            
            stat[t]["conf"] = aggregated_preds[..., conf_idx]
            stat[t]["pred_cls"] = aggregated_preds[..., cls_idx]
            
            if nl > 0:
                pm = pred_masks if pred_masks is not None and len(pred_masks) else torch.zeros((npr, *gt_masks.shape[1:]), device=self.device, dtype=torch.uint8)
                preds_dict = {
                    "bboxes": aggregated_preds[:, :4],
                    f"cls_{t}": aggregated_preds[:, 5 + 2 * t],
                    "masks": pm,
                }
                batch_dict = {
                    "bboxes": gt_bboxes_xyxy,
                    "cls": gt_cls_task.unsqueeze(1) if gt_cls_task.dim() == 1 else gt_cls[:, t : t + 1],
                    "masks": gt_masks,
                }
                proc = self._process_batch(preds_dict, batch_dict, task=0)
                stat[t]["tp"] = proc["tp"]
                stat[t]["tp_m"] = proc["tp_m"]
                
                if self.args.plots:
                    det = aggregated_preds[..., [0, 1, 2, 3, conf_idx, cls_idx]]
                    self.confusion_matrices[t].process_batch(det, gt_bboxes_xyxy, gt_cls_task)
            
            for k in self.stats[t].keys():
                self.stats[t][k].append(stat[t][k])

    def _build_gt_masks(
        self, segments: list[Any], original_shape: tuple[int, int]
    ) -> torch.Tensor:
        """Build ground truth masks from polygon segments.

        Args:
            segments (list[Any]): List of polygon segments (normalized coordinates).
            original_shape (tuple[int, int]): (h, w) of image.

        Returns:
            (torch.Tensor): Boolean masks of shape [N, H, W].
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

    def _generate_pred_masks_for_sahi(
        self,
        aggregated_preds: torch.Tensor,
        nms_indices: torch.Tensor,
        img_key: str,
        original_shape: tuple[int, int],
        mask_start_col: int,
    ) -> torch.Tensor | None:
        """Generate prediction masks for SAHI-aggregated detections.

        Uses stored proto features and model-space boxes to generate masks via
        process_mask, then transforms them to original image coordinates via
        _transform_mask_to_original. Groups detections by source crop for
        efficient batch processing.

        Args:
            aggregated_preds (torch.Tensor): Post-NMS predictions [M, cols].
            nms_indices (torch.Tensor): Indices mapping NMS results to original predictions.
            img_key (str): Image key in aggregator.
            original_shape (tuple[int, int]): (h, w) of original image.
            mask_start_col (int): Column index where mask coefficients start.

        Returns:
            (torch.Tensor | None): Boolean masks [N, H, W] or None if empty.
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
                
                # Clear intermediate tensors to free memory
                del mask_lb, pred_mask
            
            # Clear batch tensors after processing each crop
            del masks_lb, batch_boxes, batch_coeffs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return pred_masks

    def _transform_mask_to_original(
        self,
        mask_lb: torch.Tensor,
        box_model: torch.Tensor,
        box_orig: torch.Tensor,
        meta: dict[str, Any],
        original_shape: tuple[int, int],
    ) -> torch.Tensor | None:
        """Transform a mask from letterbox/model space to original image coordinates.

        Args:
            mask_lb (torch.Tensor): Mask in letterbox space [H, W].
            box_model (torch.Tensor): Box in model space (xyxy).
            box_orig (torch.Tensor): Box in original image space (xyxy).
            meta (dict[str, Any]): Crop metadata dict with 'imgsz', etc.
            original_shape (tuple[int, int]): (h, w) of original image.

        Returns:
            (torch.Tensor | None): Boolean mask in original image coordinates, or None.
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

    def get_stats(self) -> dict[str, Any]:
        """Compute and return metrics statistics.

        Concatenates self.stats (numpy or tensor), computes nt_per_class/nt_per_image,
        calls metrics.process for each task, returns combined results with fitness.

        Returns:
            (dict[str, Any]): Dictionary with task{i}_* keys and 'fitness'.
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
                stats = {}
                for k, v in st.items():
                    if not v:
                        stats[k] = np.zeros(0)
                    elif hasattr(v[0], "cpu"):
                        stats[k] = torch.cat(v, 0).cpu().numpy()
                    else:
                        stats[k] = np.concatenate(v, 0)
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

    def finalize_metrics(self, *args: Any, **kwargs: Any) -> None:
        """Finalize metrics after validation.

        Processes any remaining incomplete SAHI images in the aggregator, then
        calls parent finalize_metrics.
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

    def plot_val_samples(self, batch: dict[str, Any], ni: int) -> None:
        """Plot validation samples with masks and bounding boxes.

        Args:
            batch (dict[str, Any]): Batch containing images and annotations.
            ni (int): Batch index.
        """
        if self.sahi_enabled and getattr(self, "plotter", None) is not None:
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

    def plot_predictions(
        self, batch: dict[str, Any], preds: list[dict[str, torch.Tensor]], ni: int
    ) -> None:
        """Plot batch predictions with masks and bounding boxes.

        Args:
            batch (dict[str, Any]): Batch containing images and annotations.
            preds (list[dict[str, torch.Tensor]]): List of predictions from the model.
            ni (int): Batch index.
        """
        if self.sahi_enabled and getattr(self, "plotter", None) is not None:
            self.plotter.plot_predictions(batch, preds, ni)
            return
        if not preds:
            return
        max_det = self.args.max_det
        batch_idx = torch.cat([torch.full_like(p["conf_0"], i) for i, p in enumerate(preds)], 0)
        cls = torch.cat([p["cls_0"][:max_det] for p in preds], 0)
        bboxes = ops.xyxy2xywh(torch.cat([p["bboxes"][:max_det] for p in preds], 0))
        confs = torch.cat([p["conf_0"][:max_det] for p in preds], 0)
        masks_list = [p.get("masks", torch.zeros(0, dtype=torch.uint8))[:max_det] for p in preds]
        masks = torch.cat(masks_list, 0) if any(m.numel() for m in masks_list) else torch.zeros(0, dtype=torch.uint8)
        plot_images(
            batch["img"],
            batch_idx,
            cls,
            bboxes,
            confs=confs,
            masks=masks,
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names[0] if isinstance(self.names[0], dict) else self.names,
            on_plot=self.on_plot,
        )

    def scale_preds(
        self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        """Scale predictions and masks to original image size.

        Args:
            predn (dict[str, torch.Tensor]): Predictions with 'bboxes', 'masks', etc.
            pbatch (dict[str, Any]): Batch with 'ori_shape', 'ratio_pad'.

        Returns:
            (dict[str, torch.Tensor]): Scaled predictions including 'masks'.
        """
        out = super().scale_preds(predn, pbatch)
        if "masks" in predn:
            out["masks"] = ops.scale_masks(predn["masks"][None], pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"])[0].byte()
        return out

    def save_one_txt(
        self,
        predn: dict[str, torch.Tensor],
        save_conf: bool,
        shape: tuple[int, int],
        file: Path,
    ) -> None:
        """Save YOLO detections to a txt file in normalized coordinates.

        Args:
            predn (dict[str, torch.Tensor]): Prediction dictionary containing 'bboxes',
                'conf_0', 'cls_0', and 'masks' keys.
            save_conf (bool): Whether to save confidence scores.
            shape (tuple[int, int]): Shape of the original image (height, width).
            file (Path): File path to save the detections.
        """
        from ultralytics.engine.results import Results

        names = self.names[0] if isinstance(self.names, list) else self.names
        boxes = torch.cat([predn["bboxes"], predn["conf_0"].unsqueeze(-1), predn["cls_0"].unsqueeze(-1)], dim=1)
        masks = torch.as_tensor(predn["masks"], dtype=torch.uint8) if "masks" in predn else torch.zeros((0, *shape), dtype=torch.uint8)
        Results(np.zeros((shape[0], shape[1]), dtype=np.uint8), path=None, names=names, boxes=boxes, masks=masks).save_txt(file, save_conf=save_conf)

    def pred_to_json(
        self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]
    ) -> None:
        """Save one JSON result for COCO evaluation with RLE segmentation masks.

        Args:
            predn (dict[str, torch.Tensor]): Predictions containing bboxes, masks,
                conf_0, cls_0.
            pbatch (dict[str, Any]): Batch dictionary containing 'imgsz', 'ori_shape',
                'ratio_pad', and 'im_file'.
        """
        def to_string(counts: list[int]) -> str:
            """Convert RLE counts to compact string (delta + variable-length encoding)."""
            result = []
            for i in range(len(counts)):
                x = int(counts[i])
                if i > 2:
                    x -= int(counts[i - 2])
                while True:
                    c = x & 0x1F
                    x >>= 5
                    more = (x != -1) if (c & 0x10) else (x != 0)
                    if more:
                        c |= 0x20
                    c += 48
                    result.append(chr(c))
                    if not more:
                        break
            return "".join(result)

        def multi_encode(pixels: torch.Tensor) -> list[list[int]]:
            """Convert binary masks to RLE counts per row."""
            transitions = pixels[:, 1:] != pixels[:, :-1]
            row_idx, col_idx = torch.where(transitions)
            col_idx = col_idx + 1
            counts = []
            for i in range(pixels.shape[0]):
                positions = col_idx[row_idx == i]
                if len(positions):
                    count = torch.diff(positions).tolist()
                    count.insert(0, positions[0].item())
                    count.append(len(pixels[i]) - positions[-1].item())
                else:
                    count = [len(pixels[i])]
                if pixels[i][0].item() == 1:
                    count = [0, *count]
                counts.append(count)
            return counts

        super().pred_to_json(predn, pbatch)
        if "masks" in predn and len(predn["masks"]):
            pred_masks = predn["masks"].transpose(2, 1).contiguous().view(len(predn["masks"]), -1)
            h, w = predn["masks"].shape[1:3]
            counts = multi_encode(pred_masks)
            rles = [{"size": [h, w], "counts": to_string(c)} for c in counts]
            for i, r in enumerate(rles):
                self.jdict[-len(rles) + i]["segmentation"] = r
