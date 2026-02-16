# Ultralytics YOLO 🚀, AGPL-3.0 license

"""
YOLO Detection Validator.

This module provides the DetectionValidator class for validating detection models.
Supports both standard validation and SAHI (Slicing Aided Hyper Inference) mode
for high-resolution images.

Features:
    - Multi-task detection support
    - SAHI integration for large images
    - COCO/LVIS evaluation support
    - Confusion matrix and metrics plotting
    - JSON and TXT output formats
"""

import os
from pathlib import Path

import torch
import numpy as np

from ultralytics.data import build_dataloader, build_yolo_dataset, converter
from ultralytics.engine.validator import BaseValidator
from ultralytics.models.yolo.detect.sahi_val import SAHICropAggregator
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics, box_iou
from ultralytics.utils.plotting import ValidatorPlotter
from ultralytics.data.sahi_dataset import SAHIDataset


class DetectionValidator(BaseValidator):
    """
    Detection validator extending BaseValidator.
    
    Supports:
        - Standard single/multi-task detection
        - SAHI mode for high-resolution images
        - COCO/LVIS dataset evaluation
        - Various output formats (JSON, TXT)
    
    Attributes:
        is_coco: Whether validating on COCO dataset
        is_lvis: Whether validating on LVIS dataset
        class_map: Class ID mapping for COCO format
        metrics: List of DetMetrics for each task
        confusion_matrices: List of ConfusionMatrix for each task
        sahi_enabled: Whether SAHI mode is active
        sahi_aggregator: SAHICropAggregator instance (if SAHI enabled)
    
    Example:
        ```python
        from ultralytics.models.yolo.detect import DetectionValidator
        
        args = dict(model="yolov8n.pt", data="coco8.yaml")
        validator = DetectionValidator(args=args)
        validator()
        ```
    """

    # ==================== Initialization ====================

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """
        Initialize detection validator.
        
        Args:
            dataloader: Validation dataloader
            save_dir: Directory to save results
            pbar: Progress bar
            args: Validation arguments
            _callbacks: Callback functions
        """
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        
        # Dataset flags
        self.is_coco = False
        self.is_lvis = False
        self.class_map = None
        
        # Task configuration
        self.args.task = "detect"
        self.metrics: list[DetMetrics] = [DetMetrics(save_dir=self.save_dir, on_plot=self.on_plot)]
        self.iouv = torch.linspace(0.5, 0.95, 10)  # IoU vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()
        self.lb = []  # for autolabelling
        
        # Hybrid mode warning
        if self.args.save_hybrid:
            LOGGER.warning(
                "WARNING ⚠️ 'save_hybrid=True' will append ground truth to predictions for autolabelling.\n"
                "WARNING ⚠️ 'save_hybrid=True' will cause incorrect mAP.\n"
            )
        
        # SAHI configuration
        self.sahi_aggregator = None
        self.sahi_enabled = False
        
        if self.dataloader is not None and hasattr(self.dataloader, 'dataset'):
            if isinstance(self.dataloader.dataset, SAHIDataset):
                self.sahi_enabled = True
                from ultralytics.models.yolo.detect.sahi_val import SAHICropAggregator
                self.sahi_aggregator = SAHICropAggregator(self)
                self.sahi_aggregator.calculate_expected_crops(self.dataloader.dataset)

    def init_metrics(self, model):
        """
        Initialize evaluation metrics for YOLO.
        
        Sets up:
            - Class names and counts (nc)
            - Metrics instances for each task
            - Confusion matrices
            - Statistics containers
            - Plotter for visualization
        
        Args:
            model: YOLO model being validated
        """
        # Store model reference for SAHI raw prediction extraction
        self.model = model
        
        val = self.data.get(self.args.split, "")
        
        # Detect COCO/LVIS datasets
        self.is_coco = (
            isinstance(val, str)
            and "coco" in val
            and (val.endswith(f"{os.sep}val2017.txt") or val.endswith(f"{os.sep}test-dev2017.txt"))
        )
        self.is_lvis = isinstance(val, str) and "lvis" in val and not self.is_coco
        
        # Parse class names
        data_names = self.data.get('names', [])
        
        # Store model NC before single_cls override (needed for mask coefficients)
        if data_names:
            if isinstance(data_names[0], list):
                self._model_nc = [len(task_names) for task_names in data_names]
            elif isinstance(data_names[0], dict):
                self._model_nc = [len(task_dict) for task_dict in data_names]
            else:
                self._model_nc = [len(data_names)]
        else:
            nc_val = self.data.get('nc', [1])
            self._model_nc = [nc_val] if isinstance(nc_val, int) else nc_val
        
        # Configure names and nc (possibly overridden by single_cls)
        if self.args.single_cls:
            self.names = [{0: 'object'}]
            self.nc = [1]
        elif data_names:
            if isinstance(data_names[0], list):
                # Multi-task: names: [['heavy', 'light'], ['dmg', 'undmg']]
                self.names = [{i: name for i, name in enumerate(task_names)} for task_names in data_names]
                self.nc = [len(task_names) for task_names in data_names]
            elif isinstance(data_names[0], dict):
                # Multi-task: names already in dict format
                self.names = data_names
                self.nc = [len(task_dict) for task_dict in data_names]
            else:
                # Single task: names: ['class1', 'class2', ...]
                self.names = [{i: name for i, name in enumerate(data_names)}]
                self.nc = [len(data_names)]
        else:
            self.names = [{i: f'class{i}' for i in range(nc_i)} for nc_i in self.data.get('nc', [1])]
            self.nc = self.data.get('nc', [1])
        
        # Class mapping for COCO evaluation
        self.class_map = converter.coco80_to_coco91_class() if self.is_coco else list(range(self.nc[0]))
        self.args.save_json |= (self.is_coco or self.is_lvis) and not self.training
        
        # Initialize per-task metrics
        self.num_tasks = len(self.nc)
        self.metrics = [
            DetMetrics(save_dir=self.save_dir, on_plot=self.on_plot, names=names)
            for names in self.names
        ]
        self.confusion_matrices = [ConfusionMatrix(nc=nc_i, conf=self.args.conf) for nc_i in self.nc]
        self.stats = [dict(tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[]) for _ in self.nc]
        
        self.seen = 0
        self.jdict = []
        
        # Initialize plotter
        self.plotter = ValidatorPlotter(
            save_dir=self.save_dir,
            names=self.names,
            nc=self.nc,
            on_plot=self.on_plot,
            sahi_enabled=self.sahi_enabled,
            dataloader=self.dataloader,
            max_det=self.args.max_det
        )

    def preprocess(self, batch):
        """
        Preprocess batch of images for YOLO validation.
        
        Handles:
            - Moving images to device
            - Converting to half/float precision
            - Normalizing to [0, 1] (skipped for models with built-in normalization)
            - Setting up hybrid labels if enabled
        
        Args:
            batch: Batch dict from dataloader
            
        Returns:
            Preprocessed batch dict
        """
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        
        # Get model format flags with safe defaults
        is_rknn = getattr(self, 'rknn', False)
        is_hef = getattr(self, 'hef', False)
        is_int8 = getattr(self, 'int8', False)
        
        batch["img"] = batch["img"].to(torch.uint8 if is_int8 else torch.float16 if self.args.half else torch.float32)
        
        # RKNN and Hailo have their own normalization
        if not is_rknn and not is_hef:
            # Normalize images based on bit depth from config
            bit_depth = getattr(self.args, 'image_bit_depth', 8)
            batch["img"] /= 255.0 if bit_depth == 8 else 65_535.0

        # Store image dimensions (H, W) - batch is always BCHW from dataloader
        self._img_hw = (int(batch["img"].shape[2]), int(batch["img"].shape[3]))
        
        for k in ["batch_idx", "cls", "bboxes"]:
            batch[k] = batch[k].to(self.device)

        if self.args.save_hybrid:
            height, width = self._img_hw  # Use stored NHWC-aware dimensions
            nb = len(batch["img"])
            bboxes = batch["bboxes"] * torch.tensor((width, height, width, height), device=self.device)
            self.lb = [
                torch.cat([batch["cls"][batch["batch_idx"] == i], bboxes[batch["batch_idx"] == i]], dim=-1)
                for i in range(nb)
            ]

        return batch

    def postprocess(self, preds):
        """
        Apply Non-maximum suppression to predictions.
        
        For SAHI mode, also stores raw predictions for aggregation.
        For end2end models (YOLO26/v10), extracts raw predictions before postprocess.
        
        Args:
            preds: Model predictions
            
        Returns:
            List of post-NMS predictions per image
        """
        # Handle RKNN models - process raw DFL outputs first
        if getattr(self, 'rknn', False):
            img_hw = getattr(self, '_img_hw', (self.args.imgsz, self.args.imgsz))
            end2end = getattr(self, 'end2end', False)
            if not end2end:
                preds = ops.process_rknn_dfl_results(
                    input_data=preds,
                    imgsz=img_hw,
                    conf_thres=self.args.conf,
                )
            else:
                preds = ops.process_rknn_end2end_results(
                    input_data=preds,
                    imgsz=img_hw,
                    conf_thres=self.args.conf,
                    nc=self.nc,
                )
        
        # Handle Hailo models - process raw outputs first
        elif getattr(self, 'hef', False):
            img_hw = getattr(self, '_img_hw', (self.args.imgsz, self.args.imgsz))
            # Check if NMS is in the graph (from metadata)
            nms = getattr(self, 'nms', False)
            if nms:
                # NMS already applied in model graph
                preds = ops.process_nms_hef_results(preds, img_hw=img_hw)
            else:
                # No NMS in graph - process DFL/end2end outputs
                end2end = getattr(self, 'end2end', False)
                if not end2end:
                    preds = ops.process_hef_dfl_results(
                        input_data=preds,
                        imgsz=img_hw,
                        conf_thres=self.args.conf,
                    )
                else:
                    preds = ops.process_hef_end2end_results(
                        input_data=preds,
                        imgsz=img_hw,
                        conf_thres=self.args.conf,
                        nc=self.nc,
                    )
        
        if isinstance(preds, (list, tuple)):
            actual_preds = preds[0]
            raw_dict = preds[1] if len(preds) > 1 else None
        else:
            actual_preds = preds
            raw_dict = None
        
        if not isinstance(actual_preds, torch.Tensor):
            LOGGER.error(f"Error in postprocess: 'actual_preds' is not a tensor, but {type(actual_preds)}")
            return []

        # Store raw predictions for SAHI aggregation
        if self.sahi_enabled:
            # For end2end models, we need to reconstruct raw predictions
            # because actual_preds is already postprocessed to [batch, max_det, 6]
            if raw_dict is not None and isinstance(raw_dict, dict) and 'one2one' in raw_dict:
                raw_preds = self._get_raw_preds_from_end2end(raw_dict['one2one'])
                if raw_preds is not None:
                    self._last_raw_preds = raw_preds
                else:
                    # Fallback to actual_preds if extraction failed
                    self._last_raw_preds = actual_preds.clone()
            else:
                self._last_raw_preds = actual_preds.clone()
            
        return ops.non_max_suppression(
            actual_preds,
            self.args.conf,
            self.args.iou,
            labels=self.lb,
            agnostic=self.args.single_cls or self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=[1] if self.args.single_cls else self.nc,
        )
    
    def _get_raw_preds_from_end2end(self, one2one_feats):
        """
        Reconstruct raw predictions from end2end one2one features.
        
        For SAHI, we need predictions in format [batch, 4+nc, anchors] before
        the end2end postprocess() converts them to [batch, max_det, 6].
        
        Args:
            one2one_feats: List of feature maps from one2one head [BCHW, ...]
        
        Returns:
            Raw predictions tensor [batch, 4+nc, anchors]
        """
        from ultralytics.utils.tal import make_anchors
        
        if not isinstance(one2one_feats, list) or len(one2one_feats) == 0:
            return None
        
        # Get model's detection head for decoding
        detect_head = None
        # self.model is already unwrapped (DetectionModel), so iterate directly
        model_to_search = self.model.model if hasattr(self.model, 'model') else self.model
        for m in model_to_search.modules():
            if hasattr(m, 'decode_bboxes') and hasattr(m, 'dfl'):
                detect_head = m
                break
        
        if detect_head is None:
            LOGGER.warning("Could not find detection head for end2end raw prediction extraction")
            return None
        
        try:
            # Concatenate features from all scales
            shape = one2one_feats[0].shape  # BCHW
            no = detect_head.no if hasattr(detect_head, 'no') else one2one_feats[0].shape[1]
            x_cat = torch.cat([xi.view(shape[0], no, -1) for xi in one2one_feats], 2)
            
            # Decode boxes (same as _inference() but returns full raw format)
            reg_max = detect_head.reg_max if hasattr(detect_head, 'reg_max') else 16
            box = x_cat[:, : reg_max * 4]
            cls = x_cat[:, reg_max * 4 :]
            
            # Make anchors if needed
            if not hasattr(detect_head, 'anchors') or detect_head.anchors.numel() == 0:
                detect_head.anchors, detect_head.strides = (
                    x.transpose(0, 1) for x in make_anchors(one2one_feats, detect_head.stride, 0.5)
                )
            
            dbox = detect_head.decode_bboxes(
                detect_head.dfl(box), 
                detect_head.anchors.unsqueeze(0)
            ) * detect_head.strides
            
            # Return concatenated [dbox, cls.sigmoid()] - same format as _inference()
            # Shape: [batch, 4+nc, anchors]
            return torch.cat((dbox, cls.sigmoid()), 1)
            
        except Exception as e:
            LOGGER.warning(f"Error extracting raw predictions for SAHI: {e}")
            return None

    def _prepare_batch(self, si, batch):
        """
        Prepare a single sample from batch for validation.
        
        Extracts and transforms:
            - Class labels
            - Bounding boxes (to xyxy format in original image space)
            - Image shape information
        
        Args:
            si: Sample index within batch
            batch: Full batch dict
            
        Returns:
            Prepared batch dict with cls, bbox, ori_shape, imgsz, ratio_pad
        """
        from ultralytics.models.yolo.detect.sahi_val import _fix_ratio_pad, _to_tuple
        
        # Get indices for this sample
        if 'batch_idx' in batch and batch['batch_idx'].numel() > 0:
            idx = batch["batch_idx"] == si
        else:
            idx = torch.ones(len(batch["cls"]), dtype=torch.bool, device=batch["cls"].device) if si == 0 else torch.zeros(len(batch["cls"]), dtype=torch.bool, device=batch["cls"].device)
        
        cls = batch["cls"][idx] if idx.any() else batch["cls"]
        bbox = batch["bboxes"][idx] if idx.any() else batch["bboxes"]
        
        # Get original shape
        if isinstance(batch["ori_shape"], list):
            ori_shape = batch["ori_shape"][si] if si < len(batch["ori_shape"]) else batch["ori_shape"][0]
        else:
            ori_shape = batch["ori_shape"][si] if len(batch["ori_shape"]) > si else batch["ori_shape"][0]
        
        # Get image size - batch is always BCHW from dataloader
        imgsz = batch["img"].shape[2:]  # NCHW: (batch, C, H, W)
        
        # Get ratio_pad
        if isinstance(batch["ratio_pad"], list):
            ratio_pad = batch["ratio_pad"][si] if si < len(batch["ratio_pad"]) else batch["ratio_pad"][0]
        else:
            ratio_pad = batch["ratio_pad"][si] if len(batch["ratio_pad"]) > si else batch["ratio_pad"][0]
        
        # Fix ratio_pad format
        ratio_pad = _fix_ratio_pad(_to_tuple(ratio_pad))
        if ratio_pad is None:
            ratio_pad = ((1.0, 1.0), (0, 0))
        elif isinstance(ratio_pad, tuple) and len(ratio_pad) == 2:
            if not isinstance(ratio_pad[0], tuple):
                ratio_pad = (ratio_pad, (0, 0))
        
        # Transform boxes to original image space
        if len(cls):
            bbox = ops.xywh2xyxy(bbox) * torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]]
            ops.scale_boxes(imgsz, bbox, ori_shape, ratio_pad=ratio_pad)
        
        return {"cls": cls, "bbox": bbox, "ori_shape": ori_shape, "imgsz": imgsz, "ratio_pad": ratio_pad}

    def _prepare_pred(self, pred, pbatch):
        """
        Prepare predictions for metric computation.
        
        Scales boxes from model input space to original image space.
        
        Args:
            pred: Predictions tensor
            pbatch: Prepared batch dict
            
        Returns:
            Scaled predictions
        """
        predn = pred.clone()
        ops.scale_boxes(
            pbatch["imgsz"], predn[:, :4], pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"]
        )
        return predn

    # ==================== Metrics Update ====================

    def update_metrics(self, preds, batch):
        """
        Update metrics with predictions from current batch.
        
        Routes to SAHI or standard metrics update based on mode.
        
        Args:
            preds: Post-NMS predictions
            batch: Current batch dict
        """
        if self.sahi_enabled and self.sahi_aggregator is not None:
            self._update_metrics_sahi(preds, batch)
        else:
            self._update_metrics_standard(preds, batch)

    def _update_metrics_standard(self, preds, batch):
        """
        Standard metrics update (non-SAHI).
        
        For each prediction in batch:
            1. Prepare ground truth
            2. Match predictions to ground truth
            3. Compute true positives
            4. Update confusion matrix
            5. Save outputs if requested
        """
        for si, pred in enumerate(preds):
            self.seen += 1
            npr = len(pred)
            
            # Initialize statistics for this sample
            stat = [
                dict(
                    conf=torch.zeros(0, device=self.device),
                    pred_cls=torch.zeros(0, device=self.device),
                    tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
                )
                for _ in range(self.num_tasks)
            ]

            pbatch = self._prepare_batch(si, batch)
            cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")
            nl = len(cls)

            # Store target info
            for t in range(self.num_tasks):
                gt_cls = cls[:, t]
                stat[t]["target_cls"] = gt_cls
                stat[t]["target_img"] = gt_cls.unique()
            
            # Handle empty predictions
            if npr == 0:
                if nl:
                    for t in range(self.num_tasks):
                        for k in self.stats[t].keys():
                            self.stats[t][k].append(stat[t][k])
                        if self.args.plots:
                            self.confusion_matrices[t].process_batch(
                                detections=None, gt_bboxes=bbox, gt_cls=cls[:, t]
                            )
                continue

            # Process predictions
            if self.args.single_cls:
                pred[:, 5] = 0
            predn = self._prepare_pred(pred, pbatch)
            
            for t in range(self.num_tasks):
                # Extract conf/cls for task t (format: [x1,y1,x2,y2, conf0,cls0, conf1,cls1, ...])
                stat[t]["conf"] = predn[..., 4 + 2 * t]
                stat[t]["pred_cls"] = predn[..., 5 + 2 * t]
                
                if nl:
                    stat[t]["tp"] = self._process_batch(predn, bbox, cls[:, t], task=t)
                    if self.args.plots:
                        det = predn[..., [0, 1, 2, 3, 4 + 2 * t, 5 + 2 * t]]
                        self.confusion_matrices[t].process_batch(det, bbox, cls[:, t])
                
                for k in self.stats[t].keys():
                    self.stats[t][k].append(stat[t][k])

            # Save outputs
            if self.args.save_json:
                self.pred_to_json(predn, batch["im_file"][si])
            if self.args.save_txt:
                self.save_one_txt(
                    predn,
                    self.args.save_conf,
                    pbatch["ori_shape"],
                    self.save_dir / "labels" / f"{Path(batch['im_file'][si]).stem}.txt",
                )

    def _update_metrics_sahi(self, preds, batch):
        """
        SAHI mode metrics update.
        
        Collects predictions from crops and processes complete images.
        """
        if not hasattr(self, '_sahi_batch_count'):
            self._sahi_batch_count = 0
        self._sahi_batch_count += 1
        
        # Get raw predictions
        if hasattr(self, '_last_raw_preds'):
            raw_preds = self._last_raw_preds
        else:
            LOGGER.warning("Raw predictions not available for SAHI aggregation, skipping batch")
            return
        
        # Add predictions to aggregator
        success = self.sahi_aggregator.add_crop_predictions(batch, raw_preds, preds)
        self._last_raw_preds = None  # Clear reference
        
        if not success:
            LOGGER.warning("Cannot add_crop_predictions for SAHI, skipping batch")
            return
        
        # Process completed images
        completed_images = self.sahi_aggregator.get_completed_images()
        
        for img_key in completed_images:
            try:
                self._process_complete_image(img_key)
            except Exception as e:
                LOGGER.error(f"Error processing complete image {img_key}: {e}")
                import traceback
                traceback.print_exc()
            finally:
                self.sahi_aggregator.cleanup_image(img_key)

    def _process_complete_image(self, img_key):
        """
        Process a complete SAHI image with all crops aggregated.
        
        Steps:
            1. Get aggregated predictions
            2. Apply NMS to merged predictions
            3. Get ground truth from original labels
            4. Update metrics
        """
        # Get aggregated predictions
        aggregated_preds_raw = self.sahi_aggregator.get_aggregated_predictions(img_key)
        
        # nc for NMS should be a list (the fork's NMS uses sum(nc))
        nc_for_nms = [1] if self.args.single_cls else self.nc
        
        min_cols = 4 + 2 * len(self.nc)  # xyxy + (conf, cls) per task
        
        if len(aggregated_preds_raw) == 0:
            aggregated_preds = torch.empty((0, min_cols), device=self.device)
        else:
            # Apply NMS
            preds_for_nms = aggregated_preds_raw.unsqueeze(0).permute(0, 2, 1)
            
            nms_results = ops.non_max_suppression(
                preds_for_nms,
                self.args.conf,
                self.args.iou,
                labels=[],
                agnostic=self.args.single_cls or self.args.agnostic_nms,
                max_det=self.args.max_det,
                nc=nc_for_nms,  # Pass as list (fork's NMS uses sum(nc))
            )
            aggregated_preds = nms_results[0] if nms_results else torch.empty((0, min_cols), device=self.device)
        
        # Get ground truth
        img_idx = self.sahi_aggregator.image_crops[img_key]['original_img_idx']
        original_shape = self.sahi_aggregator.image_crops[img_key]['original_shape']
        original_labels = self.dataloader.dataset.labels[img_idx]
        
        if 'cls' in original_labels and len(original_labels['cls']) > 0:
            gt_cls = torch.tensor(original_labels['cls'], device=self.device, dtype=torch.float32)
            gt_bboxes = torch.tensor(original_labels['bboxes'], device=self.device, dtype=torch.float32)
        else:
            num_tasks = len(self.nc)
            gt_cls = torch.empty((0, num_tasks), device=self.device, dtype=torch.float32)
            gt_bboxes = torch.empty((0, 4), device=self.device, dtype=torch.float32)
        
        # Ensure correct shape
        if gt_cls.dim() == 1:
            gt_cls = gt_cls.unsqueeze(1)
        
        # Store for plotting
        if self.args.plots and len(self.plotter._sahi_plot_cache) < 16:
            if img_idx not in self.plotter._sahi_plot_cache:
                self.plotter._sahi_plot_cache[img_idx] = {
                    'im_file': self.dataloader.dataset.im_files[img_idx],
                    'original_shape': original_shape,
                    'predictions': aggregated_preds.clone().cpu(),
                    'gt_cls': gt_cls.clone().cpu(),
                    'gt_bboxes': gt_bboxes.clone().cpu()
                }
        
        # Create synthetic batch for metrics
        synthetic_batch = {
            'cls': gt_cls,
            'bboxes': gt_bboxes,
            'batch_idx': torch.zeros(len(gt_bboxes), device=self.device, dtype=torch.long),
            'ori_shape': [original_shape],
            'img': torch.zeros((1, 3, original_shape[0], original_shape[1]), device=self.device),
            'im_file': [self.dataloader.dataset.im_files[img_idx]],
            'resized_shape': [original_shape],
            'ratio_pad': [((1.0, 1.0), (0, 0))],
        }
        
        self._update_metrics_for_sahi_image(aggregated_preds, synthetic_batch)

    def _update_metrics_for_sahi_image(self, preds, batch):
        """
        Update metrics for a single SAHI-aggregated image.
        
        Predictions are already in xyxy format in original image coords.
        """
        self.seen += 1
        npr = len(preds)
        
        # Validate prediction tensor shape
        min_cols_required = 4 + 2 * self.num_tasks  # xyxy + (conf, cls) per task
        
        if npr > 0 and preds.shape[1] < min_cols_required:
            LOGGER.warning(f"SAHI: preds has {preds.shape[1]} columns, "
                         f"expected at least {min_cols_required}. Skipping this image for metrics.")
            return
        
        stat = [
            dict(
                conf=torch.zeros(0, device=self.device),
                pred_cls=torch.zeros(0, device=self.device),
                tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
            )
            for _ in range(self.num_tasks)
        ]
        
        cls = batch["cls"]
        bboxes = batch["bboxes"]
        ori_shape = batch["ori_shape"][0]
        nl = len(cls)
        
        # Convert GT to pixel xyxy
        if nl > 0:
            h, w = ori_shape
            gt_bboxes_xyxy = ops.xywh2xyxy(bboxes) * torch.tensor([w, h, w, h], device=self.device)
        else:
            gt_bboxes_xyxy = torch.empty((0, 4), device=self.device)
        
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
                            detections=None, gt_bboxes=gt_bboxes_xyxy,
                            gt_cls=cls[:, t] if cls.dim() > 1 else cls,
                        )
            return
        
        if self.args.single_cls:
            preds[:, 5] = 0
        
        for t in range(self.num_tasks):
            gt_cls = cls[:, t] if cls.dim() > 1 else cls
            
            # Safety check for column indices
            conf_idx = 4 + 2 * t
            cls_idx = 5 + 2 * t
            
            if cls_idx >= preds.shape[1]:
                LOGGER.warning(f"SAHI: Cannot access column {cls_idx} in tensor with {preds.shape[1]} columns")
                continue
            
            stat[t]["conf"] = preds[..., conf_idx]
            stat[t]["pred_cls"] = preds[..., cls_idx]
            
            if nl:
                stat[t]["tp"] = self._process_batch(preds, gt_bboxes_xyxy, gt_cls, task=t)
                if self.args.plots:
                    det = preds[..., [0, 1, 2, 3, conf_idx, cls_idx]]
                    self.confusion_matrices[t].process_batch(det, gt_bboxes_xyxy, gt_cls)
            
            for k in self.stats[t].keys():
                self.stats[t][k].append(stat[t][k])

    def _process_batch(self, detections, gt_bboxes, gt_cls, task=0):
        """
        Compute correct prediction matrix (true positives).
        
        Args:
            detections: Tensor [N, 6+] with (x1, y1, x2, y2, conf, class, ...)
            gt_bboxes: Tensor [M, 4] with ground truth boxes
            gt_cls: Tensor [M] with ground truth classes
            task: Task index for multi-task models
            
        Returns:
            Tensor [N, niou] of true positive flags at each IoU threshold
        """
        iou = box_iou(gt_bboxes, detections[:, :4])
        return self.match_predictions(detections[:, 5 + 2 * task], gt_cls, iou)

    # ==================== Results & Statistics ====================

    def get_stats(self):
        """
        Compute and return metrics statistics.
        
        Returns:
            Dict with all task metrics and overall fitness
        """
        results = {}
        self.nt_per_class, self.nt_per_image = [], []
        fitness_values = []
        
        for i, (m, st) in enumerate(zip(self.metrics, self.stats)):
            stats = {k: torch.cat(v, 0).cpu().numpy() for k, v in st.items()}
            ntc = np.bincount(stats["target_cls"].astype(int), minlength=self.nc[i])
            nti = np.bincount(stats["target_img"].astype(int), minlength=self.nc[i])
            self.nt_per_class.append(ntc)
            self.nt_per_image.append(nti)
            stats.pop("target_img", None)
            
            if len(stats) and stats["tp"].any():
                m.process(**stats)
            
            for k, v in m.results_dict.items():
                results[f"task{i}_{k}"] = v
            if f"task{i}_fitness" in results:
                fitness_values.append(results[f"task{i}_fitness"])
        
        if fitness_values:
            results["fitness"] = np.mean(fitness_values)

        return results

    def print_results(self):
        """Print validation metrics per class."""
        metrics_keys = self.metrics[0].keys
        pf = "%22s" + "%11i" * 2 + "%11.3g" * len(metrics_keys)
        
        for i, m in enumerate(self.metrics):
            LOGGER.info(pf % (f"task{i}", self.seen, self.nt_per_class[i].sum(), *m.mean_results()))
            if self.nt_per_class[i].sum() == 0:
                LOGGER.warning(f"WARNING ⚠️ no labels found in task{i} set, can not compute metrics without labels")

        # Per-class results
        for t in range(self.num_tasks):
            if self.args.verbose and not self.training and self.nc[t] > 1 and len(self.stats[t]):
                for i, c in enumerate(self.metrics[t].ap_class_index):
                    LOGGER.info(
                        pf % (self.names[t][c], self.nt_per_image[t][c], self.nt_per_class[t][c], 
                              *self.metrics[t].class_result(i))
                    )

            if self.args.plots:
                for normalize in True, False:
                    prefix = f"task{t}_" if self.num_tasks > 1 else ""
                    self.confusion_matrices[t].plot(
                        save_dir=self.save_dir,
                        names=list(self.names[t].values()),
                        normalize=normalize,
                        on_plot=self.on_plot,
                        prefix=prefix
                    )

    def get_desc(self):
        """Return formatted string summarizing class metrics."""
        return ("%22s" + "%11s" * 6) % ("Class", "Images", "Instances", "Box(P", "R", "mAP50", "mAP50-95)")

    # ==================== Finalization ====================

    def finalize_metrics(self, *args, **kwargs):
        """
        Finalize metrics after validation.
        
        For SAHI mode: processes any remaining incomplete images.
        """
        if self.sahi_enabled and self.sahi_aggregator is not None:
            remaining_images = list(self.sahi_aggregator.image_crops.keys())
            if remaining_images:
                LOGGER.warning(f"Processing {len(remaining_images)} incomplete images at validation end")
                for img_key in remaining_images:
                    try:
                        self._process_complete_image(img_key)
                    except Exception as e:
                        LOGGER.error(f"Error processing incomplete image {img_key}: {e}")
                    finally:
                        self.sahi_aggregator.cleanup_image(img_key)
            
            try:
                self.plotter.plot_sahi_results()
            except Exception as e:
                LOGGER.error(f"Error plotting SAHI complete images: {e}")
        
        for m, cm in zip(self.metrics, self.confusion_matrices):
            m.speed = self.speed
            m.confusion_matrix = cm

    # ==================== Dataset & Dataloader ====================

    def build_dataset(self, img_path, mode="val", batch=None):
        """
        Build YOLO dataset.
        
        Args:
            img_path: Path to images folder
            mode: 'train' or 'val' mode
            batch: Batch size for rect mode
            
        Returns:
            YOLODataset or SAHIDataset instance
        """
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, stride=self.stride)

    def get_dataloader(self, dataset_path, batch_size):
        """
        Construct and return dataloader.
        
        Args:
            dataset_path: Path to dataset
            batch_size: Batch size
            
        Returns:
            DataLoader instance
        """
        dataset = self.build_dataset(dataset_path, batch=batch_size, mode="val")
        if isinstance(dataset, SAHIDataset):
            self.sahi_enabled = True
            self.sahi_aggregator = SAHICropAggregator(self)
            self.sahi_aggregator.calculate_expected_crops(dataset)
            LOGGER.info(f"SAHI validation enabled: {len(dataset.im_files)} images, {len(dataset)} crops")
        else:
            self.sahi_enabled = False
            self.sahi_aggregator = None
        return build_dataloader(dataset, batch_size, self.args.workers, shuffle=False, rank=-1, drop_last=False)

    # ==================== Plotting ====================

    def plot_val_samples(self, batch, ni):
        """Plot validation image samples."""
        self.plotter.plot_val_samples(batch, ni)

    def plot_predictions(self, batch, preds, ni):
        """Plot predicted bounding boxes on input images."""
        self.plotter.plot_predictions(batch, preds, ni)

    # ==================== Output Saving ====================

    def save_one_txt(self, predn, save_conf, shape, file):
        """Save YOLO detections to txt file in normalized coordinates."""
        from ultralytics.engine.results import Results

        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),
            path=None,
            names=self.names,
            boxes=predn[:, :6],
        ).save_txt(file, save_conf=save_conf)

    def pred_to_json(self, predn, filename):
        """Serialize YOLO predictions to COCO JSON format."""
        stem = Path(filename).stem
        image_id = int(stem) if stem.isnumeric() else stem
        box = ops.xyxy2xywh(predn[:, :4])
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
        
        for p, b in zip(predn.tolist(), box.tolist()):
            self.jdict.append({
                "image_id": image_id,
                "category_id": self.class_map[int(p[5])] + (1 if self.is_lvis else 0),
                "bbox": [round(x, 3) for x in b],
                "score": round(p[4], 5),
            })

    def eval_json(self, stats):
        """
        Evaluate YOLO output in JSON format using COCO/LVIS API.
        
        Args:
            stats: Current statistics dict
            
        Returns:
            Updated statistics with COCO/LVIS metrics
        """
        if self.args.save_json and (self.is_coco or self.is_lvis) and len(self.jdict):
            pred_json = self.save_dir / "predictions.json"
            anno_json = (
                self.data["path"]
                / "annotations"
                / ("instances_val2017.json" if self.is_coco else f"lvis_v1_{self.args.split}.json")
            )
            pkg = "pycocotools" if self.is_coco else "lvis"
            LOGGER.info(f"\nEvaluating {pkg} mAP using {pred_json} and {anno_json}...")
            
            try:
                for x in pred_json, anno_json:
                    assert x.is_file(), f"{x} file not found"
                check_requirements("pycocotools>=2.0.6" if self.is_coco else "lvis>=0.5.3")
                
                if self.is_coco:
                    from pycocotools.coco import COCO
                    from pycocotools.cocoeval import COCOeval

                    anno = COCO(str(anno_json))
                    pred = anno.loadRes(str(pred_json))
                    val = COCOeval(anno, pred, "bbox")
                else:
                    from lvis import LVIS, LVISEval

                    anno = LVIS(str(anno_json))
                    pred = anno._load_json(str(pred_json))
                    val = LVISEval(anno, pred, "bbox")
                
                val.params.imgIds = [int(Path(x).stem) for x in self.dataloader.dataset.im_files]
                val.evaluate()
                val.accumulate()
                val.summarize()
                
                if self.is_lvis:
                    val.print_results()
                
                # Update mAP metrics
                stats[self.metrics.keys[-1]], stats[self.metrics.keys[-2]] = (
                    val.stats[:2] if self.is_coco else [val.results["AP50"], val.results["AP"]]
                )
            except Exception as e:
                LOGGER.warning(f"{pkg} unable to run: {e}")
        
        return stats
