# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

import torch.distributed as dist

from ultralytics.data import build_dataloader, build_yolo_dataset, converter
from ultralytics.data.sahi_dataset import SAHIDataset
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import LOGGER, RANK, nms, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics, box_iou
from ultralytics.utils.plotting import plot_images


class DetectionValidator(BaseValidator):
    """Detection validator with multitask, SAHI, end2end, and RKNN support.

    Attributes:
        nc (list[int]): Number of classes per task head. Always a list.
        names (list[dict]): Class names per task. List of {idx: name} dicts.
        num_tasks (int): Number of task heads.
        end2end (bool): Whether model uses end-to-end (NMS-free) detection.
        sahi_enabled (bool): Whether SAHI mode is active.
        metrics (list[DetMetrics]): Per-task metrics instances.
        confusion_matrices (list[ConfusionMatrix]): Per-task confusion matrices.
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initialize detection validator."""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.is_coco = False
        self.is_lvis = False
        self.class_map = None
        self.args.task = "detect"
        self.iouv = torch.linspace(0.5, 0.95, 10)
        self.niou = self.iouv.numel()

        # SAHI state
        self.sahi_enabled = False
        self.sahi_aggregator = None
        self.nms_strategy = getattr(self.args, "nms_strategy", "usual")

        # Detect SAHI dataset from dataloader
        if self.dataloader is not None and isinstance(self.dataloader.dataset, SAHIDataset):
            from ultralytics.models.yolo.detect.sahi_val import SAHICropAggregator
            crop_size = getattr(self.args, "crop_size", 640)
            overlap_ratio = getattr(self.args, "overlap_ratio", 0.2)
            self.sahi_enabled = True
            self.sahi_aggregator = SAHICropAggregator(self)
            self.sahi_aggregator.calculate_expected_crops(self.dataloader.dataset)
            LOGGER.info(f"SAHI inference enabled: crop_size={crop_size}, overlap_ratio={overlap_ratio}, {self.args.nms_strategy} NMS.")

    def init_metrics(self, model: torch.nn.Module) -> None:
        """Initialize evaluation metrics.

        Args:
            model (torch.nn.Module): Model to validate.
        """
        self.model = model  # store for SAHI raw pred extraction

        val = self.data.get(self.args.split, "")
        self.is_coco = (
            isinstance(val, str)
            and "coco" in val
            and (val.endswith(f"{os.sep}val2017.txt") or val.endswith(f"{os.sep}test-dev2017.txt"))
        )
        self.is_lvis = isinstance(val, str) and "lvis" in val and not self.is_coco
        self.end2end = getattr(model, "end2end", False)

        # names is already list[dict] from check_class_names (autobackend / data config)
        self.names = model.names
        self.nc = [len(d) for d in self.names]
        self.num_tasks = len(self.nc)
        self.class_map = converter.coco80_to_coco91_class() if self.is_coco else list(range(self.nc[0]))
        self.args.save_json |= (self.is_coco or self.is_lvis) and not self.training

        # Per-task metrics and confusion matrices
        self.metrics = [
            DetMetrics(save_dir=self.save_dir, on_plot=self.on_plot, names=names)
            for names in self.names
        ]
        self.confusion_matrices = [ConfusionMatrix(nc=nc_i) for nc_i in self.nc]

        self.seen = 0
        self.jdict = []

    def get_desc(self):
        """Return formatted string summarizing class metrics."""
        return ("%22s" + "%11s" * 6) % ("Class", "Images", "Instances", "Box(P", "R", "mAP50", "mAP50-95)")

    def preprocess(self, batch):
        """Preprocess batch: move to device, normalize images.

        Handles RKNN (skip normalization) and 16-bit images.
        """
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=self.device.type == "cuda")

        is_rknn = getattr(self, "rknn", False)
        is_hef = getattr(self, 'hef', False)
        is_int8 = getattr(self, "int8", False)

        batch["img"] = batch["img"].to(torch.uint8 if is_int8 else torch.float16 if self.args.half else torch.float32)
        
        # RKNN and Hailo have their own normalization
        if not is_rknn and not is_hef:
            bit_depth = getattr(self.args, "image_bit_depth", 8)
            if bit_depth == 8:
                batch["img"] /= 255.0
            elif bit_depth == 16:
                batch["img"] /= 65_535.0
            else:
                LOGGER.error(f"BitDepth {bit_depth} unsupported.")

        # Store image dimensions for RKNN
        self._img_hw = (int(batch["img"].shape[2]), int(batch["img"].shape[3]))

        return batch

    def postprocess(self, preds):
        """Apply NMS to predictions and convert to dict format.

        Handles RKNN preprocessing, end2end models, and SAHI raw pred extraction.

        Returns:
            list[dict]: Per-image predictions with keys:
                bboxes (Tensor [N, 4]): xyxy boxes
                conf_t (Tensor [N]): confidence for task t
                cls_t (Tensor [N]): class index for task t
                extra (Tensor [N, E]): mask coefficients, etc.
        """
        # Handle RKNN models
        if getattr(self, "rknn", False):
            if self.end2end:
                preds = ops.process_rknn_end2end_results(
                    input_data=preds,
                    imgsz=self._img_hw,
                    nc=self.nc,
                    strides=self.model.stride if hasattr(self.model, "stride") else (8, 16, 32),
                )
            else:
                preds = ops.process_rknn_dfl_results(
                    input_data=preds,
                    imgsz=self._img_hw,
                    conf_thres=self.args.conf,
                )
        # Handle Hailo models
        elif getattr(self, 'hef', False):
            # Check if NMS is in the graph (from metadata)
            if getattr(self, 'nms', False):
                return ops.process_nms_hef_results(preds, img_hw=self._img_hw)
            else:
                if not self.end2end:
                    preds = ops.process_hef_dfl_results(
                        input_data=preds,
                        imgsz=self._img_hw,
                        conf_thres=self.args.conf,
                    )
                else:
                    preds = ops.process_hef_end2end_results(
                        input_data=preds,
                        imgsz=self._img_hw,
                        conf_thres=self.args.conf,
                        nc=self.nc,
                    )

        # Separate inference output from raw features dict
        if isinstance(preds, (list, tuple)):
            actual_preds = preds[0]
            raw_dict = preds[1] if len(preds) > 1 else None
        else:
            actual_preds = preds
            raw_dict = None

        # Store raw predictions for SAHI aggregation
        if self.sahi_enabled:
            raw_for_sahi = actual_preds[0] if isinstance(actual_preds, (list, tuple)) and len(actual_preds) >= 1 else actual_preds
            if raw_dict is not None and isinstance(raw_dict, dict) and "one2one" in raw_dict:
                raw_preds = self._get_raw_preds_from_end2end(raw_dict["one2one"])
                if raw_preds is not None:
                    self._last_raw_preds = raw_preds
                elif hasattr(raw_for_sahi, "clone"):
                    self._last_raw_preds = raw_for_sahi.clone()
                else:
                    self._last_raw_preds = actual_preds
            elif hasattr(raw_for_sahi, "clone"):
                self._last_raw_preds = raw_for_sahi.clone()
            else:
                self._last_raw_preds = actual_preds

        # Run NMS and convert to dict format
        outputs = nms.non_max_suppression(
            actual_preds,
            self.args.conf,
            self.args.iou,
            agnostic=self.args.single_cls or self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=[1] if self.args.single_cls else self.nc,
            end2end=self.end2end,
            rotated=self.args.task == "obb",
            multi_label=True,
        )
        return [self._tensor_to_pred_dict(x) for x in outputs]

    def _tensor_to_pred_dict(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Convert NMS output tensor to dict format {bboxes, conf_0, cls_0, ...}.

        Handles end2end [N, 6] and standard multitask [N, 4+2*num_tasks+extra].
        """
        dev = x.device
        n = x.shape[0]
        out = {"bboxes": x[:, :4]}
        if x.shape[1] == 6:
            # End2end: [x1,y1,x2,y2, conf, cls]
            out["conf_0"] = x[:, 4].flatten()
            out["cls_0"] = x[:, 5].long().float().flatten()
            out["extra"] = torch.empty((n, 0), device=dev)
        else:
            # Multitask: [box, conf0, cls0, conf1, cls1, ..., extra]
            for t in range(self.num_tasks):
                out[f"conf_{t}"] = x[:, 4 + 2 * t].flatten()
                out[f"cls_{t}"] = x[:, 5 + 2 * t].long().float().flatten()
            out["extra"] = x[:, 4 + 2 * self.num_tasks :] if x.shape[1] > 4 + 2 * self.num_tasks else torch.empty((n, 0), device=dev)
        return out

    def _get_raw_preds_from_end2end(self, one2one_preds):
        """
        Reconstruct raw predictions from end2end one2one head output.
        
        For SAHI, we need predictions in format [batch, 4+nc, anchors] before
        the end2end postprocess() converts them to [batch, max_det, 6].
        
        Args:
            one2one_preds: Dict with boxes/scores/feats from one2one head.
        
        Returns:
            Raw predictions tensor [batch, 4+nc, anchors]
        """
        required_keys = {"boxes", "scores", "feats"}
        if not isinstance(one2one_preds, dict) or not required_keys.issubset(one2one_preds):
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
            raw_preds = detect_head._inference(one2one_preds)

            # SAHI aggregators expect xywh in first 4 channels.
            if getattr(detect_head, "end2end", False):
                boxes_xyxy = raw_preds[:, :4, :].permute(0, 2, 1).contiguous()
                boxes_xywh = ops.xyxy2xywh(boxes_xyxy)
                raw_preds = torch.cat((boxes_xywh.permute(0, 2, 1), raw_preds[:, 4:, :]), dim=1)
            return raw_preds
            
        except Exception as e:
            LOGGER.warning(f"Error extracting raw predictions for SAHI: {e}")
            return None

    def _prepare_batch(self, si, batch):
        """Prepare ground truth for a single image.

        Returns:
            Dict with cls (M, num_tasks), bboxes (M, 4) xyxy, ori_shape, imgsz, ratio_pad, im_file.
        """
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx]  # (M, num_tasks) or (M, 1)
        if cls.dim() == 1:
            cls = cls.unsqueeze(1)
        bbox = batch["bboxes"][idx]
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]

        if cls.shape[0]:
            bbox = ops.xywh2xyxy(bbox) * torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]]

        return {
            "cls": cls,
            "bboxes": bbox,
            "ori_shape": ori_shape,
            "imgsz": imgsz,
            "ratio_pad": ratio_pad,
            "im_file": batch["im_file"][si],
        }

    def _prepare_pred(self, pred: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Prepare predictions for evaluation against ground truth."""
        if self.args.single_cls:
            n = pred["bboxes"].shape[0]
            dev = pred["bboxes"].device
            for t in range(self.num_tasks):
                pred[f"cls_{t}"] = torch.zeros(n, device=dev, dtype=torch.float32)
        return pred

    def update_metrics(self, preds, batch):
        """Route to standard or SAHI metrics update."""
        if self.sahi_enabled and self.sahi_aggregator is not None:
            self._update_metrics_sahi(preds, batch)
        else:
            self._update_metrics_standard(preds, batch)

    def _update_metrics_standard(self, preds, batch):
        """Standard per-image metrics update."""
        for si, pred in enumerate(preds):
            self.seen += 1
            pbatch = self._prepare_batch(si, batch)
            predn = self._prepare_pred(pred)

            cls = pbatch["cls"]  # (M, num_tasks)
            nl = len(cls)
            no_pred = predn["cls_0"].shape[0] == 0

            for t in range(self.num_tasks):
                gt_cls = cls[:, t]
                stat = {
                    "target_cls": gt_cls,
                    "target_img": gt_cls.unique(),
                }

                if no_pred:
                    stat["conf"] = torch.zeros(0, device=self.device)
                    stat["pred_cls"] = torch.zeros(0, device=self.device)
                    stat["tp"] = np.zeros((0, self.niou), dtype=bool)
                else:
                    stat["conf"] = predn[f"conf_{t}"]
                    stat["pred_cls"] = predn[f"cls_{t}"]
                    if nl:
                        stat.update(self._process_batch(predn, pbatch, task=t))
                    else:
                        stat["tp"] = np.zeros((predn["cls_0"].shape[0], self.niou), dtype=bool)

                self.metrics[t].update_stats(stat)

                if self.args.plots and nl:
                    self.confusion_matrices[t].process_batch(
                        detections=torch.cat([predn["bboxes"], predn[f"conf_{t}"].unsqueeze(1), predn[f"cls_{t}"].unsqueeze(1)], 1) if not no_pred else None,
                        gt_bboxes=pbatch["bboxes"],
                        gt_cls=gt_cls,
                    )

            if no_pred:
                continue

            # Save outputs
            if self.args.save_json or self.args.save_txt:
                predn_scaled = self.scale_preds(predn, pbatch)
            if self.args.save_json:
                self.pred_to_json(predn_scaled, pbatch)
            if self.args.save_txt:
                self.save_one_txt(
                    predn_scaled,
                    self.args.save_conf,
                    pbatch["ori_shape"],
                    self.save_dir / "labels" / f"{Path(pbatch['im_file']).stem}.txt",
                )

    def _update_metrics_sahi(self, preds, batch):
        """SAHI mode: collect crop predictions and process completed images."""
        raw_preds = getattr(self, "_last_raw_preds", None)
        self._last_raw_preds = None

        if raw_preds is None:
            return

        self.sahi_aggregator.add_crop_predictions(batch, raw_preds)

        for img_key in self.sahi_aggregator.get_completed_images():
            self._process_complete_image(img_key)
            self.sahi_aggregator.cleanup_image(img_key)

    def _process_complete_image(self, img_key):
        """Process a complete SAHI image: aggregate, NMS, compute metrics."""
        aggregated = self.sahi_aggregator.get_aggregated_predictions(img_key)

        if len(aggregated) == 0:
            empty_pred = self._tensor_to_pred_dict(
                torch.empty((0, 4 + 2 * self.num_tasks), device=self.device)
            )
            aggregated_pred = empty_pred
        else:
            # NMS on aggregated predictions: [N, 4+sum(nc)] xywh format -> BCN for NMS
            preds_for_nms = aggregated.T.unsqueeze(0)  # [1, channels, N]
            nms_results = nms.non_max_suppression(
                preds_for_nms,
                self.args.conf,
                self.args.iou,
                agnostic=self.args.single_cls or self.args.agnostic_nms,
                max_det=self.args.max_det,
                nc=self.nc,
                nms_strategy=self.nms_strategy,
            )
            aggregated_pred = self._tensor_to_pred_dict(nms_results[0])

        # Get ground truth from original labels
        img_idx = self.sahi_aggregator.image_crops[img_key]["original_img_idx"]
        original_shape = self.sahi_aggregator.image_crops[img_key]["original_shape"]
        original_labels = self.dataloader.dataset.labels[img_idx]

        if "cls" in original_labels and len(original_labels["cls"]) > 0:
            gt_cls = torch.tensor(original_labels["cls"], device=self.device, dtype=torch.float32)
            gt_bboxes = torch.tensor(original_labels["bboxes"], device=self.device, dtype=torch.float32)
        else:
            gt_cls = torch.empty((0, self.num_tasks), device=self.device, dtype=torch.float32)
            gt_bboxes = torch.empty((0, 4), device=self.device, dtype=torch.float32)

        if gt_cls.dim() == 1:
            gt_cls = gt_cls.unsqueeze(1)

        # Convert GT bboxes from normalized xywh to pixel xyxy
        h, w = original_shape
        if len(gt_bboxes):
            gt_bboxes_xyxy = ops.xywh2xyxy(gt_bboxes) * torch.tensor([w, h, w, h], device=self.device)
        else:
            gt_bboxes_xyxy = torch.empty((0, 4), device=self.device)

        # Update metrics
        self.seen += 1
        npr = aggregated_pred["bboxes"].shape[0]
        nl = len(gt_cls)

        for t in range(self.num_tasks):
            gt_cls_t = gt_cls[:, t] if gt_cls.dim() > 1 else gt_cls
            stat = {
                "target_cls": gt_cls_t,
                "target_img": gt_cls_t.unique(),
            }

            if npr == 0:
                stat["conf"] = torch.zeros(0, device=self.device)
                stat["pred_cls"] = torch.zeros(0, device=self.device)
                stat["tp"] = torch.zeros(0, self.niou, dtype=torch.bool, device=self.device)
            else:
                stat["conf"] = aggregated_pred[f"conf_{t}"]
                stat["pred_cls"] = aggregated_pred[f"cls_{t}"]
                if nl:
                    gt_batch = {"bboxes": gt_bboxes_xyxy, "cls": gt_cls}
                    stat["tp"] = self._process_batch(aggregated_pred, gt_batch, task=t)["tp"]
                else:
                    stat["tp"] = torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device)

            self.metrics[t].update_stats(stat)

            if self.args.plots and nl:
                det = torch.cat([
                    aggregated_pred["bboxes"],
                    aggregated_pred[f"conf_{t}"].unsqueeze(1),
                    aggregated_pred[f"cls_{t}"].unsqueeze(1),
                ], 1) if npr > 0 else None
                self.confusion_matrices[t].process_batch(detections=det, gt_bboxes=gt_bboxes_xyxy, gt_cls=gt_cls_t)

    def _process_batch(
        self, preds: dict[str, torch.Tensor], batch: dict[str, Any], task: int = 0
    ) -> dict[str, np.ndarray]:
        """Compute true positive matrix. Aligns with upstream (preds, batch) -> dict.

        Args:
            preds (dict[str, torch.Tensor]): Prediction dict with bboxes, cls_{task}.
            batch (dict[str, Any]): Batch dict with bboxes, cls (M, num_tasks) or (M,).
            task (int): Task index.

        Returns:
            (dict[str, np.ndarray]): Dictionary with 'tp' key (numpy array of shape (N, niou)).
        """
        gt_cls = batch["cls"][:, task] if batch["cls"].dim() > 1 else batch["cls"]
        pred_cls = preds[f"cls_{task}"]

        if gt_cls.shape[0] == 0 or pred_cls.shape[0] == 0:
            return {"tp": np.zeros((pred_cls.shape[0], self.niou), dtype=bool)}

        iou = box_iou(batch["bboxes"], preds["bboxes"])
        return {"tp": self.match_predictions(pred_cls, gt_cls, iou).cpu().numpy()}

    def finalize_metrics(self, *args, **kwargs):
        """Finalize metrics. Process remaining SAHI images."""
        if self.sahi_enabled and self.sahi_aggregator is not None:
            remaining = list(self.sahi_aggregator.image_crops.keys())
            if remaining:
                LOGGER.warning(f"Processing {len(remaining)} incomplete SAHI images at validation end")
                for img_key in remaining:
                    self._process_complete_image(img_key)
                    self.sahi_aggregator.cleanup_image(img_key)

        for m, cm in zip(self.metrics, self.confusion_matrices):
            m.speed = self.speed
            m.confusion_matrix = cm

        if self.args.plots:
            for t, cm in enumerate(self.confusion_matrices):
                prefix = f"task{t}_" if self.num_tasks > 1 else ""
                for normalize in True, False:
                    cm.plot(
                        save_dir=self.save_dir,
                        names=list(self.names[t].values()),
                        normalize=normalize,
                        on_plot=self.on_plot,
                        prefix=prefix,
                    )

    def gather_stats(self) -> None:
        """Gather stats from all GPUs (multitask: per-task stats)."""
        stats_to_gather = [m.stats for m in self.metrics]
        if RANK == 0:
            gathered = [None] * dist.get_world_size()
            dist.gather_object(stats_to_gather, gathered, dst=0)
            for t, m in enumerate(self.metrics):
                merged = {k: [] for k in m.stats}
                for rank_stats in gathered:
                    if rank_stats and t < len(rank_stats):
                        for k in merged:
                            merged[k].extend(rank_stats[t].get(k, []))
                m.stats = merged
            gathered_jdict = [None] * dist.get_world_size()
            dist.gather_object(self.jdict, gathered_jdict, dst=0)
            self.jdict = []
            for jdict in gathered_jdict:
                self.jdict.extend(jdict)
            self.seen = len(self.dataloader.dataset)
        else:
            dist.gather_object(stats_to_gather, None, dst=0)
            dist.gather_object(self.jdict, None, dst=0)
            self.jdict = []
            for m in self.metrics:
                m.clear_stats()

    def get_stats(self):
        """Compute and return per-task metrics statistics."""
        results = {}
        self.nt_per_class, self.nt_per_image = [], []
        fitness_values = []

        for i, m in enumerate(self.metrics):
            prefix = f"task{i}_" if self.num_tasks > 1 else ""
            m.process(save_dir=self.save_dir, plot=self.args.plots, on_plot=self.on_plot, prefix=f"{prefix}Box")
            self.nt_per_class.append(m.nt_per_class if m.nt_per_class is not None else np.zeros(self.nc[i]))
            self.nt_per_image.append(m.nt_per_image if m.nt_per_image is not None else np.zeros(self.nc[i]))

            for k, v in m.results_dict.items():
                results[f"{prefix}{k}"] = v
            fitness_key = f"{prefix}fitness"
            if fitness_key in results:
                fitness_values.append(results[fitness_key])

        if fitness_values:
            results["fitness"] = np.mean(fitness_values)

        return results

    def print_results(self) -> None:
        """Print validation metrics per class and per task."""
        metrics_keys = self.metrics[0].keys
        pf = "%22s" + "%11i" * 2 + "%11.3g" * len(metrics_keys)  # print format

        for i, m in enumerate(self.metrics):
            label = f"task{i}" if self.num_tasks > 1 else "all"
            LOGGER.info(pf % (label, self.seen, self.nt_per_class[i].sum(), *m.mean_results()))
            if self.nt_per_class[i].sum() == 0:
                LOGGER.warning(f"no labels found in {label} set, cannot compute metrics without labels")

        # Per-class results (SegmentMetrics has no .stats; use getattr for compatibility)
        for t in range(self.num_tasks):
            m_stats = getattr(self.metrics[t], "stats", {})
            tp_list = m_stats.get("tp", []) if isinstance(m_stats, dict) else []
            if self.args.verbose and not self.training and self.nc[t] > 1 and len(tp_list):
                for j, c in enumerate(self.metrics[t].ap_class_index):
                    LOGGER.info(
                        pf % (
                            self.names[t][c],
                            self.nt_per_image[t][c],
                            self.nt_per_class[t][c],
                            *self.metrics[t].class_result(j),
                        )
                    )

    def build_dataset(self, img_path: str, mode: str = "val", batch: int | None = None) -> torch.utils.data.Dataset:
        """Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`.

        Returns:
            (Dataset): YOLO dataset.
        """
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, stride=self.stride)

    def get_dataloader(self, dataset_path: str, batch_size: int) -> torch.utils.data.DataLoader:
        """Construct and return dataloader.

        Args:
            dataset_path (str): Path to the dataset.
            batch_size (int): Size of each batch.

        Returns:
            (torch.utils.data.DataLoader): DataLoader for validation.
        """
        dataset = self.build_dataset(dataset_path, batch=batch_size, mode="val")
        if isinstance(dataset, SAHIDataset):
            self.sahi_enabled = True
            from ultralytics.models.yolo.detect.sahi_val import SAHICropAggregator
            self.sahi_aggregator = SAHICropAggregator(self)
            self.sahi_aggregator.calculate_expected_crops(dataset)
        else:
            self.sahi_enabled = False
            self.sahi_aggregator = None

        return build_dataloader(
            dataset,
            batch_size,
            self.args.workers,
            shuffle=False,
            rank=-1,
            drop_last=self.args.compile,
            pin_memory=self.training,
        )

    def plot_val_samples(self, batch: dict[str, Any], ni: int) -> None:
        """Plot validation image samples."""
        plot_images(
            batch["img"],
            batch["batch_idx"],
            batch["cls"],
            batch["bboxes"],
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def plot_predictions(
        self,
        batch: dict[str, Any],
        preds: list[dict[str, torch.Tensor]],
        ni: int,
        max_det: int | None = None,
    ) -> None:
        """Plot predicted bounding boxes on input images and save the result.

        Args:
            batch (dict[str, Any]): Batch containing images and annotations.
            preds (list[dict[str, torch.Tensor]]): List of predictions from the model.
            ni (int): Batch index.
            max_det (int | None): Maximum number of detections to plot.
        """
        if not preds:
            return
        max_det = max_det or self.args.max_det
        batch_idx = torch.cat([torch.full_like(p["conf_0"], i) for i, p in enumerate(preds)], 0)
        cls = torch.cat([p["cls_0"][:max_det] for p in preds], 0)
        bboxes = ops.xyxy2xywh(torch.cat([p["bboxes"][:max_det] for p in preds], 0))
        confs = torch.cat([p["conf_0"][:max_det] for p in preds], 0)
        plot_images(
            batch["img"],
            batch_idx,
            cls,
            bboxes,
            confs=confs,
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names[0] if isinstance(self.names[0], dict) else self.names,
            on_plot=self.on_plot,
        )

    def save_one_txt(self, predn: dict[str, torch.Tensor], save_conf: bool, shape: tuple[int, int], file: Path) -> None:
        """Save YOLO detections to a txt file in normalized coordinates in a specific format.

        Args:
            predn (dict[str, torch.Tensor]): Dictionary containing predictions with keys 'bboxes', 'conf', and 'cls'.
            save_conf (bool): Whether to save confidence scores.
            shape (tuple[int, int]): Shape of the original image (height, width).
            file (Path): File path to save the detections.
        """
        from ultralytics.engine.results import Results

        conf = predn["conf_0"]
        cls = predn["cls_0"]
        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),
            path=None,
            names=self.names,
            boxes=torch.cat([predn["bboxes"], conf.unsqueeze(-1), cls.unsqueeze(-1)], dim=1),
        ).save_txt(file, save_conf=save_conf)

    def scale_preds(self, predn, pbatch):
        """Scale predictions to original image size."""
        return {
            **predn,
            "bboxes": ops.scale_boxes(
                pbatch["imgsz"], predn["bboxes"].clone(), pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"]
            ),
        }

    def pred_to_json(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> None:
        """Serialize YOLO predictions to COCO json format.

        Args:
            predn (dict[str, torch.Tensor]): Predictions dictionary containing 'bboxes', 'conf', and 'cls' keys with
                bounding box coordinates, confidence scores, and class predictions.
            pbatch (dict[str, Any]): Batch dictionary containing 'imgsz', 'ori_shape', 'ratio_pad', and 'im_file'.

        Examples:
             >>> result = {
             ...     "image_id": 42,
             ...     "file_name": "42.jpg",
             ...     "category_id": 18,
             ...     "bbox": [258.15, 41.29, 348.26, 243.78],
             ...     "score": 0.236,
             ... }
        """
        path = Path(pbatch["im_file"])
        stem = path.stem
        image_id = int(stem) if stem.isnumeric() else stem
        box = ops.xyxy2xywh(predn["bboxes"])  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
        for b, s, c in zip(box.tolist(), predn["conf_0"].tolist(), predn["cls_0"].tolist()):
            self.jdict.append(
                {
                    "image_id": image_id,
                    "file_name": path.name,
                    "category_id": self.class_map[int(c)],
                    "bbox": [round(x, 3) for x in b],
                    "score": round(s, 5),
                }
            )

    def eval_json(self, stats: dict[str, Any]) -> dict[str, Any]:
        """Evaluate YOLO output in JSON format and return performance statistics.

        Args:
            stats (dict[str, Any]): Current statistics dictionary.

        Returns:
            (dict[str, Any]): Updated statistics dictionary with COCO/LVIS evaluation results.
        """
        pred_json = self.save_dir / "predictions.json"  # predictions
        anno_json = (
            self.data["path"]
            / "annotations"
            / ("instances_val2017.json" if self.is_coco else f"lvis_v1_{self.args.split}.json")
        )  # annotations
        return self.coco_evaluate(stats, pred_json, anno_json)

    def coco_evaluate(
        self,
        stats: dict[str, Any],
        pred_json: str,
        anno_json: str,
        iou_types: str | list[str] = "bbox",
        suffix: str | list[str] = "Box",
    ) -> dict[str, Any]:
        """Evaluate COCO/LVIS metrics using faster-coco-eval library.

        Performs evaluation using the faster-coco-eval library to compute mAP metrics for object detection. Updates the
        provided stats dictionary with computed metrics including mAP50, mAP50-95, and LVIS-specific metrics if
        applicable.

        Args:
            stats (dict[str, Any]): Dictionary to store computed metrics and statistics.
            pred_json (str | Path): Path to JSON file containing predictions in COCO format.
            anno_json (str | Path): Path to JSON file containing ground truth annotations in COCO format.
            iou_types (str | list[str]): IoU type(s) for evaluation. Can be single string or list of strings. Common
                values include "bbox", "segm", "keypoints". Defaults to "bbox".
            suffix (str | list[str]): Suffix to append to metric names in stats dictionary. Should correspond to
                iou_types if multiple types provided. Defaults to "Box".

        Returns:
            (dict[str, Any]): Updated stats dictionary containing the computed COCO/LVIS evaluation metrics.
        """
        if self.args.save_json and (self.is_coco or self.is_lvis) and len(self.jdict):
            LOGGER.info(f"\nEvaluating faster-coco-eval mAP using {pred_json} and {anno_json}...")
            try:
                for x in (pred_json, anno_json):
                    assert x.is_file(), f"{x} file not found"
                iou_types = [iou_types] if isinstance(iou_types, str) else iou_types
                suffix = [suffix] if isinstance(suffix, str) else suffix
                check_requirements("faster-coco-eval>=1.6.7")
                from faster_coco_eval import COCO, COCOeval_faster

                anno = COCO(anno_json)
                pred = anno.loadRes(pred_json)
                for i, iou_type in enumerate(iou_types):
                    val = COCOeval_faster(
                        anno, pred, iouType=iou_type, lvis_style=self.is_lvis, print_function=LOGGER.info
                    )
                    val.params.imgIds = [int(Path(x).stem) for x in self.dataloader.dataset.im_files]
                    val.evaluate()
                    val.accumulate()
                    val.summarize()

                    # update mAP50-95 and mAP50
                    stats[f"metrics/mAP50({suffix[i][0]})"] = val.stats_as_dict["AP_50"]
                    stats[f"metrics/mAP50-95({suffix[i][0]})"] = val.stats_as_dict["AP_all"]
                    # record mAP for small, medium, large objects as well
                    stats["metrics/mAP_small(B)"] = val.stats_as_dict["AP_small"]
                    stats["metrics/mAP_medium(B)"] = val.stats_as_dict["AP_medium"]
                    stats["metrics/mAP_large(B)"] = val.stats_as_dict["AP_large"]
                    # update fitness
                    stats["fitness"] = 0.9 * val.stats_as_dict["AP_all"] + 0.1 * val.stats_as_dict["AP_50"]

                    if self.is_lvis:
                        stats[f"metrics/APr({suffix[i][0]})"] = val.stats_as_dict["APr"]
                        stats[f"metrics/APc({suffix[i][0]})"] = val.stats_as_dict["APc"]
                        stats[f"metrics/APf({suffix[i][0]})"] = val.stats_as_dict["APf"]

                if self.is_lvis:
                    stats["fitness"] = stats["metrics/mAP50-95(B)"]
            except Exception as e:
                LOGGER.warning(f"faster-coco-eval unable to run: {e}")
        return stats
