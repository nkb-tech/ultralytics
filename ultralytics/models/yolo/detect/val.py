# Ultralytics YOLO 🚀, AGPL-3.0 license

import os
from pathlib import Path

import numpy as np
import torch

from ultralytics.data import build_dataloader, build_yolo_dataset, converter
from ultralytics.engine.validator import BaseValidator
from ultralytics.models.yolo.detect.sahi_debugger import SAHIValidationDebugger
from ultralytics.models.yolo.detect.sahi_val import SAHICropAggregator
from ultralytics.utils import LOGGER, ops, yaml_load
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics, box_iou
from ultralytics.utils.plotting import output_to_target, plot_images


class DetectionValidator(BaseValidator):
    """
    A class extending the BaseValidator class for validation based on a detection model.

    Example:
        ```python
        from ultralytics.models.yolo.detect import DetectionValidator

        args = dict(model="yolov8n.pt", data="coco8.yaml")
        validator = DetectionValidator(args=args)
        validator()
        ```
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initialize detection model with necessary variables and settings."""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.is_coco = False
        self.is_lvis = False
        self.class_map = None
        self.args.task = "detect"
        self.metrics: list[DetMetrics] = [DetMetrics(save_dir=self.save_dir, on_plot=self.on_plot), ]
        self.iouv = torch.linspace(0.5, 0.95, 10)  # IoU vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()
        self.lb = []  # for autolabelling
        if self.args.save_hybrid:
            LOGGER.warning(
                "WARNING ⚠️ 'save_hybrid=True' will append ground truth to predictions for autolabelling.\n"
                "WARNING ⚠️ 'save_hybrid=True' will cause incorrect mAP.\n"
            )
        if self.args.sahi_val_debug:
            self.sahi_debugger = SAHIValidationDebugger()
        self.sahi_aggregator = None
        self.sahi_enabled = False
                        
    def preprocess(self, batch):
        """Preprocesses batch of images for YOLO training."""
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = (batch["img"].half() if self.args.half else batch["img"].float()) / 255
        for k in ["batch_idx", "cls", "bboxes"]:
            batch[k] = batch[k].to(self.device)

        if self.args.save_hybrid:
            height, width = batch["img"].shape[2:]
            nb = len(batch["img"])
            bboxes = batch["bboxes"] * torch.tensor((width, height, width, height), device=self.device)
            self.lb = [
                torch.cat([batch["cls"][batch["batch_idx"] == i], bboxes[batch["batch_idx"] == i]], dim=-1)
                for i in range(nb)
            ]

        return batch

    def init_metrics(self, model):
        """Initialize evaluation metrics for YOLO."""
        val = self.data.get(self.args.split, "")  # validation path
        self.is_coco = (
            isinstance(val, str)
            and "coco" in val
            and (val.endswith(f"{os.sep}val2017.txt") or val.endswith(f"{os.sep}test-dev2017.txt"))
        )  # is COCO
        self.is_lvis = isinstance(val, str) and "lvis" in val and not self.is_coco  # is LVIS
        data_names = self.data.get('names', [])
        LOGGER.debug(f"Loaded names: {data_names}")
        if data_names:
            if isinstance(data_names[0], list):
                # Мультитаск: names: [['heavy', 'light'], ['dmg', 'undmg']]
                self.names = [{i: name for i, name in enumerate(task_names)} for task_names in data_names]
                self.nc = [len(task_names) for task_names in data_names]
            elif isinstance(data_names[0], dict):
                # Мультитаск: names: [['heavy', 'light'], ['dmg', 'undmg']]
                self.names = data_names
                self.nc = [len(task_dict) for task_dict in data_names]
            else:
                # names: ['heavy', 'light', 'art', 'truck', 'car', 'vehicle']
                self.names = [{i: name for i, name in enumerate(data_names)}]
                self.nc = [len(data_names)]
        else:
            self.names = [{i: f'class{i}' for i in range(nc_i)} for nc_i in self.data.get('nc', [1])]
            self.nc = self.data.get('nc', [1])
            
        LOGGER.debug(f"nc from data: {self.data.get('nc')}")
        self.class_map = converter.coco80_to_coco91_class() if self.is_coco else list(range(self.nc[0]))
        self.args.save_json |= (self.is_coco or self.is_lvis) and not self.training
        self.num_tasks = len(self.nc)
        LOGGER.debug(f"num_tasks: {self.num_tasks}")
        self.metrics = [
            DetMetrics(save_dir=self.save_dir, on_plot=self.on_plot, names=names)
            for names in self.names
        ]
        self.confusion_matrices = [ConfusionMatrix(nc=nc_i, conf=self.args.conf) for nc_i in self.nc]
        self.stats = [dict(tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[]) for _ in self.nc]

        self.seen = 0
        self.jdict = []
        
        if hasattr(self.args, 'sahi') and self.args.sahi:
            if hasattr(self.args, 'val_cut_strategy') and self.args.val_cut_strategy == 'grid':
                self.sahi_aggregator = SAHICropAggregator(self)
                self.sahi_enabled = True       
                if hasattr(self.dataloader, 'dataset'):
                    self.sahi_aggregator.calculate_expected_crops(self.dataloader.dataset)
                    
                LOGGER.info("SAHI aggregator initialized for grid validation")
    

    def get_desc(self):
        """Return a formatted string summarizing class metrics of YOLO model."""
        return ("%22s" + "%11s" * 6) % ("Class", "Images", "Instances", "Box(P", "R", "mAP50", "mAP50-95)")

    def postprocess(self, preds):
        """Apply Non-maximum suppression to prediction outputs."""
        
        # Добавим диагностику и правильную обработку tuple
        if isinstance(preds, tuple):
            LOGGER.info(f"postprocess input is tuple with {len(preds)} elements")
            if len(preds) > 0:
                LOGGER.info(f"First element shape: {preds[0].shape if hasattr(preds[0], 'shape') else type(preds[0])}")
            # Для мультитаск модели preds может быть (predictions, proto) или просто predictions
            # Берем первый элемент если это tuple
            actual_preds = preds[0] if isinstance(preds[0], torch.Tensor) else preds
        else:
            actual_preds = preds
            
        LOGGER.info(f"Actual predictions shape: {actual_preds.shape if hasattr(actual_preds, 'shape') else type(actual_preds)}")
        LOGGER.info(f"self.nc (num classes per task): {self.nc}")
        
        if self.sahi_enabled:
            # Сохраняем актуальные предсказания, а не tuple
            self._last_raw_preds = actual_preds.clone() if isinstance(actual_preds, torch.Tensor) else actual_preds
            
        return ops.non_max_suppression(
            actual_preds,  # Используем actual_preds вместо preds
            self.args.conf,
            self.args.iou,
            labels=self.lb,
            agnostic=self.args.single_cls or self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=[1] + self.nc[1:] if self.args.single_cls else self.nc,
        )


    def _prepare_batch(self, si, batch):
        """Prepares a batch of images and annotations for validation."""
        # Проверяем формат batch_idx
        if 'batch_idx' in batch and batch['batch_idx'].numel() > 0:
            idx = batch["batch_idx"] == si
        else:
            # Если batch_idx пустой или отсутствует, обрабатываем все данные для изображения si
            idx = torch.ones(len(batch["cls"]), dtype=torch.bool, device=batch["cls"].device) if si == 0 else torch.zeros(len(batch["cls"]), dtype=torch.bool, device=batch["cls"].device)
        
        cls = batch["cls"][idx] if idx.any() else batch["cls"]
        bbox = batch["bboxes"][idx] if idx.any() else batch["bboxes"]
        
        # Обработка ori_shape с учетом разных форматов
        if isinstance(batch["ori_shape"], list):
            ori_shape = batch["ori_shape"][si] if si < len(batch["ori_shape"]) else batch["ori_shape"][0]
        else:
            ori_shape = batch["ori_shape"][si] if len(batch["ori_shape"]) > si else batch["ori_shape"][0]
        
        imgsz = batch["img"].shape[2:]
        
        # Обработка ratio_pad с учетом разных форматов
        if isinstance(batch["ratio_pad"], list):
            ratio_pad = batch["ratio_pad"][si] if si < len(batch["ratio_pad"]) else batch["ratio_pad"][0]
        else:
            ratio_pad = batch["ratio_pad"][si] if len(batch["ratio_pad"]) > si else batch["ratio_pad"][0]
        
        if len(cls):
            bbox = ops.xywh2xyxy(bbox) * torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]]  # target boxes
            ops.scale_boxes(imgsz, bbox, ori_shape, ratio_pad=ratio_pad)  # native-space labels
        
        return {"cls": cls, "bbox": bbox, "ori_shape": ori_shape, "imgsz": imgsz, "ratio_pad": ratio_pad}

    def update_metrics(self, preds, batch):
        """Metrics."""
        if hasattr(self, 'sahi_debugger'):
            self.sahi_debugger.log_batch_info(batch, self.batch_i)
            self.sahi_debugger.track_crop_mapping(batch, self.batch_i)
            self.sahi_debugger.log_predictions(preds, batch, self.batch_i)
        
          # Добавим диагностику batch
        LOGGER.info(f"update_metrics batch keys: {batch.keys()}")
        if 'original_img_idx' in batch:
            LOGGER.info(f"  original_img_idx type: {type(batch['original_img_idx'])}")
            if hasattr(batch['original_img_idx'], 'shape'):
                LOGGER.info(f"  original_img_idx shape: {batch['original_img_idx'].shape}")
        
        if self.sahi_enabled and self.sahi_aggregator is not None:
            if hasattr(self, '_last_raw_preds'):
                raw_preds = self._last_raw_preds
            else:
                LOGGER.info("Raw predictions not available for SAHI aggregation")
                self._update_metrics_standard(preds, batch)
                return
            
            # Add predictions to aggregator
            success = self.sahi_aggregator.add_crop_predictions(batch, raw_preds, preds)
            
            if not success:
                LOGGER.warning("Can not calculate add_crop_predictions for sahi")
                self._update_metrics_standard(preds, batch)
            
            # Process completed images
            completed_images = self.sahi_aggregator.get_completed_images()
            
            for img_key in completed_images:
                self._process_complete_image(img_key)
                
            # Clean up processed images
            for img_key in completed_images:
                del self.sahi_aggregator.image_crops[img_key]
                
            return
        
        self._update_metrics_standard(preds, batch)

    def _update_metrics_standard(self, preds, batch):
        """Standard metrics update (original implementation)."""
        for si, pred in enumerate(preds):
            self.seen += 1
            npr = len(pred)
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

            for t in range(self.num_tasks):
                gt_cls = cls[:, t]
                stat[t]["target_cls"] = gt_cls
                stat[t]["target_img"] = gt_cls.unique()
            
            if npr == 0:
                if nl:
                    for t in range(self.num_tasks):
                        for k in self.stats[t].keys():
                            self.stats[t][k].append(stat[t][k])
                        if self.args.plots:
                            self.confusion_matrices[t].process_batch(
                                detections=None,
                                gt_bboxes=bbox,
                                gt_cls=cls[:, t],
                            )
                continue

            # Predictions
            if self.args.single_cls:
                pred[:, 5] = 0
            predn = self._prepare_pred(pred, pbatch)
            for t in range(self.num_tasks):
                # prediction columns are arranged as [x1,y1,x2,y2, conf0,cls0, conf1,cls1, ...]
                stat[t]["conf"] = predn[..., 4 + 2 * t]  # confidence for task t
                stat[t]["pred_cls"] = predn[..., 5 + 2 * t]  # class index for task t
                if nl:
                    stat[t]["tp"] = self._process_batch(predn, bbox, cls[:, t], task=t)
                    if self.args.plots:
                        det = predn[..., [0, 1, 2, 3, 4 + 2 * t, 5 + 2 * t]]
                        self.confusion_matrices[t].process_batch(
                            det, bbox, cls[:, t]
                        )
                for k in self.stats[t].keys():
                    self.stats[t][k].append(stat[t][k])

            # Save
            if self.args.save_json:
                self.pred_to_json(predn, batch["im_file"][si])
            if self.args.save_txt:
                self.save_one_txt(
                    predn,
                    self.args.save_conf,
                    pbatch["ori_shape"],
                    self.save_dir / "labels" / f"{Path(batch['im_file'][si]).stem}.txt",
                )

    def _process_complete_image(self, img_key):
        """Process a complete image with all crops aggregated."""
        
        LOGGER.info(f"Processing complete image: {img_key}")
        
        # Get aggregated predictions
        aggregated_preds_raw = self.sahi_aggregator.get_aggregated_predictions(img_key)
        
        LOGGER.info(f"  Aggregated predictions shape before NMS: {aggregated_preds_raw.shape}")
        
        # Apply NMS to aggregated predictions
        if len(aggregated_preds_raw) > 0:
            # Reshape for NMS function (add batch dimension)
            preds_for_nms = aggregated_preds_raw.unsqueeze(0)
            
            # Transpose back to expected format [batch, outputs, anchors]
            preds_for_nms = preds_for_nms.permute(0, 2, 1)
            
            LOGGER.info(f"  Predictions for NMS shape: {preds_for_nms.shape}")
            
            # Apply standard postprocessing (NMS)
            nms_results = self.postprocess(preds_for_nms)
            aggregated_preds = nms_results[0] if nms_results else torch.empty((0, 4 + 2 * len(self.nc)), device=self.device)
        else:
            aggregated_preds = torch.empty((0, 4 + 2 * len(self.nc)), device=self.device)
        
        LOGGER.info(f"  Aggregated predictions shape after NMS: {aggregated_preds.shape}")
        
        # Get original image ground truth
        img_idx = self.sahi_aggregator.image_crops[img_key]['original_img_idx']
        original_shape = self.sahi_aggregator.image_crops[img_key]['original_shape']
        
        # Load original GT for this image
        original_labels = self.dataloader.dataset.labels[img_idx]
        
        # Prepare ground truth data
        if 'cls' in original_labels and len(original_labels['cls']) > 0:
            gt_cls = torch.tensor(original_labels['cls'], device=self.device, dtype=torch.float32)
            gt_bboxes = torch.tensor(original_labels['bboxes'], device=self.device, dtype=torch.float32)
        else:
            num_tasks = len(self.nc)
            gt_cls = torch.empty((0, num_tasks), device=self.device, dtype=torch.float32)
            gt_bboxes = torch.empty((0, 4), device=self.device, dtype=torch.float32)
        
        # Ensure cls has correct shape
        if gt_cls.dim() == 1 and len(self.nc) > 1:
            # If single task labels, reshape for compatibility
            gt_cls = gt_cls.unsqueeze(1)
        
        LOGGER.info(f"  GT cls shape: {gt_cls.shape}, GT bboxes shape: {gt_bboxes.shape}")
        
        # Create synthetic batch for metrics calculation
        # batch_idx should match the number of ground truth instances
        batch_idx_values = torch.zeros(len(gt_bboxes), device=self.device, dtype=torch.long)
        
        synthetic_batch = {
            'cls': gt_cls,  # Shape: [num_instances, num_tasks]
            'bboxes': gt_bboxes,  # Shape: [num_instances, 4]
            'batch_idx': batch_idx_values,  # Shape: [num_instances]
            'ori_shape': [original_shape],
            'img': torch.zeros((1, 3, 640, 640), device=self.device),  # Placeholder
            'im_file': [self.dataloader.dataset.im_files[img_idx]],
            'resized_shape': [original_shape],  # For full image validation
            'ratio_pad': [(1.0, 1.0)],
        }
        
        # Update metrics with aggregated results
        self._update_metrics_standard([aggregated_preds], synthetic_batch)
        

            
    def finalize_metrics(self, *args, **kwargs):
        """Set final values for metrics speed and confusion matrices."""
        if self.sahi_enabled and self.sahi_aggregator is not None:
            remaining_images = list(self.sahi_aggregator.image_crops.keys())
            if remaining_images:
                LOGGER.warning(f"Processing {len(remaining_images)} incomplete images at validation end")
                for img_key in remaining_images:
                    crops_processed = len(self.sahi_aggregator.image_crops[img_key]['processed_crops'])
                    img_idx = self.sahi_aggregator.image_crops[img_key].get('original_img_idx', -1)
                    expected = self.sahi_aggregator.expected_crops_per_image.get(img_idx, 'unknown')
                    LOGGER.debug(f"Image {img_key}: {crops_processed}/{expected} crops processed")
        
        for m, cm in zip(self.metrics, self.confusion_matrices):
            m.speed = self.speed
            m.confusion_matrix = cm

    def get_stats(self):
        """Returns metrics statistics and results dictionary."""
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
        """Prints training/validation set metrics per class."""
        metrics_keys = self.metrics[0].keys
        pf = "%22s" + "%11i" * 2 + "%11.3g" * len(metrics_keys)
        for i, m in enumerate(self.metrics):
            LOGGER.info(pf % (f"task{i}", self.seen, self.nt_per_class[i].sum(), *m.mean_results()))
            if self.nt_per_class[i].sum() == 0:
                LOGGER.warning(
                    f"WARNING ⚠️ no labels found in task{i} set, can not compute metrics without labels"
                )

        # Print results per class
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

    def _process_batch(self, detections, gt_bboxes, gt_cls, task=0):
        """
        Return correct prediction matrix.

        Args:
            detections (torch.Tensor): Tensor of shape (N, 6) representing detections where each detection is
                (x1, y1, x2, y2, conf, class).
            gt_bboxes (torch.Tensor): Tensor of shape (M, 4) representing ground-truth bounding box coordinates. Each
                bounding box is of the format: (x1, y1, x2, y2).
            gt_cls (torch.Tensor): Tensor of shape (M,) representing target class indices.

        Returns:
            (torch.Tensor): Correct prediction matrix of shape (N, 10) for 10 IoU levels.

        Note:
            The function does not return any value directly usable for metrics calculation. Instead, it provides an
            intermediate representation used for evaluating predictions against ground truth.
        """
        iou = box_iou(gt_bboxes, detections[:, :4])
        # each detection head contributes two columns: conf and class
        return self.match_predictions(detections[:, 5 + 2 * task], gt_cls, iou)

    def build_dataset(self, img_path, mode="val", batch=None):
        """
        Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, stride=self.stride)

    def get_dataloader(self, dataset_path, batch_size):
        """Construct and return dataloader."""
        dataset = self.build_dataset(dataset_path, batch=batch_size, mode="val")
        return build_dataloader(dataset, batch_size, self.args.workers, shuffle=False, rank=-1)  # return dataloader

    def plot_val_samples(self, batch, ni):
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

    def plot_predictions(self, batch, preds, ni):
        """Plots predicted bounding boxes on input images and saves the result."""
        plot_images(
            batch["img"],
            *output_to_target(preds, max_det=self.args.max_det),
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )  # pred

    def save_one_txt(self, predn, save_conf, shape, file):
        """Save YOLO detections to a txt file in normalized coordinates in a specific format."""
        from ultralytics.engine.results import Results

        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),
            path=None,
            names=self.names,
            boxes=predn[:, :6],
        ).save_txt(file, save_conf=save_conf)

    def pred_to_json(self, predn, filename):
        """Serialize YOLO predictions to COCO json format."""
        stem = Path(filename).stem
        image_id = int(stem) if stem.isnumeric() else stem
        box = ops.xyxy2xywh(predn[:, :4])  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
        for p, b in zip(predn.tolist(), box.tolist()):
            self.jdict.append(
                {
                    "image_id": image_id,
                    "category_id": self.class_map[int(p[5])]
                    + (1 if self.is_lvis else 0),  # index starts from 1 if it's lvis
                    "bbox": [round(x, 3) for x in b],
                    "score": round(p[4], 5),
                }
            )

    def eval_json(self, stats):
        """Evaluates YOLO output in JSON format and returns performance statistics."""
        if self.args.save_json and (self.is_coco or self.is_lvis) and len(self.jdict):
            pred_json = self.save_dir / "predictions.json"  # predictions
            anno_json = (
                self.data["path"]
                / "annotations"
                / ("instances_val2017.json" if self.is_coco else f"lvis_v1_{self.args.split}.json")
            )  # annotations
            pkg = "pycocotools" if self.is_coco else "lvis"
            LOGGER.info(f"\nEvaluating {pkg} mAP using {pred_json} and {anno_json}...")
            try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
                for x in pred_json, anno_json:
                    assert x.is_file(), f"{x} file not found"
                check_requirements("pycocotools>=2.0.6" if self.is_coco else "lvis>=0.5.3")
                if self.is_coco:
                    from pycocotools.coco import COCO  # noqa
                    from pycocotools.cocoeval import COCOeval  # noqa

                    anno = COCO(str(anno_json))  # init annotations api
                    pred = anno.loadRes(str(pred_json))  # init predictions api (must pass string, not Path)
                    val = COCOeval(anno, pred, "bbox")
                else:
                    from lvis import LVIS, LVISEval

                    anno = LVIS(str(anno_json))  # init annotations api
                    pred = anno._load_json(str(pred_json))  # init predictions api (must pass string, not Path)
                    val = LVISEval(anno, pred, "bbox")
                val.params.imgIds = [int(Path(x).stem) for x in self.dataloader.dataset.im_files]  # images to eval
                val.evaluate()
                val.accumulate()
                val.summarize()
                if self.is_lvis:
                    val.print_results()  # explicitly call print_results
                # update mAP50-95 and mAP50
                stats[self.metrics.keys[-1]], stats[self.metrics.keys[-2]] = (
                    val.stats[:2] if self.is_coco else [val.results["AP50"], val.results["AP"]]
                )
            except Exception as e:
                LOGGER.warning(f"{pkg} unable to run: {e}")
        return stats