# Ultralytics YOLO 🚀, AGPL-3.0 license

import os
from pathlib import Path

import numpy as np
import torch

from ultralytics.data import build_dataloader, build_yolo_dataset, converter
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import LOGGER, ops, yaml_load
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics, box_iou
from ultralytics.utils.plotting import output_to_target, plot_images

"""
Multi-head validation design
----------------------------
Model output (before NMS)
  Single-head: (B, 4 + nc + nm, A)   – 4 box, nc conf/logits, nm masks
  Multi-head : (B, 4 + Σ(1+nc_i)+nm, A)  – 4 box, then for each head:
                                             1 conf + nc_i logits, finally nm masks
NMS output
  Single-head: List[tensor] of shape (K, 6+nm)   – (x1,y1,x2,y2,conf,cls,masks…)
  Multi-head : List[tensor] of shape (K, 4+2H+nm) – (x1,y1,x2,y2,
                                                     conf0,cls0, conf1,cls1, …,
                                                     masks…)
Hard-coded indices
  conf_t  = 4 + 2*t
  cls_t   = 5 + 2*t
These offsets are valid because the model builder guarantees the above layout.
Changing the order would require updating these constants and the NMS logic.
"""


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
        self.nt_per_class = None
        self.nt_per_image = None
        self.is_coco = False
        self.is_lvis = False
        self.class_map = None
        self.is_multihead = False
        self.tasks = []
        self.args.task = "detect"
        self.metrics = DetMetrics(save_dir=self.save_dir, on_plot=self.on_plot)
        self.iouv = torch.linspace(0.5, 0.95, 10)  # IoU vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()
        self.lb = []  # for autolabelling
        if self.args.save_hybrid:
            LOGGER.warning(
                "WARNING ⚠️ 'save_hybrid=True' will append ground truth to predictions for autolabelling.\n"
                "WARNING ⚠️ 'save_hybrid=True' will cause incorrect mAP.\n"
            )

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
        self.class_map = converter.coco80_to_coco91_class() if self.is_coco else list(range(len(model.names)))
        self.args.save_json |= (self.is_coco or self.is_lvis) and not self.training  # run on final val if training COCO
        self.names = model.names
        self.nc = len(model.names)

        # Determine multi-head configuration from dataset YAML. A list of lists
        # signals one label space per detection head.
        names_raw = yaml_load(self.data.get("yaml_file", "")).get("names", self.data.get("names"))
        if isinstance(names_raw, dict):
            names_raw = list(names_raw.values())
        if isinstance(names_raw, list) and names_raw and isinstance(names_raw[0], (list, tuple)):
            self.is_multihead = True
            self.tasks = [{"nc": len(n), "names": {i: name for i, name in enumerate(n)}} for n in names_raw]
        else:
            self.tasks = [{"nc": self.nc, "names": self.names}]

        if self.is_multihead:
            LOGGER.info("Multi-head model detected")
            self.metrics = [
                DetMetrics(save_dir=self.save_dir, on_plot=self.on_plot, names=t["names"]) for t in self.tasks
            ]
            self.confusion_matrix = [ConfusionMatrix(nc=t["nc"], conf=self.args.conf) for t in self.tasks]
            self.stats = [dict(tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[]) for _ in self.tasks]
        else:
            self.metrics.names = self.names
            self.metrics.plot = self.args.plots
            self.confusion_matrix = ConfusionMatrix(nc=self.nc, conf=self.args.conf)
            self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])

        self.seen = 0
        self.jdict = []

    def get_desc(self):
        """Return a formatted string summarizing class metrics of YOLO model."""
        return ("%22s" + "%11s" * 6) % ("Class", "Images", "Instances", "Box(P", "R", "mAP50", "mAP50-95)")

    def postprocess(self, preds):
        """Apply Non-maximum suppression to prediction outputs."""
        return ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            labels=self.lb,
            agnostic=self.args.single_cls or self.args.agnostic_nms,
            max_det=self.args.max_det,
            num_classes_per_head=[t["nc"] for t in self.tasks] if self.is_multihead else None,
            full_class_nms=getattr(self.args, "full_class_nms", False),
        )

    def _prepare_batch(self, si, batch):
        """Prepares a batch of images and annotations for validation."""
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx]
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
        if len(cls):
            bbox = ops.xywh2xyxy(bbox) * torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]]  # target boxes
            ops.scale_boxes(imgsz, bbox, ori_shape, ratio_pad=ratio_pad)  # native-space labels
        return {"cls": cls, "bbox": bbox, "ori_shape": ori_shape, "imgsz": imgsz, "ratio_pad": ratio_pad}

    def _prepare_pred(self, pred, pbatch):
        """Prepares a batch of images and annotations for validation."""
        predn = pred.clone()
        ops.scale_boxes(
            pbatch["imgsz"], predn[:, :4], pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"]
        )  # native-space pred
        return predn

    def update_metrics(self, preds, batch):
        """Metrics."""
        for si, pred in enumerate(preds):
            self.seen += 1
            npr = len(pred)
            if self.is_multihead:
                stat = [
                    dict(
                        conf=torch.zeros(0, device=self.device),
                        pred_cls=torch.zeros(0, device=self.device),
                        tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
                    )
                    for _ in self.tasks
                ]
            else:
                stat = dict(
                    conf=torch.zeros(0, device=self.device),
                    pred_cls=torch.zeros(0, device=self.device),
                    tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
                )

            pbatch = self._prepare_batch(si, batch)
            cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")
            nl = len(cls)
            if self.is_multihead:
                for t in range(len(self.tasks)):
                    gt_cls = cls[:, t] if cls.ndim > 1 else cls
                    stat[t]["target_cls"] = gt_cls
                    stat[t]["target_img"] = gt_cls.unique()
            else:
                stat["target_cls"] = cls
                stat["target_img"] = cls.unique()
            if npr == 0:
                if nl:
                    for k in self.stats.keys():
                        self.stats[k].append(stat[k])
                    if self.args.plots:
                        self.confusion_matrix.process_batch(detections=None, gt_bboxes=bbox, gt_cls=cls)
                continue

            # Predictions
            if self.args.single_cls:
                pred[:, 5] = 0
            predn = self._prepare_pred(pred, pbatch)
            if self.is_multihead:
                for t in range(len(self.tasks)):
                    # prediction columns are arranged as [x1,y1,x2,y2, conf0,cls0, conf1,cls1, ...]
                    stat[t]["conf"] = predn[:, 4 + 2 * t]  # confidence for task t
                    stat[t]["pred_cls"] = predn[:, 5 + 2 * t]  # class index for task t
                    if nl:
                        stat[t]["tp"] = self._process_batch(predn, bbox, cls[:, t] if cls.ndim > 1 else cls, task=t)
                        if self.args.plots and t == 0:
                            self.confusion_matrix[t].process_batch(
                                predn[:, :6], bbox, cls[:, t] if cls.ndim > 1 else cls
                            )
                    for k in self.stats[t].keys():
                        self.stats[t][k].append(stat[t][k])
            else:
                stat["conf"] = predn[:, 4]
                stat["pred_cls"] = predn[:, 5]

                if nl:
                    stat["tp"] = self._process_batch(predn, bbox, cls)
                    if self.args.plots:
                        self.confusion_matrix.process_batch(predn, bbox, cls)
                for k in self.stats.keys():
                    self.stats[k].append(stat[k])

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

    def finalize_metrics(self, *args, **kwargs):
        """Set final values for metrics speed and confusion matrix."""
        if self.is_multihead:
            for m, cm in zip(self.metrics, self.confusion_matrix):
                m.speed = self.speed
                m.confusion_matrix = cm
        else:
            self.metrics.speed = self.speed
            self.metrics.confusion_matrix = self.confusion_matrix

    def get_stats(self):
        """Returns metrics statistics and results dictionary."""
        if self.is_multihead:
            results = {}
            self.nt_per_class, self.nt_per_image = [], []
            for i, (m, st) in enumerate(zip(self.metrics, self.stats)):
                stats = {k: torch.cat(v, 0).cpu().numpy() for k, v in st.items()}
                ntc = np.bincount(stats["target_cls"].astype(int), minlength=self.tasks[i]["nc"])
                nti = np.bincount(stats["target_img"].astype(int), minlength=self.tasks[i]["nc"])
                self.nt_per_class.append(ntc)
                self.nt_per_image.append(nti)
                stats.pop("target_img", None)
                if len(stats) and stats["tp"].any():
                    m.process(**stats)
                results.update({f"task{i}_{k}": v for k, v in m.results_dict.items()})
            return results
        else:
            stats = {k: torch.cat(v, 0).cpu().numpy() for k, v in self.stats.items()}  # to numpy
            self.nt_per_class = np.bincount(stats["target_cls"].astype(int), minlength=self.nc)
            self.nt_per_image = np.bincount(stats["target_img"].astype(int), minlength=self.nc)
            stats.pop("target_img", None)
            if len(stats) and stats["tp"].any():
                self.metrics.process(**stats)
            return self.metrics.results_dict

    def print_results(self):
        """Prints training/validation set metrics per class."""
        metrics_keys = self.metrics[0].keys if self.is_multihead else self.metrics.keys
        pf = "%22s" + "%11i" * 2 + "%11.3g" * len(metrics_keys)
        if self.is_multihead:
            for i, m in enumerate(self.metrics):
                LOGGER.info(pf % (f"task{i}", self.seen, self.nt_per_class[i].sum(), *m.mean_results()))
        else:
            LOGGER.info(pf % ("all", self.seen, self.nt_per_class.sum(), *self.metrics.mean_results()))
            if self.nt_per_class.sum() == 0:
                LOGGER.warning(
                    f"WARNING ⚠️ no labels found in {self.args.task} set, can not compute metrics without labels"
                )

        # Print results per class
        if not self.is_multihead:
            if self.args.verbose and not self.training and self.nc > 1 and len(self.stats):
                for i, c in enumerate(self.metrics.ap_class_index):
                    LOGGER.info(
                        pf % (self.names[c], self.nt_per_image[c], self.nt_per_class[c], *self.metrics.class_result(i))
                    )

            if self.args.plots:
                for normalize in True, False:
                    self.confusion_matrix.plot(
                        save_dir=self.save_dir, names=self.names.values(), normalize=normalize, on_plot=self.on_plot
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
        cls_col = 5 + 2 * task  # column containing the predicted class for this task
        return self.match_predictions(detections[:, cls_col], gt_cls, iou)

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
            batch["cls"].squeeze(-1),
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
