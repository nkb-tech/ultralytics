# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

import sys
import time

import torch

from ultralytics.utils import LOGGER
from ultralytics.utils.metrics import batch_probiou, box_iou
from ultralytics.utils.ops import xywh2xyxy


def non_max_suppression(
    prediction,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    classes=None,
    agnostic: bool = False,
    multi_label: bool = False,
    labels=(),
    max_det: int = 300,
    nc: list[int] = [0,],  # list[int]: number of classes per task head
    max_time_img: float = 0.05,
    max_nms: int = 30000,
    max_wh: int = 7680,
    rotated: bool = False,
    end2end: bool = False,
    return_idxs: bool = False,
):
    """Perform non-maximum suppression (NMS) on prediction results.

    Supports multitask models where nc is a list of class counts per task head.
    Output layout per detection: [x1, y1, x2, y2, conf_0, cls_0, conf_1, cls_1, ..., extra].

    Args:
        prediction (torch.Tensor): Predictions (batch_size, 4 + sum(nc) + extra, num_boxes) BCN format,
            or (batch_size, num_boxes, 6) for end2end models.
        conf_thres (float): Confidence threshold (0.0 to 1.0).
        iou_thres (float): IoU threshold for NMS (0.0 to 1.0).
        classes (list[int], optional): Filter by class indices.
        agnostic (bool): Class-agnostic NMS.
        multi_label (bool): Allow multiple labels per box (single-task only).
        labels (list): A priori labels for autolabelling.
        max_det (int): Maximum detections per image.
        nc (list[int]): Number of classes per task head. Always a list.
        max_time_img (float): Max seconds per image.
        max_nms (int): Max boxes into torchvision NMS.
        max_wh (int): Max box dimension for class offset.
        rotated (bool): Handle Oriented Bounding Boxes.
        end2end (bool): End-to-end model (no NMS needed).
        return_idxs (bool): Return indices of kept detections.

    Returns:
        list[torch.Tensor]: Detections per image with shape (N, 4 + 2*num_tasks + extra).
        keepi (list[torch.Tensor]): Indices of kept detections if return_idxs=True.
    """
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"

    if isinstance(prediction, (list, tuple)):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output
    if classes is not None:
        classes = torch.tensor(classes, device=prediction.device)

    num_tasks = len(nc)
    total_nc = sum(nc)

    # End-to-end model fast path: input is (batch, max_det, 6) with [x,y,w,h, score, label]
    if prediction.shape[-1] == 6 or end2end:
        output = []
        for pred in prediction:
            pred = pred[pred[:, 4] > conf_thres][:max_det]
            if classes is not None:
                pred = pred[(pred[:, 5:6] == classes).any(1)]
            output.append(pred)
        return output

    # Standard NMS path: input is (batch, channels, anchors) BCN format
    bs = prediction.shape[0]
    total_nc = total_nc or (prediction.shape[1] - 4)
    extra = prediction.shape[1] - total_nc - 4
    mi = 4 + total_nc

    # Candidate filtering by first task confidence
    xc = prediction[:, 4:4 + nc[0]].amax(1) > conf_thres
    xinds = torch.arange(prediction.shape[-1], device=prediction.device).expand(bs, -1)[..., None]

    time_limit = 2.0 + max_time_img * bs
    multi_label &= nc[0] > 1 and num_tasks == 1  # multi_label only for single-task

    prediction = prediction.transpose(-1, -2)  # BCN -> BNC
    if not rotated:
        prediction[..., :4] = xywh2xyxy(prediction[..., :4])

    t = time.time()
    output = [torch.zeros((0, 4 + 2 * num_tasks + extra), device=prediction.device)] * bs
    keepi = [torch.zeros((0, 1), device=prediction.device)] * bs

    for xi, (x, xk) in enumerate(zip(prediction, xinds)):
        # Apply constraints
        # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
        filt = xc[xi]
        x = x[filt]
        if return_idxs:
            xk = xk[filt]

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]) and not rotated:
            lb = labels[xi]
            v = torch.zeros((len(lb), total_nc + extra + 4), device=x.device)
            v[:, :4] = xywh2xyxy(lb[:, 1:5])
            v[range(len(lb)), lb[:, 0].long() + 4] = 1.0
            x = torch.cat((x, v), 0)

        if not x.shape[0]:
            continue

        # Split into box, class scores, extra
        box = x[:, :4]
        all_cls = x[:, 4:mi]
        mask = x[:, mi:]

        if multi_label:
            # Single-task multi_label: one row per (box, class) pair above threshold
            i, j = torch.where(all_cls > conf_thres)
            x = torch.cat((box[i], all_cls[i, j, None], j[:, None].float(), mask[i]), 1)
            if return_idxs:
                xk = xk[i]
        else:
            # Best class per task head
            confs, clss = [], []
            offset = 0
            for nc_i in nc:
                task_cls = all_cls[:, offset:offset + nc_i]
                conf_i, j_i = task_cls.max(1, keepdim=True)
                confs.append(conf_i)
                clss.append(j_i.float())
                offset += nc_i

            # Filter by first task confidence
            conf_mask = confs[0].view(-1) > conf_thres
            box = box[conf_mask]
            mask = mask[conf_mask]
            confs = [c[conf_mask] for c in confs]
            clss = [j[conf_mask] for j in clss]
            if return_idxs:
                xk = xk[conf_mask]

            # Layout: [box, conf0, cls0, conf1, cls1, ..., extra]
            x = torch.cat([box] + sum([[c, j] for c, j in zip(confs, clss)], []) + [mask], 1)

        conf = x[:, 4]
        j = x[:, 5]

        # Filter by class
        if classes is not None:
            filt = (j.view(-1, 1) == classes.float()).any(1)
            x = x[filt]
            if return_idxs:
                xk = xk[filt]
            conf, j = x[:, 4], x[:, 5]

        # Check shape
        n = x.shape[0]
        if not n:
            continue
        if n > max_nms:
            filt = conf.argsort(descending=True)[:max_nms]
            x = x[filt]
            if return_idxs:
                xk = xk[filt]
            conf, j = x[:, 4], x[:, 5]

        # NMS class offsets
        if agnostic:
            c = torch.zeros_like(j.view(-1, 1))
        elif num_tasks > 1:
            # Multi-task: create unique ID from all task classes
            unique_id = x[:, 5].view(-1, 1)
            multiplier = nc[0]
            for t in range(1, num_tasks):
                task_cls_col = x[:, 5 + 2 * t]
                unique_id = unique_id + task_cls_col.view(-1, 1) * multiplier
                multiplier *= nc[t]
            c = unique_id * max_wh
        else:
            c = j.view(-1, 1) * max_wh

        scores = conf
        if rotated:
            boxes = torch.cat((x[:, :2] + c, x[:, 2:4], x[:, -1:]), dim=-1)
            i = TorchNMS.fast_nms(boxes, scores, iou_thres, iou_func=batch_probiou)
        else:
            boxes = x[:, :4] + c
            if "torchvision" in sys.modules:
                import torchvision
                i = torchvision.ops.nms(boxes.float(), scores.float(), iou_thres)
            else:
                i = TorchNMS.nms(boxes, scores, iou_thres)
        i = i[:max_det]

        output[xi] = x[i]
        if return_idxs:
            keepi[xi] = xk[i].view(-1)
        if (time.time() - t) > time_limit:
            LOGGER.warning(f"NMS time limit {time_limit:.3f}s exceeded")
            break

    return (output, keepi) if return_idxs else output


class TorchNMS:
    """Ultralytics custom NMS implementation optimized for YOLO.

    Provides static methods for NMS operations including standard NMS, fast NMS,
    and batched NMS for multi-class scenarios.
    """

    @staticmethod
    def fast_nms(
        boxes,
        scores,
        iou_threshold,
        use_triu=True,
        iou_func=box_iou,
        exit_early=True,
    ):
        """Fast-NMS using upper triangular matrix operations.

        Args:
            boxes (torch.Tensor): Bounding boxes (N, 4) in xyxy format.
            scores (torch.Tensor): Confidence scores (N,).
            iou_threshold (float): IoU threshold for suppression.
            use_triu (bool): Use torch.triu for upper triangular matrix.
            iou_func (callable): Function to compute IoU.
            exit_early (bool): Return empty if no boxes.

        Returns:
            (torch.Tensor): Indices of kept boxes.
        """
        if boxes.numel() == 0 and exit_early:
            return torch.empty((0,), dtype=torch.int64, device=boxes.device)

        sorted_idx = torch.argsort(scores, descending=True)
        boxes = boxes[sorted_idx]
        ious = iou_func(boxes, boxes)
        if use_triu:
            ious = ious.triu_(diagonal=1)
            pick = torch.nonzero((ious >= iou_threshold).sum(0) <= 0).squeeze_(-1)
        else:
            n = boxes.shape[0]
            row_idx = torch.arange(n, device=boxes.device).view(-1, 1).expand(-1, n)
            col_idx = torch.arange(n, device=boxes.device).view(1, -1).expand(n, -1)
            upper_mask = row_idx < col_idx
            ious = ious * upper_mask
            scores_ = scores[sorted_idx]
            scores_[~((ious >= iou_threshold).sum(0) <= 0)] = 0
            scores[sorted_idx] = scores_
            pick = torch.topk(scores_, scores_.shape[0]).indices
        return sorted_idx[pick]

    @staticmethod
    def nms(boxes, scores, iou_threshold):
        """Optimized NMS with early termination matching torchvision behavior.

        Args:
            boxes (torch.Tensor): Bounding boxes (N, 4) in xyxy format.
            scores (torch.Tensor): Confidence scores (N,).
            iou_threshold (float): IoU threshold for suppression.

        Returns:
            (torch.Tensor): Indices of kept boxes.
        """
        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.int64, device=boxes.device)

        x1, y1, x2, y2 = boxes.unbind(1)
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort(0, descending=True)

        keep = torch.zeros(order.numel(), dtype=torch.int64, device=boxes.device)
        keep_idx = 0
        while order.numel() > 0:
            i = order[0]
            keep[keep_idx] = i
            keep_idx += 1

            if order.numel() == 1:
                break

            rest = order[1:]
            xx1 = torch.maximum(x1[i], x1[rest])
            yy1 = torch.maximum(y1[i], y1[rest])
            xx2 = torch.minimum(x2[i], x2[rest])
            yy2 = torch.minimum(y2[i], y2[rest])

            w = (xx2 - xx1).clamp_(min=0)
            h = (yy2 - yy1).clamp_(min=0)
            inter = w * h
            if inter.sum() == 0:
                order = rest
                continue
            iou = inter / (areas[i] + areas[rest] - inter)
            order = rest[iou <= iou_threshold]

        return keep[:keep_idx]

    @staticmethod
    def batched_nms(boxes, scores, idxs, iou_threshold, use_fast_nms=False):
        """Batched NMS for class-aware suppression.

        Args:
            boxes (torch.Tensor): Bounding boxes (N, 4) in xyxy format.
            scores (torch.Tensor): Confidence scores (N,).
            idxs (torch.Tensor): Class indices (N,).
            iou_threshold (float): IoU threshold.
            use_fast_nms (bool): Use fast NMS instead of standard.

        Returns:
            (torch.Tensor): Indices of kept boxes.
        """
        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.int64, device=boxes.device)

        max_coordinate = boxes.max()
        offsets = idxs.to(boxes) * (max_coordinate + 1)
        boxes_for_nms = boxes + offsets[:, None]

        return (
            TorchNMS.fast_nms(boxes_for_nms, scores, iou_threshold)
            if use_fast_nms
            else TorchNMS.nms(boxes_for_nms, scores, iou_threshold)
        )
