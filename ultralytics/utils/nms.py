# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

import sys
import time
from typing import Dict, List

import torch

from ultralytics.utils import LOGGER
from ultralytics.utils.metrics import batch_probiou, box_iou, box_intersection
from ultralytics.utils.ops import xywh2xyxy


def _nms_multitask_scores(
    x: torch.Tensor, num_tasks: int, nms_multitask_conf: str, *, from_logits: bool
) -> torch.Tensor:
    """NMS ranking score from multitask row [box, conf0, cls0, conf1, cls1, ...]."""
    conf0 = x[:, 4]
    if num_tasks <= 1 or nms_multitask_conf == "task0":
        return conf0
    cols = [x[:, 4 + 2 * t] for t in range(num_tasks)]
    if from_logits:
        cols = [torch.sigmoid(c) for c in cols]
    task_s = torch.stack(cols, dim=1)
    if nms_multitask_conf == "min":
        return task_s.min(1)[0]
    if nms_multitask_conf == "prod":
        return task_s.prod(1)
    return conf0


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
    nms_strategy: str = "usual",  # "usual" (NMS), "nmm", "nmm_greedy"
    nms_multitask_conf: str = "task0",  # task0 | min | prod (multitask only)
):
    """Perform non-maximum suppression (NMS) on prediction results.

    Supports multitask models where nc is a list of class counts per task head.
    Output layout per detection: [x1, y1, x2, y2, conf_0, cls_0, conf_1, cls_1, ..., extra].

    Args:
        prediction (torch.Tensor): Predictions (batch_size, 4 + sum(nc) + extra, num_boxes) BCN format,
            or (batch_size, num_boxes, 4+2*num_tasks) for post-processed/end2end models.
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
        nms_strategy (str): "usual" for NMS, "nmm" or "nmm_greedy" for non-maximum merging (useful for SAHI).
        nms_multitask_conf (str): For ``len(nc) > 1``: ``task0`` — legacy (first-task logits for filter/NMS score);
            ``min`` / ``prod`` — combine per-task max-class sigmoids across heads (closer to joint confidence).

    Returns:
        list[torch.Tensor]: Detections per image with shape (N, 4 + 2*num_tasks + extra).
        keepi (list[torch.Tensor]): Indices of kept detections if return_idxs=True.
    """
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
    assert nms_multitask_conf in {"task0", "min", "prod"}, f"Invalid nms_multitask_conf={nms_multitask_conf}"

    if isinstance(prediction, (list, tuple)):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output
    if classes is not None:
        classes = torch.tensor(classes, device=prediction.device)

    num_tasks = len(nc)
    total_nc = sum(nc)

    # Post-processed format: (batch, N, 4+2*num_tasks) with [x1, y1, x2, y2, conf0, cls0, ...]
    # Already xyxy — must NOT go through BCN path (xywh2xyxy would corrupt coordinates).
    # Disambiguate from BCN (batch, channels, anchors): in postprocessed, last dim (cols) < dim 1 (N);
    # in BCN, last dim (anchors) > dim 1 (channels).
    n_cols = prediction.shape[-1]
    # BCN: (batch, 4+total_nc, anchors) — never treat as end2end-compact (even if model.end2end).
    ch_bcn = 4 + total_nc
    is_bcn = (
        prediction.dim() == 3
        and prediction.shape[1] == ch_bcn
        and prediction.shape[2] > prediction.shape[1]
    )
    is_postprocessed = (not is_bcn) and (
        ((n_cols == 4 + 2 * num_tasks) and n_cols < prediction.shape[1]) or end2end
    )

    if is_postprocessed:
        # Layout is [xyxy, conf0, cls0, ...] with width 4 + 2*k; k may differ from len(nc)
        # when end2end=True (often k=1, shape [N,6]) while nc is still multitask.
        extra = n_cols - 4
        if extra >= 2 and extra % 2 == 0:
            post_num_tasks = extra // 2
        else:
            post_num_tasks = 1
        nc_post = nc[:post_num_tasks] if len(nc) >= post_num_tasks else nc

        output = []
        for pred in prediction:
            if post_num_tasks > 1 and nms_multitask_conf != "task0":
                task_confs = torch.stack([pred[:, 4 + 2 * t] for t in range(post_num_tasks)], dim=1)
                comb = task_confs.min(1)[0] if nms_multitask_conf == "min" else task_confs.prod(1)
                pred = pred[comb > conf_thres]
            else:
                pred = pred[pred[:, 4] > conf_thres]
            if classes is not None:
                pred = pred[(pred[:, 5:6] == classes).any(1)]
            if len(pred) == 0:
                output.append(pred[:0])
                continue

            boxes = pred[:, :4]
            scores = _nms_multitask_scores(pred, post_num_tasks, nms_multitask_conf, from_logits=False)

            # Class offset: unique per combination of all task classes
            if agnostic:
                c = torch.zeros(len(pred), 1, device=pred.device)
            elif post_num_tasks > 1:
                unique_id = pred[:, 5].view(-1, 1).clone()
                multiplier = nc_post[0]
                for t in range(1, post_num_tasks):
                    task_cls_col = pred[:, 5 + 2 * t]
                    unique_id = unique_id + task_cls_col.view(-1, 1) * multiplier
                    multiplier *= nc_post[t]
                c = unique_id * max_wh
            else:
                c = pred[:, 5].view(-1, 1) * max_wh

            if nms_strategy in ("nmm", "nmm_greedy"):
                nmm_input = torch.cat(
                    [boxes + c, scores.view(-1, 1), pred[:, 5].view(-1, 1)], dim=1
                )
                keep_to_merge = _nmm_core(
                    nmm_input, match_metric="IOU", match_threshold=iou_thres,
                    greedy=(nms_strategy == "nmm_greedy"),
                )
                result = _merge_from_map(pred, keep_to_merge, max_det=max_det)
            else:
                boxes_offset = boxes + c
                if "torchvision" in sys.modules:
                    import torchvision
                    i = torchvision.ops.nms(boxes_offset.float(), scores.float(), iou_thres)
                else:
                    i = TorchNMS.nms(boxes_offset, scores, iou_thres)
                i = i[:max_det]
                result = pred[i]
            output.append(result)
        return output

    # Standard NMS path: input is (batch, channels, anchors) BCN format
    bs = prediction.shape[0]
    total_nc = total_nc or (prediction.shape[1] - 4)
    extra = prediction.shape[1] - total_nc - 4
    mi = 4 + total_nc

    # Candidate prefilter: first-task logits (legacy) or joint sigmoid over tasks
    if num_tasks > 1 and nms_multitask_conf != "task0":
        offset = 0
        parts = []
        for nc_i in nc:
            logits = prediction[:, 4 + offset : 4 + offset + nc_i, :]
            parts.append(logits.sigmoid().amax(1))
            offset += nc_i
        stacked = torch.stack(parts, dim=1)
        xc = (
            stacked.min(1)[0] > conf_thres
            if nms_multitask_conf == "min"
            else stacked.prod(1) > conf_thres
        )
    else:
        xc = prediction[:, 4 : 4 + nc[0]].amax(1) > conf_thres
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

            if num_tasks > 1 and nms_multitask_conf != "task0":
                S = torch.cat([torch.sigmoid(c) for c in confs], dim=1)
                conf_mask = (
                    S.min(1)[0] > conf_thres
                    if nms_multitask_conf == "min"
                    else S.prod(1) > conf_thres
                )
            else:
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
        scores = _nms_multitask_scores(x, num_tasks, nms_multitask_conf, from_logits=True)

        # Filter by class
        if classes is not None:
            filt = (j.view(-1, 1) == classes.float()).any(1)
            x = x[filt]
            if return_idxs:
                xk = xk[filt]
            conf, j = x[:, 4], x[:, 5]
            scores = _nms_multitask_scores(x, num_tasks, nms_multitask_conf, from_logits=True)

        # Check shape
        n = x.shape[0]
        if not n:
            continue
        if n > max_nms:
            filt = scores.argsort(descending=True)[:max_nms]
            x = x[filt]
            if return_idxs:
                xk = xk[filt]
            conf, j = x[:, 4], x[:, 5]
            scores = _nms_multitask_scores(x, num_tasks, nms_multitask_conf, from_logits=True)

        # NMS class offsets
        if agnostic:
            c = torch.zeros_like(j.view(-1, 1))
        elif num_tasks > 1:
            unique_id = x[:, 5].view(-1, 1)
            multiplier = nc[0]
            for t in range(1, num_tasks):
                task_cls_col = x[:, 5 + 2 * t]
                unique_id = unique_id + task_cls_col.view(-1, 1) * multiplier
                multiplier *= nc[t]
            c = unique_id * max_wh
        else:
            c = j.view(-1, 1) * max_wh

        if rotated:
            boxes = torch.cat((x[:, :2] + c, x[:, 2:4], x[:, -1:]), dim=-1)
            i = TorchNMS.fast_nms(boxes, scores, iou_thres, iou_func=batch_probiou)
            i = i[:max_det]
            output[xi] = x[i]
            if return_idxs:
                keepi[xi] = xk[i].view(-1)
        elif nms_strategy in ("nmm", "nmm_greedy"):
            boxes_for_nmm = x[:, :4] + c
            nmm_input = torch.cat([boxes_for_nmm, scores.view(-1, 1), j.view(-1, 1)], dim=1)
            keep_to_merge = _nmm_core(
                nmm_input, match_metric="IOU", match_threshold=iou_thres,
                greedy=(nms_strategy == "nmm_greedy"),
            )
            merged = _merge_from_map(x, keep_to_merge, max_det=max_det)
            if return_idxs and merged.numel():
                keepi[xi] = torch.arange(len(merged), device=x.device)
            output[xi] = merged
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

def _nmm_core(
    predictions: torch.Tensor,
    match_metric: str,
    match_threshold: float,
    greedy: bool,
) -> Dict[int, List[int]]:
    """
    Core NMM logic using fastquadtree for spatial indexing.

    Processes boxes in descending confidence order. Each box either becomes a "keep"
    (representative) or gets merged into an existing keep. Once merged, a box is
    invisible to the rest of the algorithm (no transitive merging).
    """
    from fastquadtree import RectQuadTree

    x1 = predictions[:, 0]
    y1 = predictions[:, 1]
    x2 = predictions[:, 2]
    y2 = predictions[:, 3]
    scores = predictions[:, 4]
    xmin = torch.minimum(x1, x2)
    xmax = torch.maximum(x1, x2)
    ymin = torch.minimum(y1, y2)
    ymax = torch.maximum(y1, y2)
    areas = (xmax - xmin) * (ymax - ymin)
    n = len(predictions)

    if n == 0:
        return {}

    min_coord = min(xmin.min().item(), ymin.min().item()) - 1
    max_coord = max(xmax.max().item(), ymax.max().item()) + 1
    bounds = (min_coord, min_coord, max_coord, max_coord)

    tree = RectQuadTree(bounds, capacity=10)
    for i in range(n):
        tree.insert((xmin[i].item(), ymin[i].item(), xmax[i].item(), ymax[i].item()), id_=i)

    sorted_idxs = torch.argsort(scores, descending=True).tolist()
    keep_to_merge_list: Dict[int, List[int]] = {}
    merge_to_keep: Dict[int, int] = {}
    suppressed: set = set()  # only used in greedy mode

    for current_idx in sorted_idxs:
        if current_idx in merge_to_keep:
            continue
        if greedy and current_idx in suppressed:
            continue

        cx1 = xmin[current_idx].item()
        cy1 = ymin[current_idx].item()
        cx2 = xmax[current_idx].item()
        cy2 = ymax[current_idx].item()
        current_area = areas[current_idx].item()
        query_rect = (cx1, cy1, cx2, cy2)

        results = tree.query(query_rect)
        matched_box_indices = []
        found_suppressed_keeps: set = set()

        for (candidate_idx, qx1, qy1, qx2, qy2) in results:
            if candidate_idx == current_idx:
                continue
            if candidate_idx in merge_to_keep:
                if greedy:
                    keep_idx = merge_to_keep[candidate_idx]
                    found_suppressed_keeps.add(keep_idx)
                continue
            if scores[candidate_idx] > scores[current_idx]:
                continue
            if scores[candidate_idx] == scores[current_idx]:
                candidate_coords = (xmin[candidate_idx].item(), ymin[candidate_idx].item(),
                                    xmax[candidate_idx].item(), ymax[candidate_idx].item())
                if candidate_coords > query_rect:
                    continue

            box_a = torch.tensor([[cx1, cy1, cx2, cy2]], dtype=torch.float32, device=predictions.device)
            box_b = torch.tensor([[qx1, qy1, qx2, qy2]], dtype=torch.float32, device=predictions.device)
            intersection = box_intersection(box_a, box_b).item()
            if match_metric == "IOU":
                union = current_area + areas[candidate_idx].item() - intersection
                metric = intersection / union if union > 0 else 0
            elif match_metric == "IOS":
                smaller = min(current_area, areas[candidate_idx].item())
                metric = intersection / smaller if smaller > 0 else 0
            else:
                raise ValueError(f"Invalid match_metric: {match_metric}")

            if metric >= match_threshold:
                matched_box_indices.append(candidate_idx)
                if greedy:
                    suppressed.add(candidate_idx)
                    merge_to_keep[candidate_idx] = int(current_idx)

        current_idx_native = int(current_idx)
        if greedy:
            if found_suppressed_keeps:
                keep_idx = max(found_suppressed_keeps, key=lambda k: scores[k].item())
                if keep_idx not in keep_to_merge_list:
                    keep_to_merge_list[keep_idx] = []
                keep_to_merge_list[keep_idx].append(current_idx_native)
                for m in matched_box_indices:
                    m_native = int(m)
                    merge_to_keep[m_native] = keep_idx
                    if m_native not in keep_to_merge_list[keep_idx]:
                        keep_to_merge_list[keep_idx].append(m_native)
                    suppressed.add(m_native)
                suppressed.add(current_idx)
                merge_to_keep[current_idx_native] = keep_idx
            else:
                keep_to_merge_list[current_idx_native] = [int(idx) for idx in matched_box_indices]
        else:
            keep_to_merge_list[current_idx_native] = []
            for matched_box_idx in matched_box_indices:
                matched_box_idx_native = int(matched_box_idx)
                if matched_box_idx_native not in merge_to_keep:
                    keep_to_merge_list[current_idx_native].append(matched_box_idx_native)
                    merge_to_keep[matched_box_idx_native] = current_idx_native

    return keep_to_merge_list


def _merge_from_map(x, keep_to_merge, max_det=300):
    """Merge overlapping boxes: keep the highest-confidence detection (coordinates + class)."""
    merged_indices = set()
    merged_boxes = []

    for keep_idx, merge_list in keep_to_merge.items():
        if keep_idx in merged_indices:
            continue

        merged_box = x[keep_idx].clone()
        merged_indices.add(keep_idx)

        if merge_list:
            for m in merge_list:
                merged_indices.add(m)

        merged_boxes.append(merged_box)

    for idx in range(len(x)):
        if idx not in merged_indices:
            merged_boxes.append(x[idx])

    return torch.stack(merged_boxes)[:max_det] if merged_boxes else x[:0]


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
