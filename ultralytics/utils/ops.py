# Ultralytics YOLO 🚀, AGPL-3.0 license

import contextlib
import math
import re
import time
from typing import List

import cv2
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from shapely import STRtree, box
from typing import Dict, List

from ultralytics.utils import LOGGER
from ultralytics.utils.metrics import batch_probiou, box_iou
from ultralytics.utils.tf import (
    xyxy2xywh,
    xywh2xyxy,
    clip_boxes,
    clip_coords,
)


class Profile(contextlib.ContextDecorator):
    """
    YOLOv8 Profile class. Use as a decorator with @Profile() or as a context manager with 'with Profile():'.

    Example:
        ```python
        from ultralytics.utils.ops import Profile

        with Profile(device=device) as dt:
            pass  # slow operation here

        print(dt)  # prints "Elapsed time is 9.5367431640625e-07 s"
        ```
    """

    def __init__(self, t=0.0, device: torch.device = None):
        """
        Initialize the Profile class.

        Args:
            t (float): Initial time. Defaults to 0.0.
            device (torch.device): Devices used for model inference. Defaults to None (cpu).
        """
        self.t = t
        self.device = device
        self.cuda = bool(device and str(device).startswith("cuda"))

    def __enter__(self):
        """Start timing."""
        self.start = self.time()
        return self

    def __exit__(self, type, value, traceback):  # noqa
        """Stop timing."""
        self.dt = self.time() - self.start  # delta-time
        self.t += self.dt  # accumulate dt

    def __str__(self):
        """Returns a human-readable string representing the accumulated elapsed time in the profiler."""
        return f"Elapsed time is {self.t} s"

    def time(self):
        """Get current time."""
        if self.cuda:
            torch.cuda.synchronize(self.device)
        return time.time()


def segment2box(segment, width=640, height=640):
    """
    Convert 1 segment label to 1 box label, applying inside-image constraint, i.e. (xy1, xy2, ...) to (xyxy).

    Args:
        segment (torch.Tensor): the segment label
        width (int): the width of the image. Defaults to 640
        height (int): The height of the image. Defaults to 640

    Returns:
        (np.ndarray): the minimum and maximum x and y values of the segment.
    """
    x, y = segment.T  # segment xy
    inside = (x >= 0) & (y >= 0) & (x <= width) & (y <= height)
    x = x[inside]
    y = y[inside]
    return (
        np.array([x.min(), y.min(), x.max(), y.max()], dtype=segment.dtype)
        if any(x)
        else np.zeros(4, dtype=segment.dtype)
    )  # xyxy


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None, padding=True, xywh=False):
    """
    Rescales bounding boxes (in the format of xyxy by default) from the shape of the image they were originally
    specified in (img1_shape) to the shape of a different image (img0_shape).

    Args:
        img1_shape (tuple): The shape of the image that the bounding boxes are for, in the format of (height, width).
        boxes (torch.Tensor): the bounding boxes of the objects in the image, in the format of (x1, y1, x2, y2)
        img0_shape (tuple): the shape of the target image, in the format of (height, width).
        ratio_pad (tuple): a tuple of (ratio, pad) for scaling the boxes. If not provided, the ratio and pad will be
            calculated based on the size difference between the two images.
        padding (bool): If True, assuming the boxes is based on image augmented by yolo style. If False then do regular
            rescaling.
        xywh (bool): The box format is xywh or not, default=False.

    Returns:
        boxes (torch.Tensor): The scaled bounding boxes, in the format of (x1, y1, x2, y2)
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (
            round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1),
            round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1),
        )  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    if padding:
        boxes[..., 0] -= pad[0]  # x padding
        boxes[..., 1] -= pad[1]  # y padding
        if not xywh:
            boxes[..., 2] -= pad[0]  # x padding
            boxes[..., 3] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    return clip_boxes(boxes, img0_shape)


def make_divisible(x, divisor):
    """
    Returns the nearest number that is divisible by the given divisor.

    Args:
        x (int): The number to make divisible.
        divisor (int | torch.Tensor): The divisor.

    Returns:
        (int): The nearest number divisible by the divisor.
    """
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor


def nms_rotated(boxes, scores, threshold=0.45):
    """
    NMS for oriented bounding boxes using probiou and fast-nms.

    Args:
        boxes (torch.Tensor): Rotated bounding boxes, shape (N, 5), format xywhr.
        scores (torch.Tensor): Confidence scores, shape (N,).
        threshold (float, optional): IoU threshold. Defaults to 0.45.

    Returns:
        (torch.Tensor): Indices of boxes to keep after NMS.
    """
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.int8)
    sorted_idx = torch.argsort(scores, descending=True)
    boxes = boxes[sorted_idx]
    ious = batch_probiou(boxes, boxes).triu_(diagonal=1)
    pick = torch.nonzero(ious.max(dim=0)[0] < threshold).squeeze_(-1)
    return sorted_idx[pick]


def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
    nc=[0],  # number of classes (optional)
    max_time_img=0.05,
    max_nms=30000,
    max_wh=7680,
    in_place=True,
    rotated=False,
    force_nms_for_e2e=False,  # Force NMS for end-to-end models (e.g., for SAHI final deduplication)
):
    """
    Perform non-maximum suppression (NMS) on a set of boxes, with support for masks and multiple labels per box.

    Args:
        prediction (torch.Tensor): A tensor of shape (batch_size, num_classes + 4 + num_masks/kpts, num_boxes)
            containing the predicted boxes, classes, and masks. The tensor should be in the format
            output by a model, such as YOLO.
        conf_thres (float): The confidence threshold below which boxes will be filtered out.
            Valid values are between 0.0 and 1.0.
        iou_thres (float): The IoU threshold below which boxes will be filtered out during NMS.
            Valid values are between 0.0 and 1.0.
        classes (List[int]): A list of class indices to consider.
            If None, all classes will be considered.
        agnostic (bool): If True, the model is agnostic to the number of classes, and all
            classes will be considered as one.
        multi_label (bool): If True, each box may have multiple labels.
        labels (List[List[Union[int, float, torch.Tensor]]]): A list of lists, where each inner
            list contains the apriori labels for a given image. The list should be in the format
            output by a dataloader, with each label being a tuple of (class_index, x1, y1, x2, y2).
        max_det (int): The maximum number of boxes to keep after NMS.
        nc (List[int], optional): The number of classes output by the model. Any indices after this will be considered masks.
        max_time_img (float): The maximum time (seconds) for processing one image.
        max_nms (int): The maximum number of boxes into torchvision.ops.nms().
        max_wh (int): The maximum box width and height in pixels.
        in_place (bool): If True, the input prediction tensor will be modified in place.
        rotated (bool): If Oriented Bounding Boxes (OBB) are being passed for NMS.

    Returns:
        (List[torch.Tensor]): A list of length batch_size, where each element is a tensor of
            shape (num_boxes, 6 + num_masks) containing the kept boxes, with columns
            (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).
    """
    import torchvision  # scope for faster 'import ultralytics'

    # Checks
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
    if isinstance(prediction, (list, tuple)):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output
    if classes is not None:
        classes = torch.tensor(classes, device=prediction.device)
        
    if force_nms_for_e2e:
        import torchvision

        boxes = prediction[:, :4]
        scores = prediction[:, 4]
        
        # Extract classes from all heads
        # Format: [x1, y1, x2, y2, conf0, cls0, conf1, cls1, conf2, cls2, ...]
        # For head i: cls is at index 4 + 2*i + 1 = 5 + 2*i
        clss = []
        for head_idx in range(len(nc)):
            cls_idx = 5 + 2 * head_idx
            clss.append(prediction[:, cls_idx])

        if agnostic:
            c = torch.zeros_like(clss[0].view(-1, 1))  # No offset for agnostic NMS
        else:
            if len(nc) > 1:  # Мультитаск
                unique_id = clss[0].view(-1, 1)
                multiplier = nc[0]
                for head_idx in range(1, len(nc)):
                    attr_class = clss[head_idx].view(-1, 1)
                    unique_id = unique_id + attr_class * multiplier
                    multiplier *= nc[head_idx]
                c = unique_id * max_wh
            else:  # Одна голова - оригинальная реализация
                c = clss[0].view(-1, 1) * max_wh

        boxes_offset = boxes + c 
        keep = torchvision.ops.nms(boxes_offset, scores, iou_thres)

        return prediction[keep]  

    if prediction.shape[-1] == 6 or prediction.shape[-2] == max_det:  # end-to-end model (BNC, i.e. 1,300,6)
        output = [pred[pred[:, 4] > conf_thres] for pred in prediction]
        if classes is not None:
            output = [pred[(pred[:, 5:6] == classes).any(1)] for pred in output]
        # nms for yolov10
        # output = [
        #     pred[torchvision.ops.nms(pred[:, :4], pred[:, 4], iou_thres)]
        #     for pred in output
        # ]  
        return output

    bs = prediction.shape[0]  # batch size (BCN, i.e. 1,84,6300)
    nm = prediction.shape[1] - 4 - sum(nc)

    # candidate boxes determined by first head confidence only
    first_nc = nc[0]
    xc = prediction[:, 4 : 4 + first_nc].amax(1) > conf_thres

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    time_limit = 2.0 + max_time_img * bs  # seconds to quit after

    prediction = prediction.transpose(-1, -2)  # shape(1,84,6300) to shape(1,6300,84)
    if not rotated:
        if in_place:
            prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # xywh to xyxy
        else:
            prediction = torch.cat((xywh2xyxy(prediction[..., :4]), prediction[..., 4:]), dim=-1)  # xywh to xyxy

    t = time.time()
    # each head contributes two columns: conf and class
    output = [torch.zeros((0, 4 + 2 * len(nc) + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]) and not rotated:
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 4), device=x.device)
            v[:, :4] = xywh2xyxy(lb[:, 1:5])  # box
            v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx6+ (xyxy, conf, cls, cls2, mask...)
        start = 4  # index of first conf column after the box coordinates
        box = x[:, :4]
        confs, clss = [], []
        for nc_i in nc:
            cls_slice = x[:, start : start + nc_i]
            conf_i, j_i = cls_slice.max(1, keepdim=True)
            confs.append(conf_i)
            clss.append(j_i.float())
            start += nc_i
        mask = x[:, start:]

        conf_mask = confs[0].view(-1) > conf_thres
        box = box[conf_mask]
        mask = mask[conf_mask]
        confs = [c[conf_mask] for c in confs]
        clss = [j_[conf_mask] for j_ in clss]

        # final layout becomes [box, conf0,cls0, conf1,cls1, ..., mask]
        x = torch.cat([box] + sum([[c, j] for c, j in zip(confs, clss)], []) + [mask], 1)
        conf, j = confs[0].view(-1), clss[0].view(-1)

        # Filter by class
        if classes is not None:
            x = x[(j.view(-1, 1) == classes).any(1)]
            conf = x[:, 4]
            j = x[:, 5]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        if n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes
            conf = x[:, 4]
            j = x[:, 5]

        if agnostic:
            c = torch.zeros_like(j.view(-1, 1))  # No offset for agnostic NMS
        else:
            if len(nc) > 1:  # Мультитаск
                unique_id = j.view(-1, 1)
                multiplier = nc[0]
                for head_idx in range(1, len(nc)):
                    attr_class = clss[head_idx].view(-1, 1)
                    unique_id = unique_id + attr_class * multiplier
                    multiplier *= nc[head_idx]
                c = unique_id * max_wh
            else:  # Одна голова - оригинальная реализация
                c = j.view(-1, 1) * max_wh
        scores = conf

        if rotated:
            boxes = torch.cat((x[:, :2] + c, x[:, 2:4], x[:, -1:]), dim=-1)  # xywhr
            i = nms_rotated(boxes, scores, iou_thres)
        else:
            boxes = x[:, :4] + c  # boxes (offset by class)
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections

        # # Experimental
        # merge = False  # use merge-NMS
        # if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
        #     # Update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
        #     from .metrics import box_iou
        #     iou = box_iou(boxes[i], boxes) > iou_thres  # IoU matrix
        #     weights = iou * scores[None]  # box weights
        #     x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
        #     redundant = True  # require redundant detections
        #     if redundant:
        #         i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            LOGGER.warning(f"WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded")
            break  # time limit exceeded

    return output

def nmm(
    predictions: torch.Tensor,
    match_metric: str = "IOU",
    match_threshold: float = 0.5,
) -> Dict[int, List[int]]:
    """
    Non-maximum merging for axis-aligned bounding boxes using STRTree.

    Args:
        predictions (torch.Tensor): Tensor of shape [num_boxes, 6] with format [x1, y1, x2, y2, score, class_id].
        match_metric (str): "IOU" or "IOS".
        match_threshold (float): The overlap threshold for match metric.

    Returns:
        (Dict[int, List[int]]): Mapping from prediction indices to keep to a list of prediction indices to be merged.
    """
    # Extract coordinates and scores as tensors
    x1 = predictions[:, 0]
    y1 = predictions[:, 1]
    x2 = predictions[:, 2]
    y2 = predictions[:, 3]
    scores = predictions[:, 4]

    # Calculate areas as tensor (vectorized operation)
    areas = (x2 - x1) * (y2 - y1)

    # Create Shapely boxes
    boxes = []
    for i in range(len(predictions)):
        boxes.append(
            box(
                x1[i].item(),
                y1[i].item(),
                x2[i].item(),
                y2[i].item(),
            )
        )

    # Sort indices by score (descending) using torch
    sorted_idxs = torch.argsort(scores, descending=True).tolist()

    # Build STRtree
    tree = STRtree(boxes)

    keep_to_merge_list = {}
    merge_to_keep = {}

    for current_idx in sorted_idxs:
        current_box = boxes[current_idx]
        current_area = areas[current_idx].item()

        # Query potential intersections using STRtree
        candidate_idxs = tree.query(current_box)

        matched_box_indices = []
        for candidate_idx in candidate_idxs:
            if candidate_idx == current_idx:
                continue

            # Only consider candidates with lower or equal score
            if scores[candidate_idx] > scores[current_idx]:
                continue

            # For equal scores, use deterministic tie-breaking based on box coordinates
            if scores[candidate_idx] == scores[current_idx]:
                current_coords = (
                    x1[current_idx].item(),
                    y1[current_idx].item(),
                    x2[current_idx].item(),
                    y2[current_idx].item(),
                )
                candidate_coords = (
                    x1[candidate_idx].item(),
                    y1[candidate_idx].item(),
                    x2[candidate_idx].item(),
                    y2[candidate_idx].item(),
                )

                # Compare coordinates lexicographically
                if candidate_coords > current_coords:
                    continue

            # Calculate intersection area
            candidate_box = boxes[candidate_idx]
            intersection = current_box.intersection(candidate_box).area

            # Calculate metric
            if match_metric == "IOU":
                union = current_area + areas[candidate_idx].item() - intersection
                metric = intersection / union if union > 0 else 0
            elif match_metric == "IOS":
                smaller = min(current_area, areas[candidate_idx].item())
                metric = intersection / smaller if smaller > 0 else 0
            else:
                raise ValueError(f"Invalid match_metric: {match_metric}")

            # Add to matched list if overlap exceeds threshold
            if metric >= match_threshold:
                matched_box_indices.append(candidate_idx)

        # Convert current_idx to native Python int
        current_idx_native = int(current_idx)

        # Create keep_ind to merge_ind_list mapping
        if current_idx_native not in merge_to_keep:
            keep_to_merge_list[current_idx_native] = []

            for matched_box_idx in matched_box_indices:
                matched_box_idx_native = int(matched_box_idx)
                if matched_box_idx_native not in merge_to_keep:
                    keep_to_merge_list[current_idx_native].append(matched_box_idx_native)
                    merge_to_keep[matched_box_idx_native] = current_idx_native
        else:
            keep_idx = merge_to_keep[current_idx_native]
            for matched_box_idx in matched_box_indices:
                matched_box_idx_native = int(matched_box_idx)
                if (
                    matched_box_idx_native not in keep_to_merge_list.get(keep_idx, [])
                    and matched_box_idx_native not in merge_to_keep
                ):
                    if keep_idx not in keep_to_merge_list:
                        keep_to_merge_list[keep_idx] = []
                    keep_to_merge_list[keep_idx].append(matched_box_idx_native)
                    merge_to_keep[matched_box_idx_native] = keep_idx

    return keep_to_merge_list

def greedy_nmm(
    predictions: torch.Tensor,
    match_metric: str = "IOU",
    match_threshold: float = 0.5,
) -> Dict[int, List[int]]:
    """
    Greedy non-maximum merging for axis-aligned bounding boxes using STRTree.

    Args:
        predictions (torch.Tensor): Tensor of shape [num_boxes, 6] with format [x1, y1, x2, y2, score, class_id].
        match_metric (str): "IOU" or "IOS".
        match_threshold (float): The overlap threshold for match metric.

    Returns:
        (Dict[int, List[int]]): Mapping from prediction indices to keep to a list of prediction indices to be merged.
    """
    # Extract coordinates and scores as tensors
    x1 = predictions[:, 0]
    y1 = predictions[:, 1]
    x2 = predictions[:, 2]
    y2 = predictions[:, 3]
    scores = predictions[:, 4]

    # Calculate areas as tensor (vectorized operation)
    areas = (x2 - x1) * (y2 - y1)

    # Create Shapely boxes
    boxes = []
    for i in range(len(predictions)):
        boxes.append(
            box(
                x1[i].item(),
                y1[i].item(),
                x2[i].item(),
                y2[i].item(),
            )
        )

    # Sort indices by score (descending) using torch
    sorted_idxs = torch.argsort(scores, descending=True).tolist()

    # Build STRtree
    tree = STRtree(boxes)

    keep_to_merge_list = {}
    suppressed = set()

    for current_idx in sorted_idxs:
        if current_idx in suppressed:
            continue

        current_box = boxes[current_idx]
        current_area = areas[current_idx].item()

        # Query potential intersections using STRtree
        candidate_idxs = tree.query(current_box)

        merge_list = []
        for candidate_idx in candidate_idxs:
            if candidate_idx == current_idx or candidate_idx in suppressed:
                continue

            # Only consider candidates with lower or equal score
            if scores[candidate_idx] > scores[current_idx]:
                continue

            # For equal scores, use deterministic tie-breaking based on box coordinates
            if scores[candidate_idx] == scores[current_idx]:
                current_coords = (
                    x1[current_idx].item(),
                    y1[current_idx].item(),
                    x2[current_idx].item(),
                    y2[current_idx].item(),
                )
                candidate_coords = (
                    x1[candidate_idx].item(),
                    y1[candidate_idx].item(),
                    x2[candidate_idx].item(),
                    y2[candidate_idx].item(),
                )

                # Compare coordinates lexicographically
                if candidate_coords > current_coords:
                    continue

            # Calculate intersection area
            candidate_box = boxes[candidate_idx]
            intersection = current_box.intersection(candidate_box).area

            # Calculate metric
            if match_metric == "IOU":
                union = current_area + areas[candidate_idx].item() - intersection
                metric = intersection / union if union > 0 else 0
            elif match_metric == "IOS":
                smaller = min(current_area, areas[candidate_idx].item())
                metric = intersection / smaller if smaller > 0 else 0
            else:
                raise ValueError(f"Invalid match_metric: {match_metric}")

            # Add to merge list if overlap exceeds threshold
            if metric >= match_threshold:
                merge_list.append(candidate_idx)
                suppressed.add(candidate_idx)

        keep_to_merge_list[int(current_idx)] = [int(idx) for idx in merge_list]

    return keep_to_merge_list

def _merge_from_map(x, keep_to_merge, max_det=300):
    """Merge boxes according to keep_to_merge mapping."""
    merged_indices = set()
    merged_boxes = []

    for keep_idx, merge_list in keep_to_merge.items():
        if keep_idx in merged_indices:
            continue

        merged_box = x[keep_idx].clone()
        merged_indices.add(keep_idx)

        if merge_list:
            all_boxes = [x[keep_idx]]
            all_weights = [x[keep_idx, 4].item()]

            for merge_idx in merge_list:
                if merge_idx not in merged_indices:
                    all_boxes.append(x[merge_idx])
                    all_weights.append(x[merge_idx, 4].item())
                    merged_indices.add(merge_idx)

            if len(all_boxes) > 1:
                stacked = torch.stack(all_boxes)
                weights = torch.tensor(all_weights, device=x.device, dtype=x.dtype)
                weights = weights / weights.sum()

                merged_box[:2] = stacked[:, :2].min(dim=0)[0]  # x1, y1
                merged_box[2:4] = stacked[:, 2:4].max(dim=0)[0]  # x2, y2
                merged_box[4] = stacked[:, 4].max()  # keep best confidence
                max_conf_idx = stacked[:, 4].argmax()
                merged_box[5] = stacked[max_conf_idx, 5]  # class from best

        merged_boxes.append(merged_box)

    # Add boxes that weren't merged
    for idx in range(len(x)):
        if idx not in merged_indices:
            merged_boxes.append(x[idx])

    return torch.stack(merged_boxes)[:max_det] if merged_boxes else x[:0]

def non_max_merging(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
    nc=[0],  # number of classes (optional)
    max_time_img=0.05,
    max_nms=30000,
    max_wh=7680,
    in_place=True,
    rotated=False,
    force_nmm_for_e2e=False,  # apply NMM to already-decoded boxes (shape [...,6])
    merge_mode="greedy",  # "greedy" or "full"
):
    """
    Perform non-maximum merging (NMM) on a set of boxes - similar to NMS but merges overlapping boxes instead of suppressing them.

    Args:
        prediction (torch.Tensor): A tensor of shape (batch_size, num_classes + 4 + num_masks/kpts, num_boxes)
            containing the predicted boxes, classes, and masks.
        conf_thres (float): The confidence threshold below which boxes will be filtered out.
        iou_thres (float): The IoU threshold below which boxes will be filtered out during NMM.
        classes (List[int]): A list of class indices to consider. If None, all classes will be considered.
        agnostic (bool): If True, the model is agnostic to the number of classes.
        multi_label (bool): If True, each box may have multiple labels.
        labels (List[List[Union[int, float, torch.Tensor]]]): Apriori labels for given image.
        max_det (int): The maximum number of boxes to keep after NMM.
        nc (List[int], optional): The number of classes output by the model.
        max_time_img (float): The maximum time (seconds) for processing one image.
        max_nms (int): The maximum number of boxes into merging algorithm.
        max_wh (int): The maximum box width and height in pixels.
        in_place (bool): If True, the input prediction tensor will be modified in place.
        rotated (bool): If Oriented Bounding Boxes (OBB) are being passed for NMM.
        merge_mode (str): "greedy" for greedy NMM or "full" for full NMM.

    Returns:
        (List[torch.Tensor]): A list of length batch_size, where each element is a tensor of
            shape (num_boxes, 6 + num_masks) containing the kept/merged boxes.
    """
    import torchvision  # scope for faster 'import ultralytics'


    # Checks
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]
    if classes is not None:
        classes = torch.tensor(classes, device=prediction.device)

    if force_nmm_for_e2e:
        import torchvision

        boxes = prediction[:, :4]
        scores = prediction[:, 4]
        
        # Extract classes from all heads
        # Format: [x1, y1, x2, y2, conf0, cls0, conf1, cls1, conf2, cls2, ...]
        # For head i: cls is at index 4 + 2*i + 1 = 5 + 2*i
        clss = []
        for head_idx in range(len(nc)):
            cls_idx = 5 + 2 * head_idx
            clss.append(prediction[:, cls_idx])

        if agnostic:
            c = torch.zeros_like(clss[0].view(-1, 1))  # No offset for agnostic NMS
        else:
            if len(nc) > 1:  # Мультитаск
                unique_id = clss[0].view(-1, 1)
                multiplier = nc[0]
                for head_idx in range(1, len(nc)):
                    attr_class = clss[head_idx].view(-1, 1)
                    unique_id = unique_id + attr_class * multiplier
                    multiplier *= nc[head_idx]
                c = unique_id * max_wh
            else:  # Одна голова - оригинальная реализация
                c = clss[0].view(-1, 1) * max_wh

        boxes_offset = boxes + c 

        # Create tensor for NMM: [x1, y1, x2, y2, score, class_id]
        # Use class from first head for NMM input
        nmm_input = torch.cat([boxes_offset, scores.view(-1, 1), clss[0].view(-1, 1)], dim=1)

        # Apply NMM - with offset, we can use single call for all boxes (like NMS)
        # Offset ensures boxes of different classes won't merge
        if merge_mode == "greedy":
            keep_to_merge = greedy_nmm(nmm_input, match_metric="IOU", match_threshold=iou_thres)
        else:  # full NMM
            keep_to_merge = nmm(nmm_input, match_metric="IOU", match_threshold=iou_thres)

        merged = _merge_from_map(prediction, keep_to_merge)

        return merged
    
    if prediction.shape[-1] == 6 or prediction.shape[-2] == max_det:  # end-to-end model (BNC, i.e. 1,300,6)
        output = [pred[pred[:, 4] > conf_thres] for pred in prediction]
        if classes is not None:
            output = [pred[(pred[:, 5:6] == classes).any(1)] for pred in output]
            # nms for yolov10
            # output = [
            #     pred[torchvision.ops.nms(pred[:, :4], pred[:, 4], iou_thres)]
            #     for pred in output
            # ]  
        return output

    bs = prediction.shape[0]
    nm = prediction.shape[1] - 4 - sum(nc)

    # candidate boxes determined by first head confidence only
    first_nc = nc[0]
    xc = prediction[:, 4 : 4 + first_nc].amax(1) > conf_thres

    # Settings
    time_limit = 2.0 + max_time_img * bs

    prediction = prediction.transpose(-1, -2)
    if not rotated:
        if in_place:
            prediction[..., :4] = xywh2xyxy(prediction[..., :4])
        else:
            prediction = torch.cat((xywh2xyxy(prediction[..., :4]), prediction[..., 4:]), dim=-1)

    t = time.time()
    output = [torch.zeros((0, 4 + 2 * len(nc) + nm), device=prediction.device)] * bs
    
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]) and not rotated:
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 4), device=x.device)
            v[:, :4] = xywh2xyxy(lb[:, 1:5])
            v[range(len(lb)), lb[:, 0].long() + 4] = 1.0
            x = torch.cat((x, v), 0)

        if not x.shape[0]:
            continue

        # Detections matrix
        start = 4
        box = x[:, :4]
        confs, clss = [], []
        for nc_i in nc:
            cls_slice = x[:, start : start + nc_i]
            conf_i, j_i = cls_slice.max(1, keepdim=True)
            confs.append(conf_i)
            clss.append(j_i.float())
            start += nc_i
        mask = x[:, start:]

        conf_mask = confs[0].view(-1) > conf_thres
        box = box[conf_mask]
        mask = mask[conf_mask]
        confs = [c[conf_mask] for c in confs]
        clss = [j_[conf_mask] for j_ in clss]

        x = torch.cat([box] + sum([[c, j] for c, j in zip(confs, clss)], []) + [mask], 1)
        conf, j = confs[0].view(-1), clss[0].view(-1)

        # Filter by class
        if classes is not None:
            x = x[(j.view(-1, 1) == classes).any(1)]
            conf = x[:, 4]
            j = x[:, 5]

        n = x.shape[0]
        if not n:
            continue
        if n > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]
            conf = x[:, 4]
            j = x[:, 5]

        # Prepare boxes for NMM (format: [x1, y1, x2, y2, score, class_id])
        # Use offset approach (same as NMS): offset separates boxes by class in space,
        # allowing single NMM call for all boxes instead of per-class batching
        if agnostic:
            c = torch.zeros_like(j.view(-1, 1))
        else:
            if len(nc) > 1:  # Multi-task
                unique_id = j.view(-1, 1)
                multiplier = nc[0]
                for head_idx in range(1, len(nc)):
                    attr_class = clss[head_idx].view(-1, 1)
                    unique_id = unique_id + attr_class * multiplier
                    multiplier *= nc[head_idx]
                c = unique_id * max_wh
            else:  # Single head
                c = j.view(-1, 1) * max_wh

        boxes_for_nmm = x[:, :4].clone()
        if not agnostic:
            boxes_for_nmm = boxes_for_nmm + c  # offset separates boxes by class in space
        
        # Create tensor for NMM: [x1, y1, x2, y2, score, class_id]
        nmm_input = torch.cat([boxes_for_nmm, conf.view(-1, 1), j.view(-1, 1)], dim=1)

        # Apply NMM - with offset, we can use single call for all boxes (like NMS)
        # Offset ensures boxes of different classes won't merge
        if merge_mode == "greedy":
            keep_to_merge = greedy_nmm(nmm_input, match_metric="IOU", match_threshold=iou_thres)
        else:  # full NMM
            keep_to_merge = nmm(nmm_input, match_metric="IOU", match_threshold=iou_thres)

        merged = _merge_from_map(x, keep_to_merge)
        if merged.numel():
            output[xi] = merged
        
        if (time.time() - t) > time_limit:
            LOGGER.warning(f"WARNING ⚠️ NMM time limit {time_limit:.3f}s exceeded")
            break

    return output


def scale_image(masks, im0_shape, ratio_pad=None):
    """
    Takes a mask, and resizes it to the original image size.

    Args:
        masks (np.ndarray): resized and padded masks/images, [h, w, num]/[h, w, 3].
        im0_shape (tuple): the original image shape
        ratio_pad (tuple): the ratio of the padding to the original image.

    Returns:
        masks (np.ndarray): The masks that are being returned with shape [h, w, num].
    """
    # Rescale coordinates (xyxy) from im1_shape to im0_shape
    im1_shape = masks.shape
    if im1_shape[:2] == im0_shape[:2]:
        return masks
    if ratio_pad is None:  # calculate from im0_shape
        gain = min(im1_shape[0] / im0_shape[0], im1_shape[1] / im0_shape[1])  # gain  = old / new
        pad = (im1_shape[1] - im0_shape[1] * gain) / 2, (im1_shape[0] - im0_shape[0] * gain) / 2  # wh padding
    else:
        # gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    top, left = int(pad[1]), int(pad[0])  # y, x
    bottom, right = int(im1_shape[0] - pad[1]), int(im1_shape[1] - pad[0])

    if len(masks.shape) < 2:
        raise ValueError(f'"len of masks shape" should be 2 or 3, but got {len(masks.shape)}')
    masks = masks[top:bottom, left:right]
    masks = cv2.resize(masks, (im0_shape[1], im0_shape[0]))
    if len(masks.shape) == 2:
        masks = masks[:, :, None]

    return masks

def segments2boxes(segments):
    """
    It converts segment labels to box labels, i.e. (cls, xy1, xy2, ...) to (cls, xywh).

    Args:
        segments (list): list of segments, each segment is a list of points, each point is a list of x, y coordinates

    Returns:
        (np.ndarray): the xywh coordinates of the bounding boxes.
    """
    boxes = []
    for s in segments:
        x, y = s.T  # segment xy
        boxes.append([x.min(), y.min(), x.max(), y.max()])  # cls, xyxy
    return xyxy2xywh(np.array(boxes))  # cls, xywh


def resample_segments(segments, n=1000):
    """
    Inputs a list of segments (n,2) and returns a list of segments (n,2) up-sampled to n points each.

    Args:
        segments (list): a list of (n,2) arrays, where n is the number of points in the segment.
        n (int): number of points to resample the segment to. Defaults to 1000

    Returns:
        segments (list): the resampled segments.
    """
    for i, s in enumerate(segments):
        s = np.concatenate((s, s[0:1, :]), axis=0)
        x = np.linspace(0, len(s) - 1, n)
        xp = np.arange(len(s))
        segments[i] = (
            np.concatenate([np.interp(x, xp, s[:, i]) for i in range(2)], dtype=np.float32).reshape(2, -1).T
        )  # segment xy
    return segments


def crop_mask(masks, boxes):
    """
    It takes a mask and a bounding box, and returns a mask that is cropped to the bounding box.

    Args:
        masks (torch.Tensor): [n, h, w] tensor of masks
        boxes (torch.Tensor): [n, 4] tensor of bbox coordinates in relative point form

    Returns:
        (torch.Tensor): The masks are being cropped to the bounding box.
    """
    _, h, w = masks.shape
    x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)  # x1 shape(n,1,1)
    r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]  # rows shape(1,1,w)
    c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]  # cols shape(1,h,1)

    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


def process_mask(protos, masks_in, bboxes, shape, upsample=False):
    """
    Apply masks to bounding boxes using the output of the mask head.

    Args:
        protos (torch.Tensor): A tensor of shape [mask_dim, mask_h, mask_w].
        masks_in (torch.Tensor): A tensor of shape [n, mask_dim], where n is the number of masks after NMS.
        bboxes (torch.Tensor): A tensor of shape [n, 4], where n is the number of masks after NMS.
        shape (tuple): A tuple of integers representing the size of the input image in the format (h, w).
        upsample (bool): A flag to indicate whether to upsample the mask to the original image size. Default is False.

    Returns:
        (torch.Tensor): A binary mask tensor of shape [n, h, w], where n is the number of masks after NMS, and h and w
            are the height and width of the input image. The mask is applied to the bounding boxes.
    """
    c, mh, mw = protos.shape  # CHW
    ih, iw = shape
    masks = (masks_in @ protos.float().view(c, -1)).view(-1, mh, mw)  # CHW
    width_ratio = mw / iw
    height_ratio = mh / ih

    downsampled_bboxes = bboxes.clone()
    downsampled_bboxes[:, 0] *= width_ratio
    downsampled_bboxes[:, 2] *= width_ratio
    downsampled_bboxes[:, 3] *= height_ratio
    downsampled_bboxes[:, 1] *= height_ratio

    masks = crop_mask(masks, downsampled_bboxes)  # CHW
    if upsample:
        masks = F.interpolate(masks[None], shape, mode="bilinear", align_corners=False)[0]  # CHW
    return masks.gt_(0.0)


def process_mask_native(protos, masks_in, bboxes, shape):
    """
    It takes the output of the mask head, and crops it after upsampling to the bounding boxes.

    Args:
        protos (torch.Tensor): [mask_dim, mask_h, mask_w]
        masks_in (torch.Tensor): [n, mask_dim], n is number of masks after nms
        bboxes (torch.Tensor): [n, 4], n is number of masks after nms
        shape (tuple): the size of the input image (h,w)

    Returns:
        masks (torch.Tensor): The returned masks with dimensions [h, w, n]
    """
    c, mh, mw = protos.shape  # CHW
    masks = (masks_in @ protos.float().view(c, -1)).view(-1, mh, mw)
    masks = scale_masks(masks[None], shape)[0]  # CHW
    masks = crop_mask(masks, bboxes)  # CHW
    return masks.gt_(0.0)


def scale_masks(masks, shape, padding=True):
    """
    Rescale segment masks to shape.

    Args:
        masks (torch.Tensor): (N, C, H, W).
        shape (tuple): Height and width.
        padding (bool): If True, assuming the boxes is based on image augmented by yolo style. If False then do regular
            rescaling.
    """
    mh, mw = masks.shape[2:]
    gain = min(mh / shape[0], mw / shape[1])  # gain  = old / new
    pad = [mw - shape[1] * gain, mh - shape[0] * gain]  # wh padding
    if padding:
        pad[0] /= 2
        pad[1] /= 2
    top, left = (int(pad[1]), int(pad[0])) if padding else (0, 0)  # y, x
    bottom, right = (int(mh - pad[1]), int(mw - pad[0]))
    masks = masks[..., top:bottom, left:right]

    masks = F.interpolate(masks, shape, mode="bilinear", align_corners=False)  # NCHW
    return masks


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None, normalize=False, padding=True):
    """
    Rescale segment coordinates (xy) from img1_shape to img0_shape.

    Args:
        img1_shape (tuple): The shape of the image that the coords are from.
        coords (torch.Tensor): the coords to be scaled of shape n,2.
        img0_shape (tuple): the shape of the image that the segmentation is being applied to.
        ratio_pad (tuple): the ratio of the image size to the padded image size.
        normalize (bool): If True, the coordinates will be normalized to the range [0, 1]. Defaults to False.
        padding (bool): If True, assuming the boxes is based on image augmented by yolo style. If False then do regular
            rescaling.

    Returns:
        coords (torch.Tensor): The scaled coordinates.
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    if padding:
        coords[..., 0] -= pad[0]  # x padding
        coords[..., 1] -= pad[1]  # y padding
    coords[..., 0] /= gain
    coords[..., 1] /= gain
    coords = clip_coords(coords, img0_shape)
    if normalize:
        coords[..., 0] /= img0_shape[1]  # width
        coords[..., 1] /= img0_shape[0]  # height
    return coords


def regularize_rboxes(rboxes):
    """
    Regularize rotated boxes in range [0, pi/2].

    Args:
        rboxes (torch.Tensor): Input boxes of shape(N, 5) in xywhr format.

    Returns:
        (torch.Tensor): The regularized boxes.
    """
    x, y, w, h, t = rboxes.unbind(dim=-1)
    # Swap edge and angle if h >= w
    w_ = torch.where(w > h, w, h)
    h_ = torch.where(w > h, h, w)
    t = torch.where(w > h, t, t + math.pi / 2) % math.pi
    return torch.stack([x, y, w_, h_, t], dim=-1)  # regularized boxes


def masks2segments(masks, strategy="largest"):
    """
    It takes a list of masks(n,h,w) and returns a list of segments(n,xy).

    Args:
        masks (torch.Tensor): the output of the model, which is a tensor of shape (batch_size, 160, 160)
        strategy (str): 'concat' or 'largest'. Defaults to largest

    Returns:
        segments (List): list of segment masks
    """
    segments = []
    for x in (masks.int().cpu().numpy() if isinstance(masks, torch.Tensor) else masks).astype("uint8"):
        c = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if c:
            if strategy == "concat":  # concatenate all segments
                c = np.concatenate([x.reshape(-1, 2) for x in c])
            elif strategy == "largest":  # select largest segment
                c = np.array(c[np.array([len(x) for x in c]).argmax()]).reshape(-1, 2)
        else:
            c = np.zeros((0, 2))  # no segments found
        segments.append(c.astype("float32"))
    return segments


def convert_torch2numpy_batch(batch: torch.Tensor) -> np.ndarray:
    """
    Convert a batch of FP32 torch tensors (0.0-1.0) to a NumPy uint8 array (0-255), changing from BCHW to BHWC layout.

    Args:
        batch (torch.Tensor): Input tensor batch of shape (Batch, Channels, Height, Width) and dtype torch.float32.

    Returns:
        (np.ndarray): Output NumPy array batch of shape (Batch, Height, Width, Channels) and dtype uint8.
    """
    return (batch.permute(0, 2, 3, 1).contiguous() * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()


def clean_str(s):
    """
    Cleans a string by replacing special characters with '_' character.

    Args:
        s (str): a string needing special characters replaced

    Returns:
        (str): a string with special characters replaced by an underscore _
    """
    return re.sub(pattern="[|@#!¡·$€%&()=?¿^*;:,¨´><+]", repl="_", string=s)


def process_nms_trt_results(preds: List[Tensor], names: List[str]) -> List[Tensor]:
    """
    Filter TensorRT-like bounding box structure via `max_det`

    Args:
        preds (List[torch.Tensor]): list of
            `num_dets` (shape: Bx1)
            `bboxes` (shape: BxTOP_Kx4)
            `labels` (shape: BxTOP_Kx1)
            `scores` (shape: BxTOP_Kx1).
            Order is not guaranteed but sync with names.
        names (List[str]): list of outputs names like [`num_dets`, ...].

    Returns:
       (List[torch.Tensor]): YOLO-like list of length batch_size, where each element is a tensor of
            shape (num_boxes, 6) containing the kept boxes, with columns
            (x1, y1, x2, y2, confidence, class).
    """

    named_dict = dict(zip(names, preds))  # dict and zip dont copy data
    outputs = []

    for boxes, scores, labels, num_dets in zip(
        named_dict["bboxes"],
        named_dict["scores"],
        named_dict["labels"],
        named_dict["num_dets"],
    ):
        boxes, scores, labels = boxes[:num_dets], scores[:num_dets, None], labels[:num_dets, None]
        outputs.append(torch.hstack([boxes, scores, labels]))

    return outputs


def process_nms_onnx_results(preds: Tensor) -> List[Tensor]:
    """
    Filter ONNX-like bounding box structure via `max_det`

    Args:
        preds (torch.Tensor): Tensor of shape Nx7. Contains
            `batch_index`     - 1
            `bboxes`          - 4
            `max_confidence`  - 1
            `class`           - 1

    Returns:
       (List[torch.Tensor]): YOLO-like list of length batch_size, where each element is a tensor of
            shape (num_boxes, 6) containing the kept boxes, with columns
            (x1, y1, x2, y2, confidence, class).
    """

    batch_index, yolo_dets = preds[..., 0], preds[..., 1:]
    bs = int(batch_index[-1])

    outputs = []

    for i in range(bs + 1):
        outputs.append(yolo_dets[batch_index == i])

    return outputs