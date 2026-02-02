# Ultralytics YOLO 🚀, AGPL-3.0 license

import contextlib
import math
import re
import time
from typing import List, Sequence

import cv2
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from ultralytics.utils import LOGGER
from ultralytics.utils.metrics import batch_probiou
from ultralytics.utils.tf import (
    xyxy2xywh,
    xywh2xyxy,
    clip_boxes,
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
    prediction: Tensor,
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
    end2end=False,
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

    if prediction.shape[-1] == 6 or prediction.shape[-2] == max_det or end2end:  # end-to-end model (BNC, i.e. 1,300,6)
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
    first_nc = nc[0] or nm
    xc = prediction[:, 4: 4 + first_nc].amax(1) > conf_thres

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
            if len(nc) > 1:  # multi-task
                unique_id = j.view(-1, 1)
                multiplier = nc[0]
                for head_idx in range(1, len(nc)):
                    attr_class = clss[head_idx].view(-1, 1)
                    unique_id = unique_id + attr_class * multiplier
                    multiplier *= nc[head_idx]
                c = unique_id * max_wh
            else:  # one head - original implementation
                c = j.view(-1, 1) * max_wh
        scores = conf

        if rotated:
            boxes = torch.cat((x[:, :2] + c, x[:, 2:4], x[:, -1:]), dim=-1)  # xywhr
            i = nms_rotated(boxes, scores, iou_thres)
        else:
            boxes = x[:, :4] + c  # boxes (offset by class)
            # torchvision NMS requires float32 on CPU
            i = torchvision.ops.nms(boxes.float(), scores.float(), iou_thres)  # NMS
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


def dfl(position: Tensor) -> Tensor:
    # Distribution Focal Loss (DFL)
    n, c, h, w = position.shape
    p_num = 4
    mc = c // p_num
    y = position.view(n, p_num, mc, h, w).softmax(dim=2)
    bins = torch.arange(mc, device=position.device, dtype=position.dtype).view(1, 1, mc, 1, 1)
    return (y * bins).sum(2)


def box_process(position: Tensor, imgsz: tuple[int, int]) -> Tensor:
    """
    Process DFL results into YOLO-style predictions.

    Args:
        position (Tensor): Tensor containing the DFL results.
        imgsz (tuple[int, int]): Image size.

    Returns:
        Tensor: xywh layout shaped (batch, 4, H, W).
    """
    device, dtype = position.device, position.dtype
    grid_h, grid_w = position.shape[2:4]
    y = torch.arange(grid_h, device=device, dtype=dtype)
    x = torch.arange(grid_w, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    grid = torch.stack((grid_x, grid_y), dim=0).unsqueeze(0)  # (1,2,H,W) with channel0=x, channel1=y
    # stride aligns x with width (grid_w) and y with height (grid_h)
    stride = torch.tensor([imgsz[1] / grid_w, imgsz[0] / grid_h], device=device, dtype=dtype).view(1, 2, 1, 1)

    position = dfl(position)
    xywh = torch.empty_like(position[:, :4])
    neg = position[:, 0:2]
    pos = position[:, 2:4]

    # center = (grid + 0.5) + (pos - neg) / 2
    xywh[:, 0:2] = (grid + 0.5 + (pos - neg) * 0.5) * stride
    xywh[:, 2:4] = (neg + pos) * stride

    return xywh

def process_rknn_dfl_results(
    input_data: List[Tensor],
    default_branch: int = 3,
    imgsz: tuple[int, int] = (640, 640),
    conf_thres: float = 0.01,
) -> Tensor:
    """
    Process RKNN DFL results into YOLO-style predictions.

    Args:
        input_data (List[Tensor]): List of tensors containing the DFL results.
        default_branch (int): Number of default branches.
        imgsz (tuple[int, int]): Image size.

    Returns:
        Tensor: Tensor shaped (batch, 4 + sum(num_classes), num_boxes) ready for NMS.
    """
    boxes, classes_conf, scores_conf = [], [], []
    pair_per_branch = len(input_data)//default_branch
    for i in range(default_branch):
        boxes.append(box_process(input_data[pair_per_branch*i], imgsz=imgsz))
        classes_conf.append(input_data[pair_per_branch*i+1])
        scores_conf.append(input_data[pair_per_branch*i+2])

    def sp_flatten(_in: Tensor) -> Tensor:
        b, ch, h, w = _in.shape
        return _in.reshape(b, ch, h * w)

    boxes = torch.cat([sp_flatten(_v) for _v in boxes], dim=2)
    classes_conf = torch.cat([sp_flatten(_v) for _v in classes_conf], dim=2)
    obj_conf = torch.cat([sp_flatten(_v) for _v in scores_conf], dim=2)
    # drop cells below objectness threshold while keeping shape
    keep = (obj_conf >= conf_thres).to(boxes.dtype)
    boxes = boxes * keep
    classes_conf = classes_conf * keep

    return torch.cat((boxes, classes_conf), dim=1)

def process_rknn_end2end_results(
    input_data: List[Tensor],
    imgsz: tuple[int, int] = (640, 640),
    conf_thres: float = 0.01,
    nc: list[int] = [80],
    strides: tuple[int, ...] = (8, 16, 32),
) -> Tensor:
    """Process RKNN end2end model outputs into predictions format for NMS.

    This function takes raw outputs from an RKNN-exported YOLO model (with end2end=True)
    and converts them to the standard prediction format expected by non_max_suppression.

    Args:
        input_data: List of tensors from RKNN model in format [reg0, cls0, reg1, cls1, ...]
                   where reg shape is (bs, 4, h, w) and cls shape is (bs, nc, h, w).
                   For multitask models: [reg0, cls0_task0, cls0_task1, ..., reg1, ...]
        imgsz: Model input size (height, width) used during export.
        conf_thres: Confidence threshold for early filtering (optional optimization).
        nc: Number of classes. Can be int for single task or list for multitask.
        strides: Feature map strides for each detection layer.

    Returns:
        Tensor: Predictions with shape (batch_size, num_anchors, 4 + sum(nc))
                Box format is xyxy coordinates, scores are after sigmoid.

    Examples:
        >>> outputs = model(img)  # RKNN model outputs
        >>> preds = process_rknn_end2end_results(outputs, imgsz=(640, 640), nc=80)
        >>> results = non_max_suppression(preds, conf_thres=0.25, iou_thres=0.45)
    """
    from ultralytics.utils.tal import dist2bbox, make_anchors

    num_tasks = len(nc)
    total_nc = sum(nc)

    # Determine number of detection layers
    # Format: [reg0, cls0_t0, cls0_t1, ..., reg1, cls1_t0, ...]
    # Each scale has 1 reg + num_tasks cls outputs
    outputs_per_scale = 1 + num_tasks
    nl = len(input_data) // outputs_per_scale

    bs = input_data[0].shape[0]

    regs, clss, feats = [], [], []

    for i in range(nl):
        base_idx = i * outputs_per_scale
        reg = input_data[base_idx]  # (bs, 4, h, w)
        h, w = reg.shape[2], reg.shape[3]

        regs.append(reg.view(bs, 4, -1))  # (bs, 4, h*w)
        feats.append(reg)

        # Collect all task cls outputs for this scale
        scale_cls = []
        for t in range(num_tasks):
            cls = input_data[base_idx + 1 + t]  # (bs, nc[t], h, w)
            scale_cls.append(cls.view(bs, nc[t], -1))  # (bs, nc[t], h*w)
        clss.append(torch.cat(scale_cls, dim=1))  # (bs, total_nc, h*w)

    # Concatenate across scales
    boxes = torch.cat(regs, dim=-1)  # (bs, 4, total_anchors)
    scores = torch.cat(clss, dim=-1)  # (bs, total_nc, total_anchors)

    # Generate anchors and strides
    stride_tensor = torch.tensor(
        strides[:nl],
        device=input_data[0].device,
        dtype=input_data[0].dtype,
    )
    anchors, strides_out = make_anchors(feats, stride_tensor, 0.5)
    anchors = anchors.transpose(0, 1)  # (2, total_anchors)
    strides_out = strides_out.transpose(0, 1)  # (1, total_anchors)

    # Decode boxes: boxes contains [left, top, right, bottom] distances from anchor
    # dist2bbox converts ltrb distances to xyxy coordinates
    dbox = dist2bbox(boxes, anchors.unsqueeze(0), xywh=False, dim=1) * strides_out

    # Apply sigmoid to class scores
    scores = scores.sigmoid()

    # Combine and transpose: (bs, 4+nc, anchors) -> (bs, anchors, 4+nc)
    preds = torch.cat([dbox, scores], dim=1).permute(0, 2, 1)

    return preds
