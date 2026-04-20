# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Model head modules."""

from __future__ import annotations

import copy
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.init import constant_, xavier_uniform_

from ultralytics.utils import NOT_MACOS14
from ultralytics.utils.tal import dist2bbox, dist2rbox, make_anchors
from ultralytics.utils.torch_utils import disable_dynamo, smart_inference_mode, TORCH_1_11, fuse_conv_and_bn

from .block import DFL, BNContrastiveHead, ContrastiveHead, Proto, Proto26, RealNVP, EfficientTRTNMS, ONNXNMS
from .conv import Conv, Conv2, DWConv
from .transformer import MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer
from .utils import bias_init_with_prob, linear_init

__all__ = (
    "Detect",
    "Segment",
    "Segment26",
    "Pose",
    "Pose26",
    "Classify",
    "OBB",
    "OBB26",
    "RTDETRDecoder",
    "v10Detect",
    "v10Pose",
    "v10Segment",
    "WorldDetect",
    "PostDetectONNXNMS",
    "PostDetectTRTNMS",
)


class Detect(nn.Module):
    """YOLO Detect head for object detection models with multitask support.

    This class implements the detection head used in YOLO models for predicting bounding boxes
    and class probabilities. It supports multitask classification where nc is a list of class
    counts per task, and includes end-to-end detection capabilities.

    Attributes:
        dynamic (bool): Force grid reconstruction.
        export (bool): Export mode flag.
        format (str): Export format.
        max_det (int): Maximum detections per image.
        shape (tuple): Input shape.
        anchors (torch.Tensor): Anchor points.
        strides (torch.Tensor): Feature map strides.
        head_mode (str): Head convolution mode - "legacy", "efficient", or "accurate".
        nc (list[int]): Number of classes per task.
        nl (int): Number of detection layers.
        reg_max (int): DFL channels.
        no (int): Number of outputs per anchor.
        stride (torch.Tensor): Strides computed during build.
        cv2 (nn.ModuleList): Convolution layers for box regression.
        cv3 (nn.ModuleList): Nested convolution layers for classification (per-task, per-scale).
        dfl (nn.Module): Distribution Focal Loss layer.
        one2one_cv2 (nn.ModuleList): One-to-one convolution layers for box regression.
        one2one_cv3 (nn.ModuleList): One-to-one convolution layers for classification.

    Examples:
        Create a detection head for multitask with 80 and 10 classes
        >>> detect = Detect(nc=[80, 10], ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = detect(x)
    """

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    format = None  # export format
    max_det = 300  # max_det
    agnostic_nms = False
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init
    # Head mode: "legacy" (Conv), "efficient" (DWConv), "accurate" (Conv2)
    head_mode = "legacy"

    def __init__(
        self,
        nc: list[int] = (80,),
        reg_max: int = 16,
        end2end: bool = False,
        ch: Tuple[int, ...] = (),
        embed_dim: int = 0,
    ):
        """Initialize the YOLO detection layer with specified number of classes and channels.

        Args:
            nc (list[int]): Number of classes per task.
            reg_max (int): Maximum number of DFL channels.
            end2end (bool): Whether to use end-to-end NMS-free detection.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
            embed_dim (int): Re-ID embedding dimension. 0 means disabled.
        """
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = reg_max
        self.embed_dim = embed_dim
        self.no = self.reg_max * 4 + sum(nc) + self.embed_dim
        self.stride = torch.zeros(self.nl)  # strides computed during build
        self._end2end = end2end

        # Channel dimensions
        c2 = max((16, ch[0] // 4, self.reg_max * 4))
        c3 = [max(ch[0], min(nc_i, 100)) for nc_i in nc]

        # Build box regression head (cv2)
        self.cv2 = nn.ModuleList(self._make_head(self.head_mode, x, c2, 4 * self.reg_max) for x in ch)

        # Build classification heads (cv3) - nested: outer=tasks, inner=scales
        self.cv3 = nn.ModuleList(
            nn.ModuleList(self._make_head(self.head_mode, x, c3[i], nc[i]) for x in ch)
            for i in range(len(nc))
        )

        # Build embedding head (emb) for Re-ID — single head shared between one2many and one2one in end2end models
        if self.embed_dim > 0:
            c_emb = max(ch[0] // 4, self.embed_dim) if len(ch) else self.embed_dim
            self.emb = nn.ModuleList(self._make_head(self.head_mode, c, c_emb, self.embed_dim) for c in ch)

        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        if end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    @staticmethod
    def _make_head(head_mode: str, c_in: int, c_mid: int, c_out: int) -> nn.Module:
        """Create a head block based on head_mode.

        Args:
            head_mode (str): Head mode - "legacy", "efficient", or "accurate".
            c_in (int): Input channels.
            c_mid (int): Intermediate channels.
            c_out (int): Output channels.

        Returns:
            (nn.Module): Sequential block.
        """
        if head_mode == "legacy":
            return nn.Sequential(Conv(c_in, c_mid, 3), Conv(c_mid, c_mid, 3), nn.Conv2d(c_mid, c_out, 1))
        elif head_mode == "accurate":
            return nn.Sequential(Conv2(c_in, c_mid, 3), Conv2(c_mid, c_mid, 3), nn.Conv2d(c_mid, c_out, 1))
        else:  # efficient
            return nn.Sequential(
                nn.Sequential(DWConv(c_in, c_in, 3), Conv(c_in, c_mid, 1)),
                nn.Sequential(DWConv(c_mid, c_mid, 3), Conv(c_mid, c_mid, 1)),
                nn.Conv2d(c_mid, c_out, 1),
            )

    @property
    def end2end(self) -> bool:
        """Check if model has one2one heads for end-to-end detection."""
        return getattr(self, "_end2end", False) and hasattr(self, "one2one_cv2")

    @end2end.setter
    def end2end(self, value: bool):
        """Override the end-to-end detection mode."""
        self._end2end = value

    @property
    def one2many(self) -> dict:
        """Returns the one-to-many head components."""
        d = dict(box_head=self.cv2, cls_head=self.cv3)
        if self.embed_dim > 0:
            d["emb_head"] = self.emb
        return d

    @property
    def one2one(self) -> dict:
        """Returns the one-to-one head components."""
        return dict(
            box_head=getattr(self, "one2one_cv2", None),
            cls_head=getattr(self, "one2one_cv3", None),
        )

    def forward_head(
        self,
        x: list[Tensor],
        box_head: nn.ModuleList | None = None,
        cls_head: nn.ModuleList | None = None,
        emb_head: nn.ModuleList | None = None,
    ) -> dict[str, Tensor] | list[Tensor]:
        """Forward pass through box, classification, and optional embedding heads.

        Args:
            x (list[Tensor]): Feature maps from backbone.
            box_head (nn.ModuleList): Box regression head modules.
            cls_head (nn.ModuleList): Classification head modules (nested for multitask).
            emb_head (nn.ModuleList | None): Embedding head modules for Re-ID.

        Returns:
            (dict[str, Tensor]): Dict with 'boxes', 'scores', 'feats' and optionally 'embeds'.
            (list[Tensor]): For RKNN export, raw outputs per scale/task.
        """
        if box_head is None or cls_head is None:
            return dict()

        # RKNN export: return raw outputs per scale/task as ordered dict
        if self.export and self.format == "rknn":
            y = dict()
            for i in range(self.nl):
                y[f"box_p{i}"] = box_head[i](x[i])
                for t, task_head in enumerate(cls_head):
                    if self.end2end:
                        y[f"cls_t{t}_p{i}"] = task_head[i](x[i])
                    else:
                        cls = task_head[i](x[i]).sigmoid_()
                        y[f"cls_t{t}_p{i}"] = cls
                        y[f"obj_t{t}_p{i}"] = cls.sum(dim=1, keepdim=True).clamp_(0, 1)
                if emb_head is not None:
                    y[f"emb_p{i}"] = emb_head[i](x[i])
            return y

        bs = x[0].shape[0]
        boxes = torch.cat([box_head[i](x[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1)

        # Concatenate all task scores
        scores_list = []
        for task_head in cls_head:  # iterate over tasks
            task_scores = torch.cat([task_head[i](x[i]).view(bs, -1, x[i].shape[-2] * x[i].shape[-1]) 
                                     for i in range(self.nl)], dim=-1)
            scores_list.append(task_scores)
        scores = torch.cat(scores_list, dim=1)  # (bs, sum(nc), num_anchors)

        out = dict(boxes=boxes, scores=scores, feats=x)

        if emb_head is not None:
            out["embeds"] = torch.cat(
                [emb_head[i](x[i]).view(bs, self.embed_dim, -1) for i in range(self.nl)],
                dim=-1,
            )

        return out

    def pre_forward(self, x: list[Tensor]) -> list[Tensor]:
        """Backward-compatible helper returning per-level raw head features."""
        out = []
        for i in range(self.nl):
            parts = [self.cv2[i](x[i])] + [task_head[i](x[i]) for task_head in self.cv3]
            if self.embed_dim > 0:
                parts.append(self.emb[i](x[i]))
            out.append(torch.cat(parts, 1))
        return out

    def forward(
        self, x: list[Tensor]
    ) -> dict[str, Tensor] | Tensor | tuple[Tensor, dict[str, Tensor]]:
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        preds = self.forward_head(x, **self.one2many)
        if self.end2end:
            x_detach = [xi.detach() for xi in x]
            one2one = self.forward_head(x_detach, **self.one2one)
            # Share Re-ID embeddings computed by the one2many branch (same emb head, same x)
            if "embeds" in preds:
                one2one["embeds"] = preds["embeds"]
            # RKNN export: return one2one raw outputs
            if self.export and self.format == "rknn":
                return one2one
            preds = {"one2many": preds, "one2one": one2one}

        # RKNN export (non-end2end): preds is already list from forward_head
        if self.training or (self.export and self.format == "rknn"):
            return preds
        y = self._inference(preds["one2one"] if self.end2end else preds)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1), self.max_det, self.nc)
        return y if self.export else (y, preds)

    @disable_dynamo
    def _inference(self, x: dict[str, Tensor]) -> Tensor:
        """Decode predicted bounding boxes and class probabilities.

        Args:
            x (dict[str, Tensor]): Dict with 'boxes', 'scores', 'feats' from forward_head.

        Returns:
            (Tensor): Concatenated decoded boxes and sigmoid scores.
        """
        shape = x["feats"][0].shape  # BCHW
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (t.transpose(0, 1) for t in make_anchors(x["feats"], self.stride, 0.5))
            self.shape = shape

        box = x["boxes"]
        cls = x["scores"]

        if self.export and self.format in {"tflite", "edgetpu"}:
            grid_h, grid_w = shape[2], shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        parts = [dbox, cls.sigmoid()]
        if "embeds" in x:
            parts.append(F.normalize(x["embeds"], p=2, dim=1))
        return torch.cat(parts, 1)

    def decode_bboxes(self, bboxes: Tensor, anchors: Tensor, xywh: bool = True) -> Tensor:
        """Decode bounding boxes from predictions.

        Args:
            bboxes (Tensor): Raw box predictions.
            anchors (Tensor): Anchor points.
            xywh (bool): Output format.

        Returns:
            (Tensor): Decoded bounding boxes.
        """
        return dist2bbox(bboxes, anchors, xywh=xywh and not self.end2end, dim=1)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        for i, (a, s) in enumerate(zip(self.cv2, self.stride)):
            a[-1].bias.data[:] = 1.0  # box
            for task_head, nc_i in zip(self.cv3, self.nc):
                task_head[i][-1].bias.data[:] = math.log(5 / nc_i / (640 / s) ** 2)

        if self.embed_dim > 0:
            for a in self.emb:
                if a[-1].bias is not None:
                    a[-1].bias.data.zero_()

        if self.end2end:
            for i, (a, s) in enumerate(zip(self.one2one_cv2, self.stride)):
                a[-1].bias.data[:] = 1.0
                for task_head, nc_i in zip(self.one2one_cv3, self.nc):
                    task_head[i][-1].bias.data[:] = math.log(5 / nc_i / (640 / s) ** 2)

    @staticmethod
    def postprocess(preds: Tensor, max_det: int, nc: list[int]) -> Tensor:
        """Post-process predictions with multitask classification support.

        Args:
            preds (Tensor): Predictions with shape (batch_size, num_anchors, 4 + sum(nc) [+ embed_dim]).
            max_det (int): Maximum number of detections.
            nc (list[int]): List of class counts per task.

        Returns:
            (Tensor): Post-processed predictions (batch_size, max_det, 6 [+ embed_dim]).
        """
        total_classes = sum(nc)
        expected = 4 + total_classes
        has_embeds = preds.shape[-1] > expected
        embeds = preds[..., expected:] if has_embeds else None
        boxes, scores = preds[..., :expected].split([4, total_classes], dim=-1)

        # Split scores by task and use first task for ranking
        start_idx = 0
        task_scores = []
        for num_classes in nc:
            end_idx = start_idx + num_classes
            task_scores.append(scores[:, :, start_idx:end_idx])
            start_idx = end_idx

        primary_scores = task_scores[0]
        max_scores = primary_scores.amax(dim=-1)
        max_scores, index = torch.topk(max_scores, min(max_det, max_scores.shape[1]), dim=-1)
        index = index.unsqueeze(-1)

        boxes = torch.gather(boxes, dim=1, index=index.repeat(1, 1, boxes.shape[-1]))
        scores = torch.gather(scores, dim=1, index=index.repeat(1, 1, scores.shape[-1]))
        if has_embeds:
            embeds = torch.gather(embeds, dim=1, index=index.repeat(1, 1, embeds.shape[-1]))

        scores, index = torch.topk(scores.flatten(1), max_det, dim=-1)
        labels = index % total_classes
        index = index // total_classes
        boxes = boxes.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, boxes.shape[-1]))

        parts = [boxes, scores.unsqueeze(-1), labels.unsqueeze(-1).to(boxes.dtype)]
        if has_embeds:
            embeds = embeds.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, embeds.shape[-1]))
            parts.append(embeds)
        return torch.cat(parts, dim=-1)

    def fuse(self) -> None:
        """Remove the one2many head for inference optimization."""
        self.cv2 = self.cv3 = None
        if self.embed_dim > 0:
            self.emb = None

    def upgrade_to_reid(self, embed_dim: int):
        """Add Re-ID embedding branches to an existing Detect head in-place."""
        self.embed_dim = embed_dim
        self.no = self.reg_max * 4 + sum(self.nc) + self.embed_dim
        ch = [cv2[0].conv.in_channels for cv2 in self.cv2]
        c_emb = max(ch[0] // 4, self.embed_dim)
        self.emb = nn.ModuleList(self._make_head(self.head_mode, c, c_emb, self.embed_dim) for c in ch)


class Segment(Detect):
    """YOLO Segment head for segmentation models with multitask support.

    Extends Detect to include mask prediction capabilities for instance segmentation.
    """

    def __init__(
        self,
        nc: list[int] = (80,),
        nm: int = 32,
        npr: int = 256,
        reg_max: int = 16,
        end2end: bool = False,
        ch: Tuple[int, ...] = (),
        embed_dim: int = 0,
    ):
        """Initialize the YOLO segmentation head.

        Args:
            nc (list[int] | int): Number of classes per task.
            nm (int): Number of masks.
            npr (int): Number of protos.
            reg_max (int): Maximum number of DFL channels.
            end2end (bool): Whether to use end-to-end detection.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
            embed_dim (int): Re-ID embedding dimension. 0 means disabled.
        """
        super().__init__(nc, reg_max, end2end, ch, embed_dim=embed_dim)
        self.nm = nm
        self.npr = npr
        self.proto = Proto(ch[0], self.npr, self.nm)

        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch)

        if end2end:
            self.one2one_cv4 = copy.deepcopy(self.cv4)

    @property
    def one2many(self) -> dict:
        """Returns the one-to-many head components including mask head."""
        d = super().one2many
        d["mask_head"] = self.cv4
        return d

    @property
    def one2one(self) -> dict:
        """Returns the one-to-one head components including mask head."""
        d = super().one2one
        d["mask_head"] = getattr(self, "one2one_cv4", None)
        return d

    def forward_head(self, x: list[Tensor], mask_head: nn.ModuleList | None = None, **kwargs) -> dict[str, Tensor]:
        """Forward pass including mask coefficients."""
        preds = super().forward_head(x, **kwargs)
        if mask_head is not None:
            if self.export and self.format == "rknn":
                for i in range(self.nl):
                    preds[f"mask_p{i}"] = mask_head[i](x[i])
            else:
                bs = x[0].shape[0]
                preds["mask_coefficient"] = torch.cat(
                    [mask_head[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2
                )
        return preds

    def forward(self, x: list[Tensor]) -> tuple | dict:
        """Return model outputs and mask coefficients."""
        outputs = super().forward(x)
        preds = outputs[1] if isinstance(outputs, tuple) else outputs
        proto = self.proto(x[0])

        if isinstance(preds, dict):
            if self.end2end:
                preds["one2many"]["proto"] = proto
                preds["one2one"]["proto"] = proto.detach()
            else:
                preds["proto"] = proto

        if self.training:
            return preds

        return (outputs, proto) if self.export else ((outputs[0], proto), preds)

    def _inference(self, x: dict[str, Tensor]) -> Tensor:
        """Decode with mask coefficients."""
        preds = super()._inference(x)
        return torch.cat([preds, x["mask_coefficient"]], dim=1)

    def postprocess(self, preds: Tensor, max_det: int, nc: list[int]) -> Tensor:
        """Post-process with mask coefficients."""
        boxes, scores, mask_coef = preds.split([4, sum(nc), self.nm], dim=-1)

        start_idx = 0
        task_scores = []
        for num_classes in nc:
            end_idx = start_idx + num_classes
            task_scores.append(scores[:, :, start_idx:end_idx])
            start_idx = end_idx

        primary_scores = task_scores[0]
        max_scores = primary_scores.amax(dim=-1)
        max_scores, index = torch.topk(max_scores, min(max_det, max_scores.shape[1]), dim=-1)
        index = index.unsqueeze(-1)

        boxes = boxes.gather(dim=1, index=index.repeat(1, 1, 4))
        scores = scores.gather(dim=1, index=index.repeat(1, 1, sum(nc)))
        mask_coef = mask_coef.gather(dim=1, index=index.repeat(1, 1, self.nm))

        scores, idx = scores.flatten(1).topk(max_det, dim=-1)
        labels = idx % sum(nc)
        idx = idx // sum(nc)
        boxes = boxes.gather(dim=1, index=idx.unsqueeze(-1).repeat(1, 1, 4))
        mask_coef = mask_coef.gather(dim=1, index=idx.unsqueeze(-1).repeat(1, 1, self.nm))

        return torch.cat([boxes, scores.unsqueeze(-1), labels.unsqueeze(-1).float(), mask_coef], dim=-1)

    def fuse(self) -> None:
        """Remove the one2many head for inference optimization."""
        super().fuse()
        self.cv4 = None


class Segment26(Segment):
    """YOLO26 Segment head with Proto26 for semantic segmentation support."""

    def __init__(
        self,
        nc: list[int] = (80,),
        nm: int = 32,
        npr: int = 256,
        reg_max: int = 16,
        end2end: bool = False,
        ch: Tuple[int, ...] = (),
    ):
        """Initialize YOLO26 Segment head with Proto26."""
        super().__init__(nc, nm, npr, reg_max, end2end, ch)
        self.proto = Proto26(ch, self.npr, self.nm, sum(self.nc))

    def forward(self, x: list[Tensor]) -> tuple | dict:
        """Return model outputs with Proto26 mask generation."""
        outputs = Detect.forward(self, x)
        preds = outputs[1] if isinstance(outputs, tuple) else outputs
        proto = self.proto(x)

        if isinstance(preds, dict):
            if self.end2end:
                preds["one2many"]["proto"] = proto
                preds["one2one"]["proto"] = proto.detach() if not isinstance(proto, tuple) else tuple(
                    p.detach() for p in proto
                )
            else:
                preds["proto"] = proto

        if self.training:
            return preds

        proto_out = proto[0] if isinstance(proto, tuple) else proto
        return (outputs, proto_out) if self.export else ((outputs[0], proto_out), preds)

    def fuse(self) -> None:
        """Remove proto semantic segmentation head for inference."""
        super().fuse()
        if hasattr(self.proto, "fuse"):
            self.proto.fuse()


class OBB(Detect):
    """YOLO OBB detection head for oriented bounding boxes with multitask support."""

    def __init__(
        self,
        nc: list[int] = (80,),
        ne: int = 1,
        reg_max: int = 16,
        end2end: bool = False,
        ch: Tuple[int, ...] = (),
    ):
        """Initialize OBB with number of classes and layer channels."""
        super().__init__(nc, reg_max, end2end, ch)
        self.ne = ne

        c4 = max(ch[0] // 4, self.ne)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.ne, 1)) for x in ch)

        if end2end:
            self.one2one_cv4 = copy.deepcopy(self.cv4)

    @property
    def one2many(self) -> dict:
        """Returns the one-to-many head components including angle head."""
        d = super().one2many
        d["angle_head"] = self.cv4
        return d

    @property
    def one2one(self) -> dict:
        """Returns the one-to-one head components including angle head."""
        d = super().one2one
        d["angle_head"] = getattr(self, "one2one_cv4", None)
        return d

    def forward_head(self, x: list[Tensor], angle_head: nn.ModuleList | None = None, **kwargs) -> dict[str, Tensor]:
        """Forward pass including angle predictions."""
        preds = super().forward_head(x, **kwargs)
        if angle_head is not None:
            if self.export and self.format == "rknn":
                for i in range(self.nl):
                    preds[f"angle_p{i}"] = angle_head[i](x[i])
            else:
                bs = x[0].shape[0]
                angle = torch.cat([angle_head[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2)
                angle = (angle.sigmoid() - 0.25) * math.pi
                preds["angle"] = angle
        return preds

    def _inference(self, x: dict[str, Tensor]) -> Tensor:
        """Decode with angle predictions."""
        self.angle = x["angle"]
        preds = super()._inference(x)
        return torch.cat([preds, x["angle"]], dim=1)

    def decode_bboxes(self, bboxes: Tensor, anchors: Tensor) -> Tensor:
        """Decode rotated bounding boxes."""
        return dist2rbox(bboxes, self.angle, anchors, dim=1)

    def postprocess(self, preds: Tensor, max_det: int, nc: list[int]) -> Tensor:
        """Post-process with angle."""
        boxes, scores, angle = preds.split([4, sum(nc), self.ne], dim=-1)

        start_idx = 0
        task_scores = []
        for num_classes in nc:
            end_idx = start_idx + num_classes
            task_scores.append(scores[:, :, start_idx:end_idx])
            start_idx = end_idx

        primary_scores = task_scores[0]
        max_scores = primary_scores.amax(dim=-1)
        max_scores, index = torch.topk(max_scores, min(max_det, max_scores.shape[1]), dim=-1)
        index = index.unsqueeze(-1)

        boxes = boxes.gather(dim=1, index=index.repeat(1, 1, 4))
        scores = scores.gather(dim=1, index=index.repeat(1, 1, sum(nc)))
        angle = angle.gather(dim=1, index=index.repeat(1, 1, self.ne))

        scores, idx = scores.flatten(1).topk(max_det, dim=-1)
        labels = idx % sum(nc)
        idx = idx // sum(nc)
        boxes = boxes.gather(dim=1, index=idx.unsqueeze(-1).repeat(1, 1, 4))
        angle = angle.gather(dim=1, index=idx.unsqueeze(-1).repeat(1, 1, self.ne))

        return torch.cat([boxes, scores.unsqueeze(-1), labels.unsqueeze(-1).float(), angle], dim=-1)

    def fuse(self) -> None:
        """Remove the one2many head for inference optimization."""
        super().fuse()
        self.cv4 = None


class OBB26(OBB):
    """YOLO26 OBB detection head with raw angle predictions (no sigmoid)."""

    def forward_head(self, x: list[Tensor], angle_head: nn.ModuleList | None = None, **kwargs) -> dict[str, Tensor]:
        """Forward pass with raw angle output (no sigmoid transformation)."""
        preds = Detect.forward_head(self, x, **kwargs)
        if angle_head is not None:
            if self.export and self.format == "rknn":
                for i in range(self.nl):
                    preds[f"angle_p{i}"] = angle_head[i](x[i])
            else:
                bs = x[0].shape[0]
                angle = torch.cat([angle_head[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2)
                preds["angle"] = angle
        return preds


class Pose(Detect):
    """YOLO Pose head for keypoint detection with multitask support."""

    def __init__(
        self,
        nc: list[int] = (80,),
        kpt_shape: tuple = (17, 3),
        reg_max: int = 16,
        end2end: bool = False,
        ch: Tuple[int, ...] = (),
    ):
        """Initialize YOLO Pose head."""
        super().__init__(nc, reg_max, end2end, ch)
        self.kpt_shape = kpt_shape
        self.nk = kpt_shape[0] * kpt_shape[1]

        c4 = max(ch[0] // 4, self.nk)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)) for x in ch)

        if end2end:
            self.one2one_cv4 = copy.deepcopy(self.cv4)

    @property
    def one2many(self) -> dict:
        """Returns the one-to-many head components including pose head."""
        d = super().one2many
        d["pose_head"] = self.cv4
        return d

    @property
    def one2one(self) -> dict:
        """Returns the one-to-one head components including pose head."""
        d = super().one2one
        d["pose_head"] = getattr(self, "one2one_cv4", None)
        return d

    def forward_head(self, x: list[Tensor], pose_head: nn.ModuleList | None = None, **kwargs) -> dict[str, Tensor]:
        """Forward pass including keypoint predictions."""
        preds = super().forward_head(x, **kwargs)
        if pose_head is not None:
            if self.export and self.format == "rknn":
                for i in range(self.nl):
                    preds[f"kpt_p{i}"] = pose_head[i](x[i])
            else:
                bs = x[0].shape[0]
                preds["kpts"] = torch.cat([pose_head[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], 2)
        return preds

    def _inference(self, x: dict[str, Tensor]) -> Tensor:
        """Decode with keypoints."""
        preds = super()._inference(x)
        kpts = self.kpts_decode(x["kpts"])
        return torch.cat([preds, kpts], dim=1)

    def kpts_decode(self, kpts: Tensor) -> Tensor:
        """Decode keypoints from predictions."""
        bs = kpts.shape[0]
        ndim = self.kpt_shape[1]

        if self.export:
            y = kpts.view(bs, *self.kpt_shape, -1)
            a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * self.strides
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
            return a.view(bs, self.nk, -1)
        else:
            y = kpts.clone()
            if ndim == 3:
                if NOT_MACOS14:
                    y[:, 2::ndim].sigmoid_()
                else:  # Apple macOS14 MPS bug https://github.com/ultralytics/ultralytics/pull/21878
                    y[:, 2::ndim] = y[:, 2::ndim].sigmoid()
            y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
            y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
            return y

    def postprocess(self, preds: Tensor, max_det: int, nc: list[int]) -> Tensor:
        """Post-process with keypoints."""
        boxes, scores, kpts = preds.split([4, sum(nc), self.nk], dim=-1)

        start_idx = 0
        task_scores = []
        for num_classes in nc:
            end_idx = start_idx + num_classes
            task_scores.append(scores[:, :, start_idx:end_idx])
            start_idx = end_idx

        primary_scores = task_scores[0]
        max_scores = primary_scores.amax(dim=-1)
        max_scores, index = torch.topk(max_scores, min(max_det, max_scores.shape[1]), dim=-1)
        index = index.unsqueeze(-1)

        boxes = boxes.gather(dim=1, index=index.repeat(1, 1, 4))
        scores = scores.gather(dim=1, index=index.repeat(1, 1, sum(nc)))
        kpts = kpts.gather(dim=1, index=index.repeat(1, 1, self.nk))

        scores, idx = scores.flatten(1).topk(max_det, dim=-1)
        labels = idx % sum(nc)
        idx = idx // sum(nc)
        boxes = boxes.gather(dim=1, index=idx.unsqueeze(-1).repeat(1, 1, 4))
        kpts = kpts.gather(dim=1, index=idx.unsqueeze(-1).repeat(1, 1, self.nk))

        return torch.cat([boxes, scores.unsqueeze(-1), labels.unsqueeze(-1).float(), kpts], dim=-1)

    def fuse(self) -> None:
        """Remove the one2many head for inference optimization."""
        super().fuse()
        self.cv4 = None


class Pose26(Pose):
    """YOLO26 Pose head with RealNVP flow model for keypoint uncertainty."""

    def __init__(
        self,
        nc: list[int] = (80,),
        kpt_shape: tuple = (17, 3),
        reg_max: int = 16,
        end2end: bool = False,
        ch: Tuple[int, ...] = (),
    ):
        """Initialize YOLO26 Pose head with RealNVP flow model."""
        super().__init__(nc, kpt_shape, reg_max, end2end, ch)
        self.flow_model = RealNVP()

        c4 = max(ch[0] // 4, kpt_shape[0] * (kpt_shape[1] + 2))
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3)) for x in ch)

        self.cv4_kpts = nn.ModuleList(nn.Conv2d(c4, self.nk, 1) for _ in ch)
        self.nk_sigma = kpt_shape[0] * 2
        self.cv4_sigma = nn.ModuleList(nn.Conv2d(c4, self.nk_sigma, 1) for _ in ch)

        if end2end:
            self.one2one_cv4 = copy.deepcopy(self.cv4)
            self.one2one_cv4_kpts = copy.deepcopy(self.cv4_kpts)
            self.one2one_cv4_sigma = copy.deepcopy(self.cv4_sigma)

    @property
    def one2many(self) -> dict:
        """Returns the one-to-many head components."""
        d = super().one2many
        d["kpts_head"] = self.cv4_kpts
        d["kpts_sigma_head"] = self.cv4_sigma
        return d

    @property
    def one2one(self) -> dict:
        """Returns the one-to-one head components."""
        d = super().one2one
        d["kpts_head"] = getattr(self, "one2one_cv4_kpts", None)
        d["kpts_sigma_head"] = getattr(self, "one2one_cv4_sigma", None)
        return d

    def forward_head(
        self,
        x: list[Tensor],
        pose_head: nn.ModuleList | None = None,
        kpts_head: nn.ModuleList | None = None,
        kpts_sigma_head: nn.ModuleList | None = None,
        **kwargs,
    ) -> dict[str, Tensor]:
        """Forward pass with keypoints and optional sigma."""
        preds = Detect.forward_head(self, x, **kwargs)
        if pose_head is not None:
            if self.export and self.format == "rknn":
                for i in range(self.nl):
                    preds[f"kpt_p{i}"] = pose_head[i](x[i])
            else:
                bs = x[0].shape[0]
                features = [pose_head[i](x[i]) for i in range(self.nl)]
                preds["kpts"] = torch.cat([kpts_head[i](features[i]).view(bs, self.nk, -1) for i in range(self.nl)], 2)
                if self.training and kpts_sigma_head is not None:
                    preds["kpts_sigma"] = torch.cat(
                        [kpts_sigma_head[i](features[i]).view(bs, self.nk_sigma, -1) for i in range(self.nl)], 2
                    )
        return preds

    def kpts_decode(self, kpts: Tensor) -> Tensor:
        """Decode keypoints (YOLO26 version without offset)."""
        bs = kpts.shape[0]
        ndim = self.kpt_shape[1]

        if self.export:
            y = kpts.view(bs, *self.kpt_shape, -1)
            a = (y[:, :, :2] + self.anchors) * self.strides
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
            return a.view(bs, self.nk, -1)
        else:
            y = kpts.clone()
            if ndim == 3:
                y[:, 2::3] = y[:, 2::3].sigmoid()
            y[:, 0::ndim] = (y[:, 0::ndim] + self.anchors[0]) * self.strides
            y[:, 1::ndim] = (y[:, 1::ndim] + self.anchors[1]) * self.strides
            return y

    def fuse(self) -> None:
        """Remove flow model and sigma heads for inference."""
        super().fuse()
        self.cv4_sigma = self.flow_model = None
        if hasattr(self, "one2one_cv4_sigma"):
            self.one2one_cv4_sigma = None


class Classify(nn.Module):
    """YOLO classification head, i.e. x(b,c1,20,20) to x(b,c2)."""

    export = False

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p: int | None = None, g: int = 1):
        """Initialize YOLO classification head."""
        super().__init__()
        c_ = 1280  # efficientnet_b0 size
        self.conv = Conv(c1, c_, k, s, p, g)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.linear = nn.Linear(c_, c2)

    def forward(self, x: list[Tensor] | Tensor) -> Tensor | tuple:
        """Performs forward pass of the YOLO model on input image data."""
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
        if self.training:
            return x
        y = x.softmax(1)
        return y if self.export else (y, x)


class WorldDetect(Detect):
    """Head for integrating YOLO detection with text embeddings."""

    def __init__(
        self,
        nc: list[int] = (80,),
        embed: int = 512,
        with_bn: bool = False,
        reg_max: int = 16,
        end2end: bool = False,
        ch: Tuple[int, ...] = (),
    ):
        """Initialize WorldDetect with text embedding support."""
        super().__init__(nc, reg_max, end2end, ch)
        # Override cv3 with embedding projection
        c3 = max(ch[0], min(sum(self.nc), 100))
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, embed, 1)) for x in ch
        )
        self.cv4 = nn.ModuleList(BNContrastiveHead(embed) if with_bn else ContrastiveHead() for _ in ch)

    def forward(self, x: list[Tensor], text: Tensor) -> dict | tuple:
        """Forward pass with text embeddings."""
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv4[i](self.cv3[i](x[i]), text)), 1)

        if self.training:
            return x

        shape = x[0].shape
        x_cat = torch.cat([xi.view(shape[0], sum(self.nc) + self.reg_max * 4, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (t.transpose(0, 1) for t in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        box, cls = x_cat.split((self.reg_max * 4, sum(self.nc)), 1)

        if self.export and self.format in {"tflite", "edgetpu"}:
            grid_h, grid_w = shape[2], shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize WorldDetect biases."""
        for a, b, s in zip(self.cv2, self.cv3, self.stride):
            a[-1].bias.data[:] = 1.0


class v10Detect(Detect):
    """v10 Detection head with light classification head."""

    end2end = True

    def __init__(self, nc: list[int] = (80,), ch: Tuple[int, ...] = ()):
        """Initialize v10Detect with light cls head."""
        super().__init__(nc, end2end=True, ch=ch)
        # Override cv3 with grouped conv (light head)
        c3 = max(ch[0], min(sum(self.nc), 100))
        self.cv3 = nn.ModuleList(
            nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(Conv(x, x, 3, g=x), Conv(x, c3, 1)),
                    nn.Sequential(Conv(c3, c3, 3, g=c3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, nc_i, 1),
                )
                for x in ch
            )
            for nc_i in self.nc
        )
        self.one2one_cv3 = copy.deepcopy(self.cv3)


class v10Pose(Pose):
    """v10 Pose head with light classification and end-to-end support."""

    end2end = True
    max_det = 300

    def __init__(self, nc: list[int] = (80,), kpt_shape: tuple = (17, 3), ch: Tuple[int, ...] = ()):
        """Initialize v10Pose with light cls head."""
        super().__init__(nc, kpt_shape, end2end=True, ch=ch)
        c3 = max(ch[0], min(sum(self.nc), 100))
        self.cv3 = nn.ModuleList(
            nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(Conv(x, x, 3, g=x), Conv(x, c3, 1)),
                    nn.Sequential(Conv(c3, c3, 3, g=c3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, nc_i, 1),
                )
                for x in ch
            )
            for nc_i in self.nc
        )
        self.one2one_cv3 = copy.deepcopy(self.cv3)


class v10Segment(Segment):
    """v10 Segment head with light classification and end-to-end support."""

    end2end = True
    max_det = 300

    def __init__(self, nc: list[int] = (80,), nm: int = 32, npr: int = 256, ch: Tuple[int, ...] = ()):
        """Initialize v10Segment with light cls head."""
        super().__init__(nc, nm, npr, end2end=True, ch=ch)
        c3 = max(ch[0], min(sum(self.nc), 100))
        self.cv3 = nn.ModuleList(
            nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(Conv(x, x, 3, g=x), Conv(x, c3, 1)),
                    nn.Sequential(Conv(c3, c3, 3, g=c3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, nc_i, 1),
                )
                for x in ch
            )
            for nc_i in self.nc
        )
        self.one2one_cv3 = copy.deepcopy(self.cv3)


class RTDETRDecoder(nn.Module):
    """Real-Time Deformable Transformer Decoder for object detection."""

    export = False

    def __init__(
        self,
        nc: int = 80,
        ch: tuple = (512, 1024, 2048),
        hd: int = 256,
        nq: int = 300,
        ndp: int = 4,
        nh: int = 8,
        ndl: int = 6,
        d_ffn: int = 1024,
        dropout: float = 0.0,
        act: nn.Module = nn.ReLU(),
        eval_idx: int = -1,
        nd: int = 100,
        label_noise_ratio: float = 0.5,
        box_noise_scale: float = 1.0,
        learnt_init_query: bool = False,
    ):
        """Initialize RTDETRDecoder."""
        super().__init__()
        self.hidden_dim = hd
        self.nhead = nh
        self.nl = len(ch)
        self.nc = nc
        self.num_queries = nq
        self.num_decoder_layers = ndl

        self.input_proj = nn.ModuleList(
            nn.Sequential(nn.Conv2d(x, hd, 1, bias=False), nn.BatchNorm2d(hd)) for x in ch
        )

        decoder_layer = DeformableTransformerDecoderLayer(hd, nh, d_ffn, dropout, act, self.nl, ndp)
        self.decoder = DeformableTransformerDecoder(hd, decoder_layer, ndl, eval_idx)

        self.denoising_class_embed = nn.Embedding(nc, hd)
        self.num_denoising = nd
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(nq, hd)
        self.query_pos_head = MLP(4, 2 * hd, hd, num_layers=2)

        self.enc_output = nn.Sequential(nn.Linear(hd, hd), nn.LayerNorm(hd))
        self.enc_score_head = nn.Linear(hd, nc)
        self.enc_bbox_head = MLP(hd, hd, 4, num_layers=3)

        self.dec_score_head = nn.ModuleList([nn.Linear(hd, nc) for _ in range(ndl)])
        self.dec_bbox_head = nn.ModuleList([MLP(hd, hd, 4, num_layers=3) for _ in range(ndl)])

        self._reset_parameters()

    def forward(self, x: list[Tensor], batch: dict | None = None) -> tuple | Tensor:
        """Forward pass returning bounding box and classification scores."""
        from ultralytics.models.utils.ops import get_cdn_group

        feats, shapes = self._get_encoder_input(x)

        dn_embed, dn_bbox, attn_mask, dn_meta = get_cdn_group(
            batch,
            self.nc,
            self.num_queries,
            self.denoising_class_embed.weight,
            self.num_denoising,
            self.label_noise_ratio,
            self.box_noise_scale,
            self.training,
        )

        embed, refer_bbox, enc_bboxes, enc_scores = self._get_decoder_input(feats, shapes, dn_embed, dn_bbox)

        dec_bboxes, dec_scores = self.decoder(
            embed,
            refer_bbox,
            feats,
            shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask,
        )
        x = dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta
        if self.training:
            return x
        y = torch.cat((dec_bboxes.squeeze(0), dec_scores.squeeze(0).sigmoid()), -1)
        return y if self.export else (y, x)

    def _get_encoder_input(self, x: list[Tensor]) -> tuple[Tensor, list]:
        """Process encoder inputs."""
        x = [self.input_proj[i](feat) for i, feat in enumerate(x)]
        feats = []
        shapes = []
        for feat in x:
            h, w = feat.shape[2:]
            feats.append(feat.flatten(2).permute(0, 2, 1))
            shapes.append([h, w])
        feats = torch.cat(feats, 1)
        return feats, shapes

    def _get_decoder_input(
        self,
        feats: Tensor,
        shapes: list,
        dn_embed: Tensor | None = None,
        dn_bbox: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Generate decoder input."""
        bs = feats.shape[0]
        anchors, valid_mask = self._generate_anchors(shapes, dtype=feats.dtype, device=feats.device)

        features = self.enc_output(valid_mask * feats)
        enc_outputs_scores = self.enc_score_head(features)

        topk_ind = torch.topk(enc_outputs_scores.max(-1).values, self.num_queries, dim=1).indices.view(-1)
        batch_ind = torch.arange(end=bs, dtype=topk_ind.dtype).unsqueeze(-1).repeat(1, self.num_queries).view(-1)

        top_k_features = features[batch_ind, topk_ind].view(bs, self.num_queries, -1)
        top_k_anchors = anchors[:, topk_ind].view(bs, self.num_queries, -1)

        refer_bbox = self.enc_bbox_head(top_k_features) + top_k_anchors

        enc_bboxes = refer_bbox.sigmoid()
        if dn_bbox is not None:
            refer_bbox = torch.cat([dn_bbox, refer_bbox], 1)
        enc_scores = enc_outputs_scores[batch_ind, topk_ind].view(bs, self.num_queries, -1)

        embeddings = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1) if self.learnt_init_query else top_k_features
        if self.training:
            refer_bbox = refer_bbox.detach()
            if not self.learnt_init_query:
                embeddings = embeddings.detach()
        if dn_embed is not None:
            embeddings = torch.cat([dn_embed, embeddings], 1)

        return embeddings, refer_bbox, enc_bboxes, enc_scores

    @staticmethod
    def _generate_anchors(
        shapes: list,
        grid_size: float = 0.05,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
        eps: float = 1e-2,
    ) -> tuple[Tensor, Tensor]:
        """Generate anchor bounding boxes."""
        anchors = []
        for i, (h, w) in enumerate(shapes):
            sy = torch.arange(end=h, dtype=dtype, device=device)
            sx = torch.arange(end=w, dtype=dtype, device=device)
            grid_y, grid_x = torch.meshgrid(sy, sx, indexing="ij") if TORCH_1_11 else torch.meshgrid(sy, sx)
            grid_xy = torch.stack([grid_x, grid_y], -1)

            valid_WH = torch.tensor([w, h], dtype=dtype, device=device)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH
            wh = torch.ones_like(grid_xy, dtype=dtype, device=device) * grid_size * (2.0 ** i)
            anchors.append(torch.cat([grid_xy, wh], -1).view(-1, h * w, 4))

        anchors = torch.cat(anchors, 1)
        valid_mask = ((anchors > eps) & (anchors < 1 - eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        anchors = anchors.masked_fill(~valid_mask, float("inf"))
        return anchors, valid_mask

    def _reset_parameters(self):
        """Initialize parameters."""
        bias_cls = bias_init_with_prob(0.01) / 80 * self.nc
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight, 0.0)
        constant_(self.enc_bbox_head.layers[-1].bias, 0.0)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight, 0.0)
            constant_(reg_.layers[-1].bias, 0.0)

        linear_init(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for layer in self.input_proj:
            xavier_uniform_(layer[0].weight)


class PostDetectTRTNMS(nn.Module):
    """YOLOv8 NMS-fused detection model for TensorRT export."""

    export = True
    shape = None
    dynamic = False
    iou = 0.65
    conf = 0.25
    max_det = 100

    def _forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor | None]:
        """Decode yolov8 model output, returning (boxes, scores, embeds_or_None)."""
        res = self.pre_forward(x)
        shape = res[0].shape
        b, b_reg_num = shape[0], self.reg_max * 4
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (t.transpose(0, 1) for t in make_anchors(x, self.stride, 0.5))
            self.shape = shape
        x = [i.view(b, self.no, -1) for i in res]
        y = torch.cat(x, 2)

        ed = getattr(self, "embed_dim", 0)
        nc_ch = self.no - b_reg_num - ed
        boxes_raw = y[:, :b_reg_num, ...]
        scores = y[:, b_reg_num : b_reg_num + nc_ch, ...].sigmoid()
        embeds = F.normalize(y[:, b_reg_num + nc_ch :, ...], p=2, dim=1) if ed > 0 else None

        boxes = boxes_raw.view(b, 4, self.reg_max, -1).permute(0, 1, 3, 2)
        boxes = boxes.softmax(-1) @ torch.arange(self.reg_max, device=boxes.device, dtype=boxes.dtype)
        boxes0, boxes1 = -boxes[:, :2, ...], boxes[:, 2:, ...]
        boxes = self.anchors.repeat(b, 2, 1) + torch.cat([boxes0, boxes1], 1)
        boxes = boxes * self.strides

        return boxes, scores, embeds

    def forward(self, x: Tensor) -> Tensor:
        """Forward with TRT NMS."""
        boxes, scores, _embeds = self._forward(x)

        return EfficientTRTNMS.apply(
            boxes.transpose(1, 2),
            scores.transpose(1, 2),
            self.iou,
            self.conf,
            self.max_det,
        )


class PostDetectONNXNMS(PostDetectTRTNMS):
    """YOLOv8 NMS-fused detection model for ONNX export."""

    def forward(self, x: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        """Forward with ONNX NMS. Returns (detections,) or (detections, embeds)."""
        boxes, scores, embeds = self._forward(x)
        transposed_boxes = boxes.transpose(1, 2)

        selected_indices = ONNXNMS.apply(
            transposed_boxes,
            scores,
            self.max_det,
            self.iou,
            self.conf,
        )

        max_score, category_id = scores.max(1)

        X, Y = selected_indices[:, 0], selected_indices[:, 2]
        selected_boxes = transposed_boxes[X, Y, :]
        selected_categories = category_id[X, Y, None].float()
        selected_scores = max_score[X, Y, None]
        X = X.unsqueeze(1).float()
        dets = torch.cat([X, selected_boxes, selected_scores, selected_categories], 1)
        if embeds is not None:
            return dets, embeds.permute(0, 2, 1)[X.long().squeeze(1).flatten(), Y, :]
        return dets


class LRPCHead(nn.Module):
    """Lightweight Region Proposal and Classification Head for efficient object detection.

    This head combines region proposal filtering with classification to enable efficient detection with dynamic
    vocabulary support.

    Attributes:
        vocab (nn.Module): Vocabulary/classification layer.
        pf (nn.Module): Proposal filter module.
        loc (nn.Module): Localization module.
        enabled (bool): Whether the head is enabled.

    Methods:
        conv2linear: Convert a 1x1 convolutional layer to a linear layer.
        forward: Process classification and localization features to generate detection proposals.

    Examples:
        Create an LRPC head
        >>> vocab = nn.Conv2d(256, 80, 1)
        >>> pf = nn.Conv2d(256, 1, 1)
        >>> loc = nn.Conv2d(256, 4, 1)
        >>> head = LRPCHead(vocab, pf, loc, enabled=True)
    """

    def __init__(self, vocab: nn.Module, pf: nn.Module, loc: nn.Module, enabled: bool = True):
        """Initialize LRPCHead with vocabulary, proposal filter, and localization components.

        Args:
            vocab (nn.Module): Vocabulary/classification module.
            pf (nn.Module): Proposal filter module.
            loc (nn.Module): Localization module.
            enabled (bool): Whether to enable the head functionality.
        """
        super().__init__()
        self.vocab = self.conv2linear(vocab) if enabled else vocab
        self.pf = pf
        self.loc = loc
        self.enabled = enabled

    @staticmethod
    def conv2linear(conv: nn.Conv2d) -> nn.Linear:
        """Convert a 1x1 convolutional layer to a linear layer."""
        assert isinstance(conv, nn.Conv2d) and conv.kernel_size == (1, 1)
        linear = nn.Linear(conv.in_channels, conv.out_channels)
        linear.weight.data = conv.weight.view(conv.out_channels, -1).data
        linear.bias.data = conv.bias.data
        return linear

    def forward(self, cls_feat: torch.Tensor, loc_feat: torch.Tensor, conf: float) -> tuple[tuple, torch.Tensor]:
        """Process classification and localization features to generate detection proposals."""
        if self.enabled:
            pf_score = self.pf(cls_feat)[0, 0].flatten(0)
            mask = pf_score.sigmoid() > conf
            cls_feat = cls_feat.flatten(2).transpose(-1, -2)
            cls_feat = self.vocab(cls_feat[:, mask] if conf else cls_feat * mask.unsqueeze(-1).int())
            return self.loc(loc_feat), cls_feat.transpose(-1, -2), mask
        else:
            cls_feat = self.vocab(cls_feat)
            loc_feat = self.loc(loc_feat)
            return (
                loc_feat,
                cls_feat.flatten(2),
                torch.ones(cls_feat.shape[2] * cls_feat.shape[3], device=cls_feat.device, dtype=torch.bool),
            )


class YOLOEDetect(Detect):
    """Head for integrating YOLO detection models with semantic understanding from text embeddings.

    This class extends the standard Detect head to support text-guided detection with enhanced semantic understanding
    through text embeddings and visual prompt embeddings.

    Attributes:
        is_fused (bool): Whether the model is fused for inference.
        cv3 (nn.ModuleList): Convolution layers for embedding features.
        cv4 (nn.ModuleList): Contrastive head layers for text-vision alignment.
        reprta (Residual): Residual block for text prompt embeddings.
        savpe (SAVPE): Spatial-aware visual prompt embeddings module.
        embed (int): Embedding dimension.

    Methods:
        fuse: Fuse text features with model weights for efficient inference.
        get_tpe: Get text prompt embeddings with normalization.
        get_vpe: Get visual prompt embeddings with spatial awareness.
        forward_lrpc: Process features with fused text embeddings for prompt-free model.
        forward: Process features with class prompt embeddings to generate detections.
        bias_init: Initialize biases for detection heads.

    Examples:
        Create a YOLOEDetect head
        >>> yoloe_detect = YOLOEDetect(nc=80, embed=512, with_bn=True, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> cls_pe = torch.randn(1, 80, 512)
        >>> outputs = yoloe_detect(x, cls_pe)
    """

    is_fused = False

    def __init__(
        self,
        nc: list[int] = (80,),
        embed: int = 512,
        with_bn: bool = False,
        reg_max=16,
        end2end=False,
        ch: tuple = (),
    ):
        """Initialize YOLO detection layer with nc classes and layer channels ch.

        Args:
            nc (list[int]): Number of classes per task.
            embed (int): Embedding dimension.
            with_bn (bool): Whether to use batch normalization in contrastive head.
            reg_max (int): Maximum number of DFL channels.
            end2end (bool): Whether to use end-to-end NMS-free detection.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, reg_max, end2end, ch)
        c3 = max(ch[0], min(sum(self.nc), 100))
        assert c3 <= embed
        assert with_bn
        self.cv3 = (
            nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, embed, 1)) for x in ch)
            if self.head_mode == "legacy"
            else nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, embed, 1),
                )
                for x in ch
            )
        )
        self.cv4 = nn.ModuleList(BNContrastiveHead(embed) if with_bn else ContrastiveHead() for _ in ch)
        if end2end:
            self.one2one_cv3 = copy.deepcopy(self.cv3)  # overwrite with new cv3
            self.one2one_cv4 = copy.deepcopy(self.cv4)

        self.reprta = Residual(SwiGLUFFN(embed, embed))
        self.savpe = SAVPE(ch, c3, embed)
        self.embed = embed

    @smart_inference_mode()
    def fuse(self, txt_feats: torch.Tensor = None):
        """Fuse text features with model weights for efficient inference."""
        if txt_feats is None:  # means eliminate one2many branch
            self.cv2 = self.cv3 = self.cv4 = None
            return
        if self.is_fused:
            return

        assert not self.training
        txt_feats = txt_feats.to(torch.float32).squeeze(0)
        self._fuse_tp(txt_feats, self.cv3, self.cv4)
        if self.end2end:
            self._fuse_tp(txt_feats, self.one2one_cv3, self.one2one_cv4)
        del self.reprta
        self.reprta = nn.Identity()
        self.is_fused = True

    def _fuse_tp(self, txt_feats: torch.Tensor, cls_head: torch.nn.Module, bn_head: torch.nn.Module) -> None:
        """Fuse text prompt embeddings with model weights for efficient inference."""
        for cls_h, bn_h in zip(cls_head, bn_head):
            assert isinstance(cls_h, nn.Sequential)
            assert isinstance(bn_h, BNContrastiveHead)
            conv = cls_h[-1]
            assert isinstance(conv, nn.Conv2d)
            logit_scale = bn_h.logit_scale
            bias = bn_h.bias
            norm = bn_h.norm

            t = txt_feats * logit_scale.exp()
            conv: nn.Conv2d = fuse_conv_and_bn(conv, norm)

            w = conv.weight.data.squeeze(-1).squeeze(-1)
            b = conv.bias.data

            w = t @ w
            b1 = (t @ b.reshape(-1).unsqueeze(-1)).squeeze(-1)
            b2 = torch.ones_like(b1) * bias

            conv = (
                nn.Conv2d(
                    conv.in_channels,
                    w.shape[0],
                    kernel_size=1,
                )
                .requires_grad_(False)
                .to(conv.weight.device)
            )

            conv.weight.data.copy_(w.unsqueeze(-1).unsqueeze(-1))
            conv.bias.data.copy_(b1 + b2)
            cls_h[-1] = conv

            bn_h.fuse()

    def get_tpe(self, tpe: torch.Tensor | None) -> torch.Tensor | None:
        """Get text prompt embeddings with normalization."""
        return None if tpe is None else F.normalize(self.reprta(tpe), dim=-1, p=2)

    def get_vpe(self, x: list[torch.Tensor], vpe: torch.Tensor) -> torch.Tensor:
        """Get visual prompt embeddings with spatial awareness."""
        if vpe.shape[1] == 0:  # no visual prompt embeddings
            return torch.zeros(x[0].shape[0], 0, self.embed, device=x[0].device)
        if vpe.ndim == 4:  # (B, N, H, W)
            vpe = self.savpe(x, vpe)
        assert vpe.ndim == 3  # (B, N, D)
        return vpe

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor | tuple:
        """Process features with class prompt embeddings to generate detections."""
        if hasattr(self, "lrpc"):  # for prompt-free inference
            return self.forward_lrpc(x[:3])
        return super().forward(x)

    def forward_lrpc(self, x: list[torch.Tensor]) -> torch.Tensor | tuple:
        """Process features with fused text embeddings to generate detections for prompt-free model."""
        boxes, scores, index = [], [], []
        bs = x[0].shape[0]
        cv2 = self.cv2 if not self.end2end else self.one2one_cv2
        cv3 = self.cv3 if not self.end2end else self.one2one_cv3
        for i in range(self.nl):
            cls_feat = cv3[i](x[i])
            loc_feat = cv2[i](x[i])
            assert isinstance(self.lrpc[i], LRPCHead)
            box, score, idx = self.lrpc[i](
                cls_feat,
                loc_feat,
                0 if self.export and not self.dynamic else getattr(self, "conf", 0.001),
            )
            boxes.append(box.view(bs, self.reg_max * 4, -1))
            scores.append(score)
            index.append(idx)
        preds = dict(boxes=torch.cat(boxes, 2), scores=torch.cat(scores, 2), feats=x, index=torch.cat(index))
        y = self._inference(preds)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)

    def _get_decode_boxes(self, x):
        """Decode predicted bounding boxes for inference."""
        dbox = super()._get_decode_boxes(x)
        if hasattr(self, "lrpc"):
            dbox = dbox if self.export and not self.dynamic else dbox[..., x["index"]]
        return dbox

    @property
    def one2many(self):
        """Returns the one-to-many head components, here for v5/v5/v8/v9/11 backward compatibility."""
        return dict(box_head=self.cv2, cls_head=self.cv3, contrastive_head=self.cv4)

    @property
    def one2one(self):
        """Returns the one-to-one head components."""
        return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3, contrastive_head=self.one2one_cv4)

    def forward_head(self, x, box_head, cls_head, contrastive_head):
        """Concatenates and returns predicted bounding boxes, class probabilities, and text embeddings."""
        assert len(x) == 4, f"Expected 4 features including 3 feature maps and 1 text embeddings, but got {len(x)}."
        if box_head is None or cls_head is None:  # for fused inference
            return dict()
        bs = x[0].shape[0]  # batch size
        boxes = torch.cat([box_head[i](x[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1)
        self.nc = x[-1].shape[1]
        scores = torch.cat(
            [contrastive_head[i](cls_head[i](x[i]), x[-1]).reshape(bs, self.nc, -1) for i in range(self.nl)], dim=-1
        )
        self.no = self.nc + self.reg_max * 4  # self.nc could be changed when inference with different texts
        return dict(boxes=boxes, scores=scores, feats=x[:3])

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        for i, (a, b, c) in enumerate(
            zip(self.one2many["box_head"], self.one2many["cls_head"], self.one2many["contrastive_head"])
        ):
            a[-1].bias.data[:] = 2.0  # box
            b[-1].bias.data[:] = 0.0
            c.bias.data[:] = math.log(5 / self.nc / (640 / self.stride[i]) ** 2)
        if self.end2end:
            for i, (a, b, c) in enumerate(
                zip(self.one2one["box_head"], self.one2one["cls_head"], self.one2one["contrastive_head"])
            ):
                a[-1].bias.data[:] = 2.0  # box
                b[-1].bias.data[:] = 0.0
                c.bias.data[:] = math.log(5 / self.nc / (640 / self.stride[i]) ** 2)


class YOLOESegment(YOLOEDetect):
    """YOLO segmentation head with text embedding capabilities.

    This class extends YOLOEDetect to include mask prediction capabilities for instance segmentation tasks with
    text-guided semantic understanding.

    Attributes:
        nm (int): Number of masks.
        npr (int): Number of protos.
        proto (Proto): Prototype generation module.
        cv5 (nn.ModuleList): Convolution layers for mask coefficients.

    Methods:
        forward: Return model outputs and mask coefficients.

    Examples:
        Create a YOLOESegment head
        >>> yoloe_segment = YOLOESegment(nc=80, nm=32, npr=256, embed=512, with_bn=True, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> text = torch.randn(1, 80, 512)
        >>> outputs = yoloe_segment(x, text)
    """

    def __init__(
        self,
        nc: list[int] = (80,),
        nm: int = 32,
        npr: int = 256,
        embed: int = 512,
        with_bn: bool = False,
        reg_max=16,
        end2end=False,
        ch: tuple = (),
    ):
        """Initialize YOLOESegment with class count, mask parameters, and embedding dimensions.

        Args:
            nc (list[int]): Number of classes per task.
            nm (int): Number of masks.
            npr (int): Number of protos.
            embed (int): Embedding dimension.
            with_bn (bool): Whether to use batch normalization in contrastive head.
            reg_max (int): Maximum number of DFL channels.
            end2end (bool): Whether to use end-to-end NMS-free detection.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, embed, with_bn, reg_max, end2end, ch)
        self.nm = nm
        self.npr = npr
        self.proto = Proto(ch[0], self.npr, self.nm)

        c5 = max(ch[0] // 4, self.nm)
        self.cv5 = nn.ModuleList(nn.Sequential(Conv(x, c5, 3), Conv(c5, c5, 3), nn.Conv2d(c5, self.nm, 1)) for x in ch)
        if end2end:
            self.one2one_cv5 = copy.deepcopy(self.cv5)

    @property
    def one2many(self):
        """Returns the one-to-many head components, here for v5/v5/v8/v9/11 backward compatibility."""
        return dict(box_head=self.cv2, cls_head=self.cv3, mask_head=self.cv5, contrastive_head=self.cv4)

    @property
    def one2one(self):
        """Returns the one-to-one head components."""
        return dict(
            box_head=self.one2one_cv2,
            cls_head=self.one2one_cv3,
            mask_head=self.one2one_cv5,
            contrastive_head=self.one2one_cv4,
        )

    def forward_lrpc(self, x: list[torch.Tensor]) -> torch.Tensor | tuple:
        """Process features with fused text embeddings to generate detections for prompt-free model."""
        boxes, scores, index = [], [], []
        bs = x[0].shape[0]
        cv2 = self.cv2 if not self.end2end else self.one2one_cv2
        cv3 = self.cv3 if not self.end2end else self.one2one_cv3
        cv5 = self.cv5 if not self.end2end else self.one2one_cv5
        for i in range(self.nl):
            cls_feat = cv3[i](x[i])
            loc_feat = cv2[i](x[i])
            assert isinstance(self.lrpc[i], LRPCHead)
            box, score, idx = self.lrpc[i](
                cls_feat,
                loc_feat,
                0 if self.export and not self.dynamic else getattr(self, "conf", 0.001),
            )
            boxes.append(box.view(bs, self.reg_max * 4, -1))
            scores.append(score)
            index.append(idx)
        mc = torch.cat([cv5[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)
        index = torch.cat(index)
        preds = dict(
            boxes=torch.cat(boxes, 2),
            scores=torch.cat(scores, 2),
            feats=x,
            index=index,
            mask_coefficient=mc * index.int() if self.export and not self.dynamic else mc[..., index],
        )
        y = self._inference(preds)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)

    def forward(self, x: list[torch.Tensor]) -> tuple | list[torch.Tensor] | dict[str, torch.Tensor]:
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        outputs = super().forward(x)
        preds = outputs[1] if isinstance(outputs, tuple) else outputs
        proto = self.proto(x[0])  # mask protos
        if isinstance(preds, dict):  # training and validating during training
            if self.end2end:
                preds["one2many"]["proto"] = proto
                preds["one2one"]["proto"] = proto.detach()
            else:
                preds["proto"] = proto
        if self.training:
            return preds
        return (outputs, proto) if self.export else ((outputs[0], proto), preds)

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Decode predicted bounding boxes and class probabilities, concatenated with mask coefficients."""
        preds = super()._inference(x)
        return torch.cat([preds, x["mask_coefficient"]], dim=1)

    def forward_head(
        self,
        x: list[torch.Tensor],
        box_head: torch.nn.Module,
        cls_head: torch.nn.Module,
        mask_head: torch.nn.Module,
        contrastive_head: torch.nn.Module,
    ) -> torch.Tensor:
        """Concatenates and returns predicted bounding boxes, class probabilities, and mask coefficients."""
        preds = super().forward_head(x, box_head, cls_head, contrastive_head)
        if mask_head is not None:
            bs = x[0].shape[0]  # batch size
            preds["mask_coefficient"] = torch.cat([mask_head[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)
        return preds

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        """Post-process YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc + nm) with last dimension
                format [x, y, w, h, class_probs, mask_coefficient].

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6 + nm) and last
                dimension format [x, y, w, h, max_class_prob, class_index, mask_coefficient].
        """
        boxes, scores, mask_coefficient = preds.split([4, self.nc, self.nm], dim=-1)
        scores, conf, idx = self.get_topk_index(scores, self.max_det)
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
        mask_coefficient = mask_coefficient.gather(dim=1, index=idx.repeat(1, 1, self.nm))
        return torch.cat([boxes, scores, conf, mask_coefficient], dim=-1)

    def fuse(self, txt_feats: torch.Tensor = None):
        """Fuse text features with model weights for efficient inference."""
        super().fuse(txt_feats)
        if txt_feats is None:  # means eliminate one2many branch
            self.cv5 = None
            if hasattr(self.proto, "fuse"):
                self.proto.fuse()
            return


class YOLOESegment26(YOLOESegment):
    """YOLOE-style segmentation head module using Proto26 for mask generation.

    This class extends the YOLOEDetect functionality to include segmentation capabilities by integrating a prototype
    generation module and convolutional layers to predict mask coefficients.

    Args:
        nc (int): Number of classes. Defaults to 80.
        nm (int): Number of masks. Defaults to 32.
        npr (int): Number of prototype channels. Defaults to 256.
        embed (int): Embedding dimensionality. Defaults to 512.
        with_bn (bool): Whether to use Batch Normalization. Defaults to False.
        reg_max (int): Maximum regression value for bounding boxes. Defaults to 16.
        end2end (bool): Whether to use end-to-end detection mode. Defaults to False.
        ch (tuple[int, ...]): Input channels for each scale.

    Attributes:
        nm (int): Number of segmentation masks.
        npr (int): Number of prototype channels.
        proto (Proto26): Prototype generation module for segmentation.
        cv5 (nn.ModuleList): Convolutional layers for generating mask coefficients from features.
        one2one_cv5 (nn.ModuleList, optional): Deep copy of cv5 for end-to-end detection branches.
    """

    def __init__(
        self,
        nc: list[int] = (80,),
        nm: int = 32,
        npr: int = 256,
        embed: int = 512,
        with_bn: bool = False,
        reg_max=16,
        end2end=False,
        ch: tuple = (),
    ):
        """Initialize YOLOESegment26 with class count, mask parameters, and embedding dimensions."""
        YOLOEDetect.__init__(self, nc, embed, with_bn, reg_max, end2end, ch)
        self.nm = nm
        self.npr = npr
        self.proto = Proto26(ch, self.npr, self.nm, sum(self.nc))  # protos

        c5 = max(ch[0] // 4, self.nm)
        self.cv5 = nn.ModuleList(nn.Sequential(Conv(x, c5, 3), Conv(c5, c5, 3), nn.Conv2d(c5, self.nm, 1)) for x in ch)
        if end2end:
            self.one2one_cv5 = copy.deepcopy(self.cv5)

    def forward(self, x: list[torch.Tensor]) -> tuple | list[torch.Tensor] | dict[str, torch.Tensor]:
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        outputs = YOLOEDetect.forward(self, x)
        preds = outputs[1] if isinstance(outputs, tuple) else outputs
        proto = self.proto([xi.detach() for xi in x], return_semseg=False)  # mask protos

        if isinstance(preds, dict):  # training and validating during training
            if self.end2end and not hasattr(self, "lrpc"):  # not prompt-free
                preds["one2many"]["proto"] = proto
                preds["one2one"]["proto"] = proto.detach()
            else:
                preds["proto"] = proto
        if self.training:
            return preds
        return (outputs, proto) if self.export else ((outputs[0], proto), preds)
