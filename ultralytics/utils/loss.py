# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import math
from typing import Any

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.metrics import CITYSCAPES_WEIGHT, OKS_SIGMA, RLE_WEIGHT
from ultralytics.utils.ops import crop_mask
from ultralytics.utils.tf import xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import (
    RotatedTaskAlignedAssigner,
    TaskAlignedAssigner,
    dist2bbox,
    dist2rbox,
    make_anchors,
    rbox2dist,
    bbox2dist,
)
from ultralytics.utils.torch_utils import autocast, disable_dynamo

from .metrics import bbox_iou, probiou, WiseIoULoss, wasserstein_loss


class MetricLearningLoss(nn.Module):
    """Self-supervised Re-ID embedding loss (SupCon-style).

    Hand-rolled supervised contrastive loss (Khosla et al., https://arxiv.org/abs/2004.11362)
    over the unique-per-instance tags emitted by the dataset. Each anchor compares against
    every other anchor in the batch as either same-tag positive or different-tag negative,
    giving a much smoother gradient than triplet-margin mining. Background anchors
    (tag <= 0) are kept as pure negatives. Embeddings are L2-normalized inside this module
    so callers don't need to normalize first.

    Magnitude normalization
    -----------------------
    Raw SupCon at τ=0.1 with N≈1k anchors lands at ~log(N)/τ ≈ 70 untrained, while the
    other YOLO loss components (box/cls/dfl/dep) sit at ~0.05–0.4 in the same regime.
    To put reid on the same magnitude scale — so ``hyp.reid = 1.0`` is consistent with
    ``hyp.box = 1.0`` etc. without needing per-coefficient hand tuning — we divide the
    per-anchor SupCon loss by ``log(N - 1) / τ`` (its untrained upper bound). After
    normalization the loss is ~1.0 for random embeddings and decreases toward 0 as the
    embedding head learns.

    Args:
        temperature: SupCon temperature τ. 0.1 is the canonical default.
        max_samples: optional cap on anchors per step (random sub-sample when exceeded).
            Keeps the (NxN) similarity matrix from blowing up on large batches.
    """

    def __init__(self, temperature: float = 0.1, max_samples: int = 4096):
        super().__init__()
        self.temperature = temperature
        self.max_samples = max_samples

    def forward(self, embeddings, tags, confidences=None, normalize=True):  # confidences kept for back-compat
        if embeddings.numel() == 0:
            return embeddings.sum() * 0.0

        n = embeddings.shape[0]
        if n > self.max_samples:
            idx = torch.randperm(n, device=embeddings.device)[: self.max_samples]
            embeddings = embeddings[idx]
            tags = tags[idx]
            n = self.max_samples

        if normalize:
            embeddings = F.normalize(embeddings.float(), p=2, dim=1)

        tags = tags.view(-1)
        # Same-tag mask, exclude self pairs.
        pos_mask = (tags.unsqueeze(0) == tags.unsqueeze(1)).float()
        pos_mask.fill_diagonal_(0.0)
        # Background tags (tag <= 0) act as pure negatives: they keep their negative
        # comparisons (everyone else is anchor) but never contribute as a positive.
        bg = (tags <= 0).float()
        pos_mask = pos_mask * (1.0 - bg).unsqueeze(0) * (1.0 - bg).unsqueeze(1)

        # Skip cleanly when no positives are available (e.g. all-unique-tags batch).
        n_pos_per_anchor = pos_mask.sum(dim=1)
        if n_pos_per_anchor.sum() == 0:
            return embeddings.sum() * 0.0

        sim = embeddings @ embeddings.t() / self.temperature
        # Mask self-similarity in the denominator for numerical stability.
        sim = sim - sim.detach().max()
        logits_mask = 1.0 - torch.eye(n, device=embeddings.device, dtype=sim.dtype)
        exp_sim = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True).clamp_min(1e-12))

        # Mean over positives per anchor; skip anchors with no positives.
        valid_anchor = n_pos_per_anchor > 0
        loss_per_anchor = -(pos_mask * log_prob).sum(dim=1) / n_pos_per_anchor.clamp_min(1)
        raw = loss_per_anchor[valid_anchor].mean()

        # Magnitude normalization to match box/cls/dfl/dep dynamic range.
        # Untrained upper bound is ~log(N - 1)/τ; dividing by it puts loss in [~0, ~1].
        denom = math.log(max(n - 1, 2)) / self.temperature
        return raw / denom


class DistillationLoss(nn.Module):
    """Criterion class for computing training losses.
    Calculates KL-divergence loss for each classification head separately and sums them up."""

    def __init__(self, model, temperature=3.0, alpha=0.5, task="detect"):  # model must be de-paralleled
        """Initializes DistillationLoss with the model, defining model-related properties and BCE loss function."""
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

        if task == "classify":
            self.forward = self.forward_classify
        else:
            m = model.model[-1]  # Detect() module
            self.reg_max = m.reg_max
            self.nc_list = m.nc if isinstance(m.nc, list) else [m.nc]
            self.no = self.reg_max * 4 + sum(self.nc_list)
            self.forward = self.forward_detect
    
    def forward_detect(self, student_logits, teacher_logits):
        """Calculates distillation loss for object detection, supporting multi-task heads."""
        s_cat = torch.cat([xi.view(student_logits[0].shape[0], self.no, -1) for xi in student_logits], 2)
        t_cat = torch.cat([xi.view(teacher_logits[0].shape[0], self.no, -1) for xi in teacher_logits], 2)

        split_sizes = (self.reg_max * 4, *self.nc_list)
        
        _, *s_scores_list = s_cat.split(split_sizes, 1)
        _, *t_scores_list = t_cat.split(split_sizes, 1)

        total_loss = 0.0
        
        for s_scores_task, t_scores_task in zip(s_scores_list, t_scores_list):
            s_scores_task = s_scores_task.permute(0, 2, 1).contiguous()
            t_scores_task = t_scores_task.permute(0, 2, 1).contiguous()

            student_soft = F.log_softmax(s_scores_task / self.temperature, dim=-1)
            teacher_soft = F.softmax(t_scores_task / self.temperature, dim=-1)
            task_loss = F.kl_div(student_soft, teacher_soft, reduction='none').sum() / s_scores_task.shape[0]
            
            total_loss += task_loss
        return self.alpha * total_loss * (self.temperature ** 2)

    def forward_classify(self, student_logits, teacher_logits):
        """Calculates distillation loss for classification."""
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)

        loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (self.temperature ** 2)
        return self.alpha * loss


class QualityFocalLoss(nn.Module):
    """Quality Focal Loss by Xiang et al. https://arxiv.org/abs/2006.04388."""
    
    def __init__(self, weight=None, *args, **kwargs):
        """Initialize the Quality focal loss class."""
        super().__init__()
        self.weight = weight

    def preprocess(self, pred_scores, gt_scores, pred_bboxes, gt_bboxes, fg_mask):
        """Pre-process the prediction and label."""
        if fg_mask.sum():
            pos_ious = bbox_iou(pred_bboxes, gt_bboxes, xywh=False).clamp(min=1e-6).detach()
            cls_iou_targets = pos_ious * gt_scores
            fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, pred_scores.shape[-1])  # (b, h*w, 80)
            condition = fg_scores_mask > 0
            targets_onehot_pos = torch.where(condition, gt_scores, 0)
            cls_iou_targets = torch.where(condition, cls_iou_targets, 0)
        else:
            cls_iou_targets = torch.zeros_like(pred_scores)
            targets_onehot_pos = torch.zeros_like(pred_scores)

        return cls_iou_targets, targets_onehot_pos.bool()

    def forward(self, pred_scores, gt_scores, pred_bboxes, gt_bboxes, fg_mask, beta=2.0, *args, **kwargs):
        """Computes quality focal loss."""
        cls_iou_targets, targets_onehot_pos = self.preprocess(pred_scores, gt_scores, pred_bboxes, gt_bboxes, fg_mask)
        
        pred_sigmoid = pred_scores.float().sigmoid()
        scale_factor = pred_sigmoid
        zerolabel = torch.zeros_like(pred_scores)

        with autocast(enabled=False):
            loss = F.binary_cross_entropy_with_logits(
                pred_scores,
                zerolabel,
                reduction='none',
                weight=self.weight,
            ) * scale_factor.pow(beta)
        scale_factor = cls_iou_targets[targets_onehot_pos] - pred_sigmoid[targets_onehot_pos]
        with autocast(enabled=False):
            loss[targets_onehot_pos] = F.binary_cross_entropy_with_logits(
                pred_scores[targets_onehot_pos],
                cls_iou_targets[targets_onehot_pos],
                reduction='none',
            ) * scale_factor.abs().pow(beta)
        return loss


class EffectiveClassMarginLoss(nn.Module):
    """Effective Class Margin Loss for long-tail object detection.
    
    Paper: https://arxiv.org/abs/2104.00466
    Adapted for Ultralytics YOLO from MMDetection.
    """
    
    def __init__(self, 
                 num_classes=6,
                 fg_bg_ratio=6.3313,
                 loss_weight=1.0,
                 reduction='none',
                 weight=None,
                 **kwargs):
        super().__init__()
        self.loss_weight = loss_weight 
        self.num_classes = num_classes 
        self.fg_bg_ratio = fg_bg_ratio
        self.reduction = reduction
        self.class_weight = weight

        if num_classes == 6:
            n = torch.tensor([3097.0, 5756.0, 1355.0, 6835.0, 5822.0, 2295.0])
            LOGGER.info(f"{colorstr('ECM Loss')}: Using hardcoded frequencies for 6-class dataset")
        else:
            LOGGER.warning(f"{colorstr('ECM Loss')}: Unexpected num_classes={num_classes}, using uniform distribution")
            n = torch.ones(num_classes)
        
        n = torch.cat([n, n.sum().unsqueeze(0) * fg_bg_ratio])
        
        total_samples = n[:-1].sum().item()
        bg_samples = n[-1].item()
        LOGGER.info(f"{colorstr('ECM Loss')}: Total FG samples: {total_samples:.0f}, BG samples: {bg_samples:.0f}")

        self.register_buffer('sample_n', n)
        self.register_buffer('detection_cls_weight', EffectiveClassMarginLoss.get_detection_weight(n))
        
        weights = self.detection_cls_weight.squeeze().tolist()
        LOGGER.info(f"{colorstr('ECM Loss')}: Class weights: {[f'{w:.3f}' for w in weights[:num_classes]]}")

    @staticmethod
    def get_detection_weight(n):
        """Compute detection weight for ECM Loss."""
        a = (n.sum() - n) / n
        w = a * ((1 + a) / a).log()
        return w[None] 

    def compute_weight(self, cls_score):
        """Compute margin weights for positive and negative samples."""
        B, C = cls_score.shape
        
        n_pos = self.sample_n 
        n_neg = self.sample_n.sum() - self.sample_n
        
        eps = 1e-9
        pos_w = (n_neg.pow(1/4) / (n_pos.pow(1/4) + n_neg.pow(1/4) + eps)).pow(-1).log()
        neg_w = (n_pos.pow(1/4) / (n_pos.pow(1/4) + n_neg.pow(1/4) + eps)).pow(-1).log()
        
        pos_w = pos_w.view(1, -1).expand(B, C)
        neg_w = neg_w.view(1, -1).expand(B, C)
        
        return pos_w, neg_w

    def forward(self, pred_scores, gt_scores, pred_bboxes=None, gt_bboxes=None, fg_mask=None, **kwargs):
        """Forward pass for ECM Loss."""
        original_shape = pred_scores.shape
        
        if pred_scores.dim() == 3:
            B, N, C = pred_scores.shape
            pred_scores = pred_scores.reshape(B * N, C)
            gt_scores = gt_scores.reshape(B * N, C)
        else:
            B = pred_scores.shape[0]
            N = 1
            C = pred_scores.shape[1]
        
        bg_logit = -torch.logsumexp(pred_scores, dim=1, keepdim=True)
        cls_score = torch.cat([pred_scores, bg_logit], dim=1)
        
        is_background = (gt_scores.sum(dim=1, keepdim=True) == 0).float()
        target = torch.cat([gt_scores, is_background], dim=1)
        
        pos_w, neg_w = self.compute_weight(cls_score)
        
        score_exp_pos = (cls_score + pos_w).exp()
        score_exp_neg = (-cls_score + neg_w).exp()
        pred_pos = score_exp_pos / (score_exp_pos + score_exp_neg + 1e-9)
        pred_neg = score_exp_neg / (score_exp_pos + score_exp_neg + 1e-9)
        
        loss_cls = -(
            (pred_pos + 1e-9).log() * target + 
            (pred_neg + 1e-9).log() * (1 - target)
        )
        
        cls_weight = self.detection_cls_weight
        loss_cls = loss_cls * cls_weight
        
        if self.class_weight is not None:
            additional_weight = torch.cat([
                self.class_weight.to(loss_cls.device),
                torch.ones(1, device=loss_cls.device)
            ])
            loss_cls = loss_cls * additional_weight.view(1, -1)
        
        loss_cls = loss_cls[:, :-1]
        
        if len(original_shape) == 3:
            loss_cls = loss_cls.reshape(B, N, C)
        
        if self.reduction == 'mean':
            loss_cls = loss_cls.mean()
        elif self.reduction == 'sum':
            loss_cls = loss_cls.sum()
        elif self.reduction == 'none':
            pass
        else:
            loss_cls = loss_cls.sum() / B
        
        return loss_cls * self.loss_weight

    
class PPLoss(nn.Module):
    """PP-Loss: Size-Aware Prioritization Loss for object detection."""
    
    def __init__(self, num_levels=3, strides=None, reduction='none', weight=None, verbose=True, **kwargs):
        super().__init__()
        self.num_levels = num_levels
        self.strides = strides if strides is not None else [8, 16, 32]
        self.reduction = reduction
        self.class_weight = weight
        
        # mu = 4 * stride (optimal object size for each FPN level)
        # sigma = 2 * stride (spread of the Gaussian prior)
        self.register_buffer('mu', torch.tensor([4.0 * s for s in self.strides]))
        self.register_buffer('sigma', torch.tensor([2.0 * s for s in self.strides]))
        
        if verbose:
            LOGGER.info(f"{colorstr('PP Loss')}: Initialized with {num_levels} FPN levels")
            LOGGER.info(f"{colorstr('PP Loss')}: Strides: {self.strides}, Mu: {self.mu.tolist()}, Sigma: {self.sigma.tolist()}")
    
    def compute_ppf(self, object_sizes, level_idx):
        """Compute Prediction Probability Function (PPF) for objects."""
        mu_l = self.mu[level_idx]
        sigma_l = self.sigma[level_idx]
        ppf = torch.exp(-((object_sizes - mu_l) ** 2) / (2 * sigma_l ** 2))
        return ppf
    
    def compute_weights(self, object_sizes, level_idx, num_levels=3):
        """Compute weights W(s, l) for each object."""
        ppf_current = self.compute_ppf(object_sizes, level_idx)
        
        ppf_sum = torch.zeros_like(ppf_current)
        for i in range(num_levels):
            level_i = torch.full_like(level_idx, i)
            ppf_sum += self.compute_ppf(object_sizes, level_i)
        
        weights = num_levels * ppf_current / (ppf_sum + 1e-8)
        return weights
    
    def apply_weights(self, loss, gt_bboxes, stride_tensor, fg_mask):
        """Apply PP size-aware weights to a per-element loss tensor.

        Args:
            loss: Unreduced loss tensor (batch, anchors, classes).
            gt_bboxes: Ground-truth boxes in xyxy format for foreground anchors.
            stride_tensor: Per-anchor stride values.
            fg_mask: Boolean foreground mask.

        Returns:
            Weighted loss tensor (same shape as input).
        """
        if fg_mask.sum() == 0:
            return loss

        gt_w = gt_bboxes[..., 2] - gt_bboxes[..., 0]
        gt_h = gt_bboxes[..., 3] - gt_bboxes[..., 1]
        object_sizes = torch.sqrt(gt_w * gt_h + 1e-8)

        if stride_tensor.dim() == 2:
            stride_tensor = stride_tensor.unsqueeze(0).expand(loss.shape[0], -1, -1)
        stride_vals = stride_tensor.squeeze(-1)

        level_idx = torch.zeros_like(stride_vals, dtype=torch.long)
        for i, stride in enumerate(self.strides):
            level_idx[stride_vals == stride] = i

        pp_weights = self.compute_weights(object_sizes[fg_mask], level_idx[fg_mask], num_levels=self.num_levels)

        weights_tensor = torch.ones_like(loss)
        weights_tensor[fg_mask] = pp_weights.unsqueeze(-1).expand(-1, loss.shape[-1])
        return loss * weights_tensor

    def forward(self, pred_scores, gt_scores, pred_bboxes=None, gt_bboxes=None, fg_mask=None,
                anchor_points=None, stride_tensor=None, **kwargs):
        """Forward pass for PP Loss."""
        with autocast(enabled=False):
            loss = F.binary_cross_entropy_with_logits(
                pred_scores.float(),
                gt_scores.float(),
                reduction='none',
                weight=self.class_weight
            )

        if gt_bboxes is not None and stride_tensor is not None and fg_mask is not None:
            loss = self.apply_weights(loss, gt_bboxes, stride_tensor, fg_mask)

        if self.reduction == 'mean':
            if fg_mask is not None and fg_mask.sum() > 0:
                return loss.sum() / fg_mask.sum()
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class PPQualityFocalLoss(QualityFocalLoss):
    """Combination of PP-Loss and Quality Focal Loss."""
    
    def __init__(self, num_levels=3, strides=None, weight=None, verbose=True, *args, **kwargs):
        super().__init__(weight=weight, *args, **kwargs)
        self.pp_loss = PPLoss(num_levels=num_levels, strides=strides, reduction='none', weight=weight, verbose=verbose)
    
    def forward(self, pred_scores, gt_scores, pred_bboxes, gt_bboxes, fg_mask,
                anchor_points=None, stride_tensor=None, beta=2.0, *args, **kwargs):
        """Computes PP-weighted Quality Focal Loss."""
        loss = super().forward(pred_scores, gt_scores, pred_bboxes, gt_bboxes, fg_mask, beta=beta)

        if stride_tensor is not None and gt_bboxes is not None and fg_mask is not None:
            loss = self.pp_loss.apply_weights(loss, gt_bboxes, stride_tensor, fg_mask)

        return loss


class VarifocalLoss(nn.Module):
    """Varifocal loss by Zhang et al. https://arxiv.org/abs/2008.13367."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75, weight=None, *args, **kwargs):
        """Initialize the VarifocalLoss class with focusing and balancing parameters."""
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.weight = weight

    def forward(self, pred_scores, gt_scores, gt_target_pos_mask=None, alpha=None, gamma=None, *args, **kwargs):
        """Computes Varifocal loss."""
        alpha = alpha if alpha is not None else self.alpha
        gamma = gamma if gamma is not None else self.gamma
        weight = alpha * (pred_scores.sigmoid() - gt_scores).abs().pow(gamma) * (gt_scores <= 0.0) + gt_scores * (gt_scores > 0.0)
        with autocast(enabled=False):
            return F.binary_cross_entropy_with_logits(pred_scores, gt_scores, reduction='none', weight=self.weight) * weight


class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn()."""

    def __init__(self, gamma: float = 1.5, alpha: float = 0.25):
        """Initialize FocalLoss class with focusing and balancing parameters."""
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))

    def forward(self, pred_scores, gt_scores, gamma=None, alpha=None, *args, **kwargs):
        """Calculates focal loss."""
        gamma = gamma if gamma is not None else self.gamma
        loss = F.binary_cross_entropy_with_logits(pred_scores, gt_scores, reduction="none")
        pred_prob = pred_scores.sigmoid()
        p_t = gt_scores * pred_prob + (1 - gt_scores) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** gamma
        loss *= modulating_factor
        if (self.alpha > 0).any():
            alpha_factor = gt_scores * self.alpha + (1 - gt_scores) * (1 - self.alpha)
            loss *= alpha_factor
        return loss


class BCELoss(nn.BCEWithLogitsLoss):
    """BCE Loss wrapper for compatibility with other loss functions."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, pred_scores, gt_scores, *args, **kwargs):
        return super().forward(pred_scores, gt_scores)


class DFLoss(nn.Module):
    """Criterion class for computing Distribution Focal Loss (DFL)."""

    def __init__(self, reg_max: int = 16) -> None:
        """Initialize the DFL module with regularization maximum."""
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return sum of left and right DFL losses from https://ieeexplore.ieee.org/document/9792391."""
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)


class BboxLoss(nn.Module):
    """Criterion class for computing training losses for bounding boxes.
    
    Enhanced with multiple IoU functions, WiseIoU, and NWD loss support.
    """

    def __init__(
        self,
        reg_max: int = 16,
        iou_loss_fn: str = "ciou",
        nwd_loss: bool = False,
        use_wiseiou: bool = False,
        iou_ratio: float = 0.5,
    ):
        """Initialize the BboxLoss module with regularization maximum and DFL settings.
        
        Args:
            reg_max: The maximum value of the regression distribution.
            iou_loss_fn: The function to use for the IoU loss.
            nwd_loss: If True, use the Wasserstein Distance loss.
            use_wiseiou: If True, use the Wise IoU loss.
            iou_ratio: The ratio of the IoU loss to the Wasserstein Distance loss.
        """
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None
        self.iou_loss_fn = iou_loss_fn.lower()
        self.iou_ratio = iou_ratio
        self.nwd_loss = nwd_loss
        assert self.iou_loss_fn in ('wiou', 'eiou', 'giou', 'diou', 'ciou', 'siou', 'shapeiou', 'piouv1', 'piouv2', 'interpiou'), \
             f"Invalid IoU loss function: {self.iou_loss_fn}"

        self.wiou_loss = WiseIoULoss(
            ltype=self.iou_loss_fn,
            monotonous=False,
            inner_iou=False,
            focaler_iou=False,
        ) if use_wiseiou else None

    def _compute_target_ltrb(
        self, anchor_points: torch.Tensor, target_bboxes: torch.Tensor, reg_max: int | None = None
    ) -> torch.Tensor:
        """Compute target LTRB distances. Override in subclasses for different bbox types."""
        return bbox2dist(anchor_points, target_bboxes, reg_max)

    def _compute_dfl_loss(
        self,
        pred_dist: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        weight: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor = None,
        stride: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute DFL or L1 loss for bounding box regression."""
        if self.dfl_loss:
            target_ltrb = self._compute_target_ltrb(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            return loss_dfl.sum() / target_scores_sum
        elif imgsz is not None and stride is not None:
            # L1 loss fallback when DFL is disabled
            target_ltrb = self._compute_target_ltrb(anchor_points, target_bboxes, None)
            target_ltrb = target_ltrb * stride
            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0]
            pred_dist_scaled = pred_dist * stride
            pred_dist_scaled[..., 0::2] /= imgsz[1]
            pred_dist_scaled[..., 1::2] /= imgsz[0]
            loss_dfl = (
                F.l1_loss(pred_dist_scaled[fg_mask], target_ltrb[fg_mask], reduction="none").mean(-1, keepdim=True) * weight
            )
            return loss_dfl.sum() / target_scores_sum
        else:
            return torch.tensor(0.0, device=pred_dist.device)

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor = None,
        stride: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute IoU and DFL losses for bounding boxes."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        
        if self.wiou_loss:
            iou = self.wiou_loss(
                pred_bboxes[fg_mask],
                target_bboxes[fg_mask],
                ret_iou=False,
                ratio=0.7,
                d=0.0,
                u=0.95,
            ).unsqueeze(-1)
        else:
            iou = 1.0 - bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, **{self.iou_loss_fn: True})

        loss_iou = (iou * weight).sum() / target_scores_sum

        if self.nwd_loss:
            nwd = wasserstein_loss(pred_bboxes[fg_mask], target_bboxes[fg_mask])
            nwd_loss = ((1.0 - nwd) * weight).sum() / target_scores_sum
            loss_iou = self.iou_ratio * loss_iou + (1 - self.iou_ratio) * nwd_loss

        loss_dfl = self._compute_dfl_loss(
            pred_dist, anchor_points, target_bboxes, weight, target_scores_sum, fg_mask, imgsz, stride
        )

        return loss_iou, loss_dfl


class RotatedBboxLoss(BboxLoss):
    """Criterion class for computing training losses for rotated bounding boxes."""

    def __init__(self, reg_max: int):
        """Initialize the RotatedBboxLoss module with regularization maximum and DFL settings."""
        super().__init__(reg_max)

    def _compute_target_ltrb(
        self, anchor_points: torch.Tensor, target_bboxes: torch.Tensor, reg_max: int | None = None
    ) -> torch.Tensor:
        """Compute target LTRB distances for rotated bboxes."""
        return rbox2dist(target_bboxes[..., :4], anchor_points, target_bboxes[..., 4:5], reg_max=reg_max)

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor = None,
        stride: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute IoU and DFL losses for rotated bounding boxes."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        loss_dfl = self._compute_dfl_loss(
            pred_dist, anchor_points, target_bboxes, weight, target_scores_sum, fg_mask, imgsz, stride
        )

        return loss_iou, loss_dfl


class MultiChannelDiceLoss(nn.Module):
    """Criterion class for computing multi-channel Dice losses."""

    def __init__(self, smooth: float = 1e-6, reduction: str = "mean"):
        """Initialize MultiChannelDiceLoss with smoothing and reduction options."""
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate multi-channel Dice loss between predictions and targets."""
        assert pred.size() == target.size(), "the size of predict and target must be equal."

        pred = pred.sigmoid()
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice
        dice_loss = dice_loss.mean(dim=1)

        if self.reduction == "mean":
            return dice_loss.mean()
        elif self.reduction == "sum":
            return dice_loss.sum()
        else:
            return dice_loss


class BCEDiceLoss(nn.Module):
    """Criterion class for computing combined BCE and Dice losses."""

    def __init__(self, weight_bce: float = 0.5, weight_dice: float = 0.5):
        """Initialize BCEDiceLoss with BCE and Dice weight factors."""
        super().__init__()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = MultiChannelDiceLoss(smooth=1)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate combined BCE and Dice loss between predictions and targets."""
        _, _, mask_h, mask_w = pred.shape
        if tuple(target.shape[-2:]) != (mask_h, mask_w):
            target = F.interpolate(target, (mask_h, mask_w), mode="nearest")
        return self.weight_bce * self.bce(pred, target) + self.weight_dice * self.dice(pred, target)


class RLELoss(nn.Module):
    """Residual Log-Likelihood Estimation Loss.

    References:
        https://arxiv.org/abs/2107.11291
        https://github.com/open-mmlab/mmpose/blob/main/mmpose/models/losses/regression_loss.py
    """

    def __init__(self, use_target_weight: bool = True, size_average: bool = True, residual: bool = True):
        """Initialize RLELoss with target weight and residual options."""
        super().__init__()
        self.size_average = size_average
        self.use_target_weight = use_target_weight
        self.residual = residual

    def forward(
        self, sigma: torch.Tensor, log_phi: torch.Tensor, error: torch.Tensor, target_weight: torch.Tensor = None
    ) -> torch.Tensor:
        """Calculate RLE loss."""
        log_sigma = torch.log(sigma)
        loss = log_sigma - log_phi.unsqueeze(1)

        if self.residual:
            loss += torch.log(sigma * 2) + torch.abs(error)

        if self.use_target_weight:
            assert target_weight is not None, "'target_weight' should not be None when 'use_target_weight' is True."
            if target_weight.dim() == 1:
                target_weight = target_weight.unsqueeze(1)
            loss *= target_weight

        if self.size_average:
            loss /= len(loss)

        return loss.sum()


class KeypointLoss(nn.Module):
    """Criterion class for computing keypoint losses."""

    def __init__(self, sigmas: torch.Tensor) -> None:
        """Initialize the KeypointLoss class with keypoint sigmas."""
        super().__init__()
        self.register_buffer("sigmas", sigmas if isinstance(sigmas, torch.Tensor) else torch.tensor(sigmas))

    def forward(
        self, pred_kpts: torch.Tensor, gt_kpts: torch.Tensor, kpt_mask: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """Calculate keypoint loss factor and Euclidean distance loss for keypoints."""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()


class v8DetectionLoss:
    """Criterion class for computing training losses for YOLOv8 object detection.
    
    Enhanced with multi-task support (nc as list, multiple classification heads).
    """

    def __init__(
        self,
        model: nn.Module,
        tal_topk: int = 10,
        tal_topk2: int | None = None,
        clf_loss_weights: list[list[float]] | None = None,
        task_loss_weights: list[float] | None = None,
        clf_loss_fn: str = "qfl",
        iou_loss_fn: str = "ciou",
        nwd_loss: bool = False,
        use_wiseiou: bool = False,
        iou_ratio: float = 0.5,
        dependency_loss: bool = False,
        child_parent_map: dict | None = None,
        task_schema: dict | None = None,
        ignore_class: dict | None = None,
        verbose: bool = True,
    ):  # model must be de-paralleled
        """Initialize v8DetectionLoss with model parameters and task-aligned assignment settings."""
        device = next(model.parameters()).device
        h = model.args
        m = model.model[-1]  # Detect() module
        self.nc: list[int] = m.nc if isinstance(m.nc, list) else [m.nc]
        self.n_tasks = len(self.nc)
        assert self.n_tasks >= 1, "nc must be at least 1."
        self.task_schema = task_schema or getattr(model, "task_schema", None) or {}
        self.main_head = int(self.task_schema.get("main_head", getattr(model, "main_head", 0)))
        if not 0 <= self.main_head < self.n_tasks:
            raise ValueError(f"main_head={self.main_head} is outside model head range 0-{self.n_tasks - 1}")
        self.main_offset = sum(self.nc[:self.main_head])
        raw_parents = self.task_schema.get("dependency_parent_heads", self.task_schema.get("hierarchy_parent_heads"))
        self.dependency_parent_heads = (
            [int(p) for p in raw_parents]
            if raw_parents is not None
            else [-1] + list(range(self.n_tasks - 1))
        )
        raw_ignore = ignore_class if ignore_class is not None else self.task_schema.get("ignore_class", {})
        self.ignore_class = {int(k): {int(c) for c in v} for k, v in (raw_ignore or {}).items()}
        
        # Per-task classification loss weights
        self.clf_loss_weights = [
            torch.tensor(
                clf_loss_weights[i] if clf_loss_weights is not None else [1.0] * self.nc[i],
                device=device,
            )
            for i in range(self.n_tasks)
        ]
        
        # Initialize classification loss functions for each task
        cls_losses = []
        for i in range(self.n_tasks):
            cls_losses.append(self._build_cls_loss(clf_loss_fn, self.clf_loss_weights[i], self.nc[i], m, verbose))
        self.cls_losses = nn.ModuleList(cls_losses)
        self.cls_losses.to(device)  # Move loss modules (buffers) to model device

        # Also keep standard BCE for compatibility
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

        self.hyp = h
        self.stride = m.stride
        self.no = sum(self.nc) + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(
            topk=tal_topk,
            num_classes=self.nc[self.main_head],
            alpha=0.5,
            beta=3.0,
            stride=self.stride.tolist() if hasattr(self.stride, 'tolist') else self.stride,
            topk2=tal_topk2,
            iou_loss_fn=iou_loss_fn,
            assign_metric=getattr(h, "assign_metric", "iou"),
            assign_nwd_lambda=getattr(h, "assign_nwd_lambda", 0.0),
            assign_nwd_small_thr=getattr(h, "assign_nwd_small_thr", 0.0),
        )

        self.bbox_loss = BboxLoss(
            reg_max=m.reg_max,
            iou_loss_fn=iou_loss_fn,
            nwd_loss=nwd_loss,
            use_wiseiou=use_wiseiou,
            iou_ratio=iou_ratio,
        ).to(device)

        if verbose:
            loss_parts = [f"{clf_loss_fn} cls loss", f"{iou_loss_fn} iou loss"]
            if dependency_loss:
                loss_parts.append("dependency loss")
            if getattr(m, "embed_dim", 0) > 0:
                loss_parts.append("ReID (metric learning) loss")
            LOGGER.info(f"{colorstr('Using losses')}: {' & '.join(loss_parts)}")

        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

        # Per-task cls loss weights (higher weight = more gradient for that task).
        # Normalized so sum == n_tasks to preserve total cls loss magnitude.
        if task_loss_weights is not None and self.n_tasks > 1:
            if (
                self.task_schema
                and len(task_loss_weights) == len(self.task_schema.get("semantic_tasks", []))
                and len(task_loss_weights) != self.n_tasks
            ):
                semantic_tasks = list(self.task_schema["semantic_tasks"])
                task_loss_weights = [
                    task_loss_weights[semantic_tasks.index(flat_info["task"])]
                    for flat_info in self.task_schema["flat_to_semantic"]
                ]
            assert len(task_loss_weights) == self.n_tasks, (
                f"task_loss_weights length {len(task_loss_weights)} != n_tasks {self.n_tasks}"
            )
            tw_sum = sum(task_loss_weights)
            self.task_loss_weights = [w * self.n_tasks / tw_sum for w in task_loss_weights]
            if verbose:
                LOGGER.info(
                    f"{colorstr('Task Loss Weights')}: raw={task_loss_weights}, "
                    f"normalized={[round(w, 3) for w in self.task_loss_weights]}"
                )
        else:
            self.task_loss_weights = [1.0] * self.n_tasks

        # Hierarchical dependency loss
        self.dependency_loss = dependency_loss and self.n_tasks > 1
        self.child_parent_maps = None
        if self.dependency_loss:
            if not child_parent_map:
                LOGGER.warning(
                    f"{colorstr('Dependency Loss')}: 'child_parent_map' not provided in data YAML. "
                    "Dependency loss will be disabled."
                )
                self.dependency_loss = False
            else:
                self._load_child_parent_map(child_parent_map, device)

        self.embed_dim = getattr(m, "embed_dim", 0)
        self.has_embed = self.embed_dim > 0
        if self.has_embed:
            self.embed_loss = MetricLearningLoss().to(device)
        # Fixed-order loss layout: box, cls, dfl, [dep], [reid]
        self.loss_names = ["box", "cls", "dfl"]
        if self.dependency_loss:
            self.loss_names.append("dep")
        if self.has_embed:
            self.loss_names.append("reid")
        self.n_losses = len(self.loss_names)
        # Slot indices (-1 means "not present")
        self.box_idx = 0
        self.cls_idx = 1
        self.dfl_idx = 2
        self.dep_idx = self.loss_names.index("dep") if self.dependency_loss else -1
        self.reid_idx = self.loss_names.index("reid") if self.has_embed else -1

        disable_dynamo(self.__class__)

    def _load_child_parent_map(self, raw_map: dict, device: torch.device):
        """Preprocess the child-parent class mapping for dependency loss.

        ``raw_map`` maps hierarchy levels to {child_class_idx: parent_class_idx}.
        Level 0 has no parent. Level k maps its classes to level k-1 classes.
        Keys may be ints or strings (e.g. when coming from YAML/JSON).

        Internally builds per-level tensors for efficient vectorized computation:
        child_parent_maps[level] is a tensor of shape (nc[level],) where
        entry i = parent class index in level-1 for child class i at this level.
        """
        # Normalize top-level keys to int for uniform lookup
        norm_map = {int(k): v for k, v in raw_map.items()}

        self.child_parent_maps = {}
        expected_levels = (
            [i for i, parent in enumerate(self.dependency_parent_heads) if int(parent) >= 0]
            if self.task_schema
            else range(1, self.n_tasks)
        )
        for level in expected_levels:
            if level not in norm_map:
                LOGGER.warning(
                    f"{colorstr('Dependency Loss')}: No mapping for level {level} in child_parent_map."
                )
                continue
            level_map = norm_map[level]
            parent_indices = torch.full((self.nc[level],), -1, dtype=torch.long, device=device)
            for child_key, parent_idx in level_map.items():
                child_idx = int(child_key)
                if child_idx < self.nc[level]:
                    parent_indices[child_idx] = int(parent_idx)
            self.child_parent_maps[level] = parent_indices

    def _compute_dependency_penalty(
        self,
        pred_scores: torch.Tensor,
        target_scores: torch.Tensor,
        parent_target_scores: torch.Tensor,
        offset: int,
        task_idx: int,
    ) -> torch.Tensor:
        """Compute hierarchical dependency penalty for a given task level.

        Penalizes false-positive predictions that are inconsistent with the parent level.
        A prediction is inconsistent if it's a false positive at level k AND its corresponding
        parent class at level k-1 is also not a true positive.

        Args:
            pred_scores: Full predicted scores (bs, num_anchors, sum(nc)) - raw logits.
            target_scores: Full target scores (bs, num_anchors, nc[task]) for current task.
            offset: Start index of current task in the score dimension.
            task_idx: Current hierarchy level index.
            prev_offset: Start index of parent task in the score dimension.

        Returns:
            Scalar penalty value.
        """
        if task_idx == 0 or task_idx not in self.child_parent_maps or parent_target_scores is None:
            return torch.tensor(0.0, device=pred_scores.device)

        parent_map = self.child_parent_maps[task_idx]  # (nc_child,) -> parent cls in level-1
        nc_child = self.nc[task_idx]

        # Child confidences for current hierarchy level.
        child_preds = pred_scores[..., offset : offset + nc_child].sigmoid()
        child_is_tp = target_scores > 0
        fp_conf = child_preds * (~child_is_tp).float()

        # hyolo-style: only consider anchors that are active in both adjacent levels.
        common_anchor = (target_scores.sum(-1, keepdim=True) > 0) & (parent_target_scores.sum(-1, keepdim=True) > 0)
        fp_conf = fp_conf * common_anchor.float()

        # Filter tiny confidences.
        fp_conf = fp_conf * (fp_conf > 0.001).float()
        if fp_conf.sum() == 0:
            return torch.tensor(0.0, device=pred_scores.device)

        # Map each child class to its parent class and keep only inconsistent FPs
        # (i.e. mapped parent class is NOT a TP at this anchor).
        valid_child = parent_map >= 0
        safe_parent_map = parent_map.clamp(min=0)  # avoid gather OOB for invalid entries

        # parent_tp_mapped: [B, A, nc_child], value 1 if mapped parent is TP, else 0.
        parent_tp = (parent_target_scores > 0).float()
        gather_idx = safe_parent_map.view(1, 1, -1).expand(parent_tp.shape[0], parent_tp.shape[1], -1)
        parent_tp_mapped = torch.gather(parent_tp, 2, gather_idx)
        if (~valid_child).any():
            parent_tp_mapped[..., ~valid_child] = 0.0

        inconsistent_fp = fp_conf * (1.0 - parent_tp_mapped)
        fp_count = (inconsistent_fp > 0).float().sum()
        if fp_count > 0:
            return inconsistent_fp.sum() / fp_count
        return torch.tensor(0.0, device=pred_scores.device)

    @staticmethod
    def _build_cls_loss(name: str, weight, nc: int, detect_module, verbose: bool = True):
        """Build a single classification loss module by name.

        Args:
            name: Loss function name (bce, vfl, qfl, ecm, pp, ppqfl, focal).
            weight: Per-class weight tensor for this task.
            nc: Number of classes for this task.
            detect_module: The Detect head module (used by pp/ppqfl for stride info).
            verbose: Whether to log details.
        """
        if name == "bce":
            return BCELoss(reduction="none", weight=weight)
        if name == "vfl":
            return VarifocalLoss(weight=weight)
        if name == "qfl":
            return QualityFocalLoss(weight=weight)
        if name == "ecm":
            return EffectiveClassMarginLoss(num_classes=nc, reduction="none", weight=weight)
        if name == "pp":
            return PPLoss(
                num_levels=len(detect_module.stride),
                strides=detect_module.stride.tolist(),
                reduction="none",
                weight=weight,
                verbose=verbose,
            )
        if name == "ppqfl":
            return PPQualityFocalLoss(
                num_levels=len(detect_module.stride),
                strides=detect_module.stride.tolist(),
                weight=weight,
                verbose=verbose,
            )
        if name == "focal":
            return FocalLoss()
        raise ValueError(f"Unknown classification loss function: {name}")

    def preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor) -> torch.Tensor:
        """Preprocess targets by converting to tensor format and scaling coordinates."""
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device, dtype=targets.dtype)
        else:
            i = targets[:, 0]
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device, dtype=targets.dtype)
            for j in range(batch_size):
                matches = i == j
                if n := matches.sum():
                    out[j, :n] = targets[matches, 1:]
            out[..., -4:] = xywh2xyxy(out[..., -4:].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points: torch.Tensor, pred_dist: torch.Tensor) -> torch.Tensor:
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def get_assigned_targets_and_loss(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> tuple:
        """Calculate the sum of the loss for box, cls, dfl, and optionally reid."""
        loss = torch.zeros(self.n_losses, device=self.device)
        pred_distri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"], batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((self.n_tasks, 4), dim=2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        # Foreground assignment. When a ReID head is enabled we also ask the assigner
        # for a "wide" positive mask covering every anchor whose center is inside any
        # GT (not just the topk-aligned ones); the embedding loss uses that to get many
        # same-tag positive pairs per object.
        assigner_out = self.assigner(
            pred_scores[..., self.main_offset : self.main_offset + self.nc[self.main_head]].detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels[..., self.main_head, None],
            gt_bboxes,
            mask_gt,
            return_reid=self.has_embed,
        )
        if self.has_embed:
            norm_align_metric, fg_mask, target_gt_idx, reid_fg_mask, reid_target_gt_idx = assigner_out
        else:
            norm_align_metric, fg_mask, target_gt_idx = assigner_out
            reid_fg_mask = None
            reid_target_gt_idx = None

        target_bboxes = self.assigner.get_bboxes(gt_bboxes, target_gt_idx, fg_mask)
        target_scores_sum, offset = max(norm_align_metric.sum(), 1), 0

        # Cls loss - iterate over each classification task/head
        target_scores_by_task = [None] * self.n_tasks
        for task_idx, (cls_loss_fn, n_cls_task) in enumerate(zip(self.cls_losses, self.nc)):
            pred_scores_task = pred_scores[..., offset: offset + n_cls_task]

            target_labels_task, target_scores_task = self.assigner.get_scores(
                gt_labels=gt_labels[..., task_idx, None],
                target_gt_idx=target_gt_idx,
                fg_mask=fg_mask,
                num_classes=self.nc[task_idx],
            )

            target_scores_task = target_scores_task * norm_align_metric
            ignored_classes = self.ignore_class.get(task_idx)
            if ignored_classes:
                ignored = torch.zeros_like(target_labels_task, dtype=torch.bool)
                for cls_idx in ignored_classes:
                    ignored |= target_labels_task == cls_idx
                valid_cls_mask = (~ignored).unsqueeze(-1)
                target_scores_task = target_scores_task * valid_cls_mask

            task_cls_loss_raw = cls_loss_fn(
                pred_scores=pred_scores_task,
                gt_scores=target_scores_task,
                pred_bboxes=pred_bboxes,
                # target_bboxes are in image pixels; PPLoss mu/sigma are pixel-sized stride priors.
                gt_bboxes=target_bboxes,
                fg_mask=fg_mask,
                anchor_points=anchor_points,
                stride_tensor=stride_tensor,
            )
            if ignored_classes:
                task_cls_loss_raw = task_cls_loss_raw * valid_cls_mask
            task_cls_loss = task_cls_loss_raw.sum() / target_scores_sum
            target_scores_by_task[task_idx] = target_scores_task

            # Hierarchical dependency penalty (levels > 0) in its own slot
            parent_idx = (
                int(self.dependency_parent_heads[task_idx])
                if task_idx < len(self.dependency_parent_heads)
                else task_idx - 1
            )
            if self.dependency_loss and parent_idx >= 0 and task_idx in self.child_parent_maps:
                dep_penalty = self._compute_dependency_penalty(
                    pred_scores,
                    target_scores_task,
                    target_scores_by_task[parent_idx],
                    offset,
                    task_idx,
                )
                # Clamp penalty so it cannot exceed base cls loss magnitude
                dep_term = (dep_penalty * self.hyp.dep).clamp(max=task_cls_loss.detach())
                loss[self.dep_idx] += dep_term * self.task_loss_weights[task_idx]

            loss[self.cls_idx] += task_cls_loss * self.task_loss_weights[task_idx]

            offset += n_cls_task

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[self.box_idx], loss[self.dfl_idx] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes,
                norm_align_metric,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )

        # ReID embedding loss. SupCon over the wide TAL positive set: every anchor
        # inside a GT contributes with that GT's tag. Many same-tag pairs per object
        # → strong contrastive signal even with unique-per-instance tags. Embeddings
        # are NOT pre-normalized here; MetricLearningLoss handles normalization.
        if self.has_embed and "embeds" in preds and reid_fg_mask is not None and reid_fg_mask.sum():
            pred_embeds = preds["embeds"].permute(0, 2, 1).contiguous().float()
            tags_batch = batch.get("tags")

            if tags_batch is not None and len(tags_batch):
                tags = tags_batch.to(self.device).long().view(-1)
                batch_idx_t = batch["batch_idx"].to(self.device).long()
                # Use n_max_boxes from the assigner so the flat indexing matches
                # target_gt_idx (= idx + b * n_max_boxes).
                n_max_boxes = self.assigner.n_max_boxes
                gt_tags = torch.zeros(batch_size, n_max_boxes, device=self.device, dtype=torch.long)
                for j in range(batch_size):
                    m = batch_idx_t == j
                    if m.any():
                        gt_tags[j, : m.sum()] = tags[m]

                gt_tags_flat = gt_tags.view(-1)
                reid_target_gt_idx_fg = reid_target_gt_idx[reid_fg_mask].long()
                valid = (reid_target_gt_idx_fg >= 0) & (reid_target_gt_idx_fg < gt_tags_flat.numel())
                if valid.any():
                    loss[self.reid_idx] = self.embed_loss(
                        pred_embeds[reid_fg_mask][valid],
                        gt_tags_flat[reid_target_gt_idx_fg[valid]],
                    )

        loss[self.box_idx] *= self.hyp.box
        loss[self.cls_idx] *= self.hyp.cls / len(self.nc)
        loss[self.dfl_idx] *= self.hyp.dfl
        if self.dependency_loss:
            # dep accumulates only over configured hierarchy edges, not auxiliary heads.
            # to keep the per-level magnitude on the same scale as cls.
            loss[self.dep_idx] *= self.hyp.dep / max(len(self.child_parent_maps or {}), 1)
        if self.has_embed:
            loss[self.reid_idx] *= self.hyp.reid

        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            loss.detach(),
        )

    def parse_output(
        self, preds: dict[str, torch.Tensor] | tuple[torch.Tensor, dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        """Parse model predictions to extract features."""
        if isinstance(preds, tuple):
            return preds[1]
        if isinstance(preds, list):
            # Backward compatibility for callers that pass raw per-level tensors.
            # Each tensor shape: (B, reg_max*4 + sum(nc), H, W)
            feats = preds
            bs = feats[0].shape[0]
            box_ch = self.reg_max * 4
            boxes = torch.cat([f[:, :box_ch].view(bs, box_ch, -1) for f in feats], dim=-1)
            scores = torch.cat([f[:, box_ch:].view(bs, -1, f.shape[-2] * f.shape[-1]) for f in feats], dim=-1)
            return {"boxes": boxes, "scores": scores, "feats": feats}
        return preds

    def __call__(
        self,
        preds: dict[str, torch.Tensor] | tuple[torch.Tensor, dict[str, torch.Tensor]],
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        return self.loss(self.parse_output(preds), batch)

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """A wrapper for get_assigned_targets_and_loss."""
        batch_size = preds["boxes"].shape[0]
        loss, loss_detach = self.get_assigned_targets_and_loss(preds, batch)[1:]
        return loss.sum() * batch_size, loss_detach


class v8SegmentationLoss(v8DetectionLoss):
    """Criterion class for computing training losses for YOLOv8 segmentation."""

    def __init__(
        self,
        model: nn.Module,
        tal_topk: int = 10,
        tal_topk2: int | None = None,
        clf_loss_weights: list[list[float]] | None = None,
        clf_loss_fn: str = "bce",
        iou_loss_fn: str = "ciou",
        nwd_loss: bool = False,
        use_wiseiou: bool = False,
        iou_ratio: float = 0.5,
        verbose: bool = True,
    ):
        """Initialize the v8SegmentationLoss class with model parameters and mask overlap setting."""
        super().__init__(
            model,
            tal_topk=tal_topk,
            tal_topk2=tal_topk2,
            clf_loss_weights=clf_loss_weights,
            clf_loss_fn=clf_loss_fn,
            iou_loss_fn=iou_loss_fn,
            nwd_loss=nwd_loss,
            use_wiseiou=use_wiseiou,
            iou_ratio=iou_ratio,
            verbose=verbose,
        )
        self.overlap = model.args.overlap_mask
        self.bcedice_loss = BCEDiceLoss(weight_bce=0.5, weight_dice=0.5)

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate and return the combined loss for detection and segmentation."""
        pred_masks, proto = preds["mask_coefficient"].permute(0, 2, 1).contiguous(), preds["proto"]
        loss = torch.zeros(5, device=self.device)  # box, seg, cls, dfl, semseg
        if isinstance(proto, tuple) and len(proto) == 2:
            proto, pred_semseg = proto
        else:
            pred_semseg = None
            
        (fg_mask, target_gt_idx, target_bboxes, _, stride_tensor), det_loss, _ = self.get_assigned_targets_and_loss(
            preds, batch
        )
        loss[0], loss[2], loss[3] = det_loss[0], det_loss[1], det_loss[2]

        batch_size, _, mask_h, mask_w = proto.shape
        if fg_mask.sum():
            masks = batch["masks"].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]

            imgsz = (
                torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=pred_masks.dtype) * self.stride[0]
            )
            loss[1] = self.calculate_segmentation_loss(
                fg_mask,
                masks,
                target_gt_idx,
                target_bboxes * stride_tensor,
                batch["batch_idx"].view(-1, 1),
                proto,
                pred_masks,
                imgsz,
            )
            if pred_semseg is not None:
                sem_masks = batch["sem_masks"].to(self.device)
                sem_masks = F.one_hot(sem_masks.long(), num_classes=self.nc[0]).permute(0, 3, 1, 2).float()

                if self.overlap:
                    mask_zero = masks == 0
                    sem_masks[mask_zero.unsqueeze(1).expand_as(sem_masks)] = 0
                else:
                    batch_idx = batch["batch_idx"].view(-1)
                    for i in range(batch_size):
                        instance_mask_i = masks[batch_idx == i]
                        if len(instance_mask_i) == 0:
                            continue
                        sem_masks[i, :, instance_mask_i.sum(dim=0) == 0] = 0

                loss[4] = self.bcedice_loss(pred_semseg, sem_masks)
                loss[4] *= self.hyp.box
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()
            if pred_semseg is not None:
                loss[4] += (pred_semseg * 0).sum()

        loss[1] *= self.hyp.box
        return loss.sum() * batch_size, loss.detach()

    @staticmethod
    def single_mask_loss(
        gt_mask: torch.Tensor, pred: torch.Tensor, proto: torch.Tensor, xyxy: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """Compute the instance segmentation loss for a single image."""
        pred_mask = torch.einsum("in,nhw->ihw", pred, proto)
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).sum()

    def calculate_segmentation_loss(
        self,
        fg_mask: torch.Tensor,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        target_bboxes: torch.Tensor,
        batch_idx: torch.Tensor,
        proto: torch.Tensor,
        pred_masks: torch.Tensor,
        imgsz: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate the loss for instance segmentation."""
        _, _, mask_h, mask_w = proto.shape
        loss = 0

        target_bboxes_normalized = target_bboxes / (imgsz[[1, 0, 1, 0]] + 1e-8)
        marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2).clamp_(min=1e-6)
        mxyxy = target_bboxes_normalized * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)
        n_max_boxes = self.assigner.n_max_boxes

        for i, single_i in enumerate(zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea, masks)):
            fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i, marea_i, masks_i = single_i
            if fg_mask_i.any():
                # target_gt_idx is flattened across the batch by TaskAlignedAssigner.prepare_targets().
                # Masks are indexed locally within each image, so convert global indices back to local ids.
                mask_idx = target_gt_idx_i[fg_mask_i] % n_max_boxes
                if self.overlap:
                    gt_mask = masks_i == (mask_idx + 1).view(-1, 1, 1)
                    gt_mask = gt_mask.float()
                else:
                    gt_mask = masks[batch_idx.view(-1) == i][mask_idx]

                loss += self.single_mask_loss(
                    gt_mask, pred_masks_i[fg_mask_i], proto_i, mxyxy_i[fg_mask_i], marea_i[fg_mask_i]
                )
            else:
                loss += (proto * 0).sum() + (pred_masks * 0).sum()

        return loss / fg_mask.sum()


class v8PoseLoss(v8DetectionLoss):
    """Criterion class for computing training losses for YOLOv8 pose estimation."""

    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None, verbose: bool = True, **kwargs):
        """Initialize v8PoseLoss with model parameters and keypoint-specific loss functions."""
        super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2, verbose=verbose, **kwargs)
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the total loss and detach it for pose estimation."""
        pred_kpts = preds["kpts"].permute(0, 2, 1).contiguous()
        loss = torch.zeros(5, device=self.device)  # box, kpt_location, kpt_visibility, cls, dfl
        (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor), det_loss, _ = (
            self.get_assigned_targets_and_loss(preds, batch)
        )
        loss[0], loss[3], loss[4] = det_loss[0], det_loss[1], det_loss[2]

        batch_size = pred_kpts.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=pred_kpts.dtype) * self.stride[0]

        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))

        if fg_mask.sum():
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            loss[1], loss[2] = self.calculate_keypoints_loss(
                fg_mask,
                target_gt_idx,
                keypoints,
                batch["batch_idx"].view(-1, 1),
                stride_tensor,
                target_bboxes,
                pred_kpts,
            )

        loss[1] *= self.hyp.pose
        loss[2] *= self.hyp.kobj

        return loss.sum() * batch_size, loss.detach()

    @staticmethod
    def kpts_decode(anchor_points: torch.Tensor, pred_kpts: torch.Tensor) -> torch.Tensor:
        """Decode predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y

    def _select_target_keypoints(
        self,
        keypoints: torch.Tensor,
        batch_idx: torch.Tensor,
        target_gt_idx: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """Select target keypoints for each anchor based on batch index and target ground truth index."""
        batch_idx = batch_idx.flatten()
        batch_size = len(masks)

        max_kpts = torch.unique(batch_idx, return_counts=True)[1].max()
        batched_keypoints = torch.zeros(
            (batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]), device=keypoints.device
        )

        for i in range(batch_size):
            keypoints_i = keypoints[batch_idx == i]
            batched_keypoints[i, : keypoints_i.shape[0]] = keypoints_i

        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)
        selected_keypoints = batched_keypoints.gather(
            1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2])
        )

        return selected_keypoints

    def calculate_keypoints_loss(
        self,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        keypoints: torch.Tensor,
        batch_idx: torch.Tensor,
        stride_tensor: torch.Tensor,
        target_bboxes: torch.Tensor,
        pred_kpts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the keypoints loss for the model."""
        selected_keypoints = self._select_target_keypoints(keypoints, batch_idx, target_gt_idx, masks)
        selected_keypoints[..., :2] /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = 0
        kpts_obj_loss = 0

        if masks.any():
            target_bboxes /= stride_tensor
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)

            if pred_kpt.shape[-1] == 3:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())

        return kpts_loss, kpts_obj_loss


class PoseLoss26(v8PoseLoss):
    """Criterion class for computing training losses for YOLOv8 pose estimation with RLE loss support."""

    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None, **kwargs):
        """Initialize PoseLoss26 with model parameters and keypoint-specific loss functions including RLE loss."""
        super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2, **kwargs)
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]
        self.rle_loss = None
        self.flow_model = model.model[-1].flow_model if hasattr(model.model[-1], "flow_model") else None
        if self.flow_model is not None:
            self.rle_loss = RLELoss(use_target_weight=True).to(self.device)
            self.target_weights = (
                torch.from_numpy(RLE_WEIGHT).to(self.device) if is_pose else torch.ones(nkpt, device=self.device)
            )

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the total loss and detach it for pose estimation."""
        pred_kpts = preds["kpts"].permute(0, 2, 1).contiguous()
        loss = torch.zeros(6 if self.rle_loss else 5, device=self.device)
        (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor), det_loss, _ = (
            self.get_assigned_targets_and_loss(preds, batch)
        )
        loss[0], loss[3], loss[4] = det_loss[0], det_loss[1], det_loss[2]

        batch_size = pred_kpts.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=pred_kpts.dtype) * self.stride[0]

        pred_kpts = pred_kpts.view(batch_size, -1, *self.kpt_shape)

        if self.rle_loss and preds.get("kpts_sigma", None) is not None:
            pred_sigma = preds["kpts_sigma"].permute(0, 2, 1).contiguous()
            pred_sigma = pred_sigma.view(batch_size, -1, self.kpt_shape[0], 2)
            pred_kpts = torch.cat([pred_kpts, pred_sigma], dim=-1)

        pred_kpts = self.kpts_decode(anchor_points, pred_kpts)

        if fg_mask.sum():
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            keypoints_loss = self.calculate_keypoints_loss(
                fg_mask,
                target_gt_idx,
                keypoints,
                batch["batch_idx"].view(-1, 1),
                stride_tensor,
                target_bboxes,
                pred_kpts,
            )
            loss[1] = keypoints_loss[0]
            loss[2] = keypoints_loss[1]
            if self.rle_loss is not None and len(keypoints_loss) > 2:
                loss[5] = keypoints_loss[2]

        loss[1] *= self.hyp.pose
        loss[2] *= self.hyp.kobj
        if self.rle_loss is not None:
            loss[5] *= getattr(self.hyp, 'rle', 1.0)

        return loss.sum() * batch_size, loss.detach()

    @staticmethod
    def kpts_decode(anchor_points: torch.Tensor, pred_kpts: torch.Tensor) -> torch.Tensor:
        """Decode predicted keypoints to image coordinates (without offset)."""
        y = pred_kpts.clone()
        y[..., 0] += anchor_points[:, [0]]
        y[..., 1] += anchor_points[:, [1]]
        return y

    def calculate_rle_loss(self, pred_kpt: torch.Tensor, gt_kpt: torch.Tensor, kpt_mask: torch.Tensor) -> torch.Tensor:
        """Calculate the RLE (Residual Log-likelihood Estimation) loss for keypoints."""
        pred_kpt_visible = pred_kpt[kpt_mask]
        gt_kpt_visible = gt_kpt[kpt_mask]
        pred_coords = pred_kpt_visible[:, 0:2]
        pred_sigma = pred_kpt_visible[:, -2:]
        gt_coords = gt_kpt_visible[:, 0:2]

        target_weights = self.target_weights.unsqueeze(0).repeat(kpt_mask.shape[0], 1)
        target_weights = target_weights[kpt_mask]

        pred_sigma = pred_sigma.sigmoid()
        error = (pred_coords - gt_coords) / (pred_sigma + 1e-9)

        valid_mask = ~(torch.isnan(error) | torch.isinf(error)).any(dim=-1)
        if not valid_mask.any():
            return torch.tensor(0.0, device=pred_kpt.device)

        error = error[valid_mask]
        error = error.clamp(-100, 100)
        pred_sigma = pred_sigma[valid_mask]
        target_weights = target_weights[valid_mask]

        log_phi = self.flow_model.log_prob(error)
        return self.rle_loss(pred_sigma, log_phi, error, target_weights)

    def calculate_keypoints_loss(
        self,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        keypoints: torch.Tensor,
        batch_idx: torch.Tensor,
        stride_tensor: torch.Tensor,
        target_bboxes: torch.Tensor,
        pred_kpts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate the keypoints loss for the model with RLE support."""
        selected_keypoints = self._select_target_keypoints(keypoints, batch_idx, target_gt_idx, masks)
        selected_keypoints[..., :2] /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = 0
        kpts_obj_loss = 0
        rle_loss = 0

        if masks.any():
            target_bboxes /= stride_tensor
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)

            if self.rle_loss is not None and (pred_kpt.shape[-1] == 4 or pred_kpt.shape[-1] == 5):
                rle_loss = self.calculate_rle_loss(pred_kpt, gt_kpt, kpt_mask)
            if pred_kpt.shape[-1] == 3 or pred_kpt.shape[-1] == 5:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())

        return kpts_loss, kpts_obj_loss, rle_loss


class v8ClassificationLoss:
    """Criterion class for computing training losses for classification."""

    def __init__(self, weights=None):
        self.loss = torch.nn.CrossEntropyLoss(weights)

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the classification loss between predictions and true labels."""
        preds = preds[1] if isinstance(preds, (list, tuple)) else preds
        loss = self.loss(preds, batch["cls"])
        return loss, loss.detach()


class v8OBBLoss(v8DetectionLoss):
    """Calculates losses for object detection, classification, and box distribution in rotated YOLO models."""

    def __init__(self, model, tal_topk=10, tal_topk2: int | None = None, **kwargs):
        """Initialize v8OBBLoss with model, assigner, and rotated bbox loss."""
        super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2, **kwargs)
        self.assigner = RotatedTaskAlignedAssigner(
            topk=tal_topk,
            num_classes=self.nc[0],
            alpha=0.5,
            beta=6.0,
            stride=self.stride.tolist() if hasattr(self.stride, 'tolist') else self.stride,
            topk2=tal_topk2,
        )
        self.bbox_loss = RotatedBboxLoss(self.reg_max).to(self.device)

    def preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor) -> torch.Tensor:
        """Preprocess targets for oriented bounding box detection."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 6, device=self.device)
        else:
            i = targets[:, 0]
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 6, device=self.device)
            for j in range(batch_size):
                matches = i == j
                if n := matches.sum():
                    bboxes = targets[matches, 2:]
                    bboxes[..., :4].mul_(scale_tensor)
                    out[j, :n] = torch.cat([targets[matches, 1:2], bboxes], dim=-1)
        return out

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate and return the loss for oriented bounding box detection."""
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl, angle
        pred_distri, pred_scores, pred_angle = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
            preds["angle"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)
        batch_size = pred_angle.shape[0]

        dtype = pred_scores.dtype
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5)), 1)
            rw, rh = targets[:, 4] * float(imgsz[1]), targets[:, 5] * float(imgsz[0])
            targets = targets[(rw >= 2) & (rh >= 2)]
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 5), 2)
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ OBB dataset incorrectly formatted or not a OBB dataset.\n"
                "This error can occur when incorrectly training a 'OBB' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolov8n-obb.pt data=dota8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'OBB' dataset using 'data=dota8.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/obb/ for help."
            ) from e

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri, pred_angle)

        bboxes_for_assigner = pred_bboxes.clone().detach()
        bboxes_for_assigner[..., :4] *= stride_tensor
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores[..., :self.nc[0]].detach().sigmoid(),
            bboxes_for_assigner.type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels[..., :1],
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        if fg_mask.sum():
            target_bboxes[..., :4] /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )
            weight = target_scores.sum(-1)[fg_mask]
            loss[3] = self.calculate_angle_loss(
                pred_bboxes, target_bboxes, fg_mask, weight, target_scores_sum
            )
        else:
            loss[0] += (pred_angle * 0).sum()

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        loss[3] *= getattr(self.hyp, 'angle', 1.0)

        return loss.sum() * batch_size, loss.detach()

    def bbox_decode(
        self, anchor_points: torch.Tensor, pred_dist: torch.Tensor, pred_angle: torch.Tensor
    ) -> torch.Tensor:
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return torch.cat((dist2rbox(pred_dist, pred_angle, anchor_points), pred_angle), dim=-1)

    def calculate_angle_loss(self, pred_bboxes, target_bboxes, fg_mask, weight, target_scores_sum, lambda_val=3):
        """Calculate oriented angle loss."""
        w_gt = target_bboxes[..., 2]
        h_gt = target_bboxes[..., 3]
        pred_theta = pred_bboxes[..., 4]
        target_theta = target_bboxes[..., 4]

        log_ar = torch.log((w_gt + 1e-9) / (h_gt + 1e-9))
        scale_weight = torch.exp(-(log_ar**2) / (lambda_val**2))

        delta_theta = pred_theta - target_theta
        delta_theta_wrapped = delta_theta - torch.round(delta_theta / math.pi) * math.pi
        ang_loss = torch.sin(2 * delta_theta_wrapped[fg_mask]) ** 2

        ang_loss = scale_weight[fg_mask] * ang_loss
        ang_loss = ang_loss * weight

        return ang_loss.sum() / target_scores_sum


class E2EDetectLoss:
    """Criterion class for computing training losses for end-to-end detection."""

    def __init__(self, model):
        """Initialize E2EDetectLoss with one-to-many and one-to-one detection losses."""
        self.one2many = v8DetectionLoss(model, tal_topk=10)
        self.one2one = v8DetectionLoss(model, tal_topk=1, verbose=False)

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        preds = preds[1] if isinstance(preds, tuple) else preds
        one2many = preds["one2many"]
        loss_one2many = self.one2many(one2many, batch)
        one2one = preds["one2one"]
        loss_one2one = self.one2one(one2one, batch)
        return loss_one2many[0] + loss_one2one[0], loss_one2many[1] + loss_one2one[1]


class E2EPoseLoss:
    """Criterion class for computing training losses for end-to-end pose estimation."""

    def __init__(self, model, **kwargs):
        """Initialize E2EPoseLoss with one-to-many and one-to-one detection losses."""
        self.one2many = v8PoseLoss(model, tal_topk=10, **kwargs)
        self.one2one = v8PoseLoss(model, tal_topk=1, verbose=False, **kwargs)

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        preds = preds[1] if isinstance(preds, tuple) else preds
        one2many = preds["one2many"]
        loss_one2many = self.one2many(one2many, batch)
        one2one = preds["one2one"]
        loss_one2one = self.one2one(one2one, batch)
        return loss_one2many[0] + loss_one2one[0], loss_one2many[1] + loss_one2one[1]


class E2ESegmentLoss:
    """Criterion class for computing training losses for end-to-end segmentation."""

    def __init__(self, model, **kwargs):
        """Initialize E2ESegmentLoss with one-to-many and one-to-one detection losses."""
        self.one2many = v8SegmentationLoss(model, tal_topk=10, **kwargs)
        self.one2one = v8SegmentationLoss(model, tal_topk=1, verbose=False, **kwargs)

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        preds = preds[1] if isinstance(preds, tuple) else preds
        one2many = preds["one2many"]
        loss_one2many = self.one2many(one2many, batch)
        one2one = preds["one2one"]
        loss_one2one = self.one2one(one2one, batch)
        return loss_one2many[0] + loss_one2one[0], loss_one2many[1] + loss_one2one[1]


class SemanticSegmentationLoss(nn.Module):
    """Loss function for semantic segmentation using cross-entropy and Dice terms.

    Attributes:
        nc (int): Number of semantic classes.
        ce (nn.CrossEntropyLoss): Cross-entropy loss with ignore_index=255.
    """

    def __init__(self, model: torch.nn.Module):
        """Initialize semantic segmentation loss.

        Args:
            model (torch.nn.Module): Model containing the SemanticSegment head.
        """
        super().__init__()
        m = model.model[-1]
        self.nc = m.nc
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        # fork: model.args is a dict (set in attempt_load_*), upstream uses a namespace
        _args = getattr(model, "args", None)
        _data = ""
        if isinstance(_args, dict):
            _data = _args.get("data", "")
        elif _args is not None:
            _data = getattr(_args, "data", "")
        data_name = Path(str(_data or "")).stem.lower()
        self.use_cityscapes_weight = data_name in {"cityscapes", "cityscapes8"} and self.nc == len(CITYSCAPES_WEIGHT)
        weight = getattr(model, "class_weights", None)  # cls_pw frequency weights, else hardcoded Cityscapes
        if weight is None and self.use_cityscapes_weight:
            weight = torch.from_numpy(CITYSCAPES_WEIGHT)
        weight = None if weight is None else weight.to(device=self.device, dtype=self.dtype)
        if self.nc == 1:
            self.ce = nn.BCEWithLogitsLoss(reduction="sum")  # binary: class weighting intentionally unsupported
        else:
            self.ce = nn.CrossEntropyLoss(ignore_index=255, reduction="sum").to(device=self.device, dtype=self.dtype)
            if weight is not None:
                # Non-persistent: weight is a deterministic constant, no need to serialize into ckpt state_dict.
                self.ce.register_buffer("weight", weight, persistent=False)

    def _resize_masks(self, masks, target_shape):
        """Resize masks to match prediction spatial dimensions."""
        if masks.shape[1:] != target_shape:
            return (
                F.interpolate(masks.float().unsqueeze(1), size=target_shape, mode="nearest").squeeze(1).to(torch.int32)
            )
        return masks

    def _ce_loss(self, preds, masks, valid):
        """Compute cross-entropy on flattened pixels to avoid the CUDA nll_loss2d path."""
        flat = masks.reshape(-1)
        if self.nc == 1:
            logits = preds.reshape(-1)[valid]
            target = flat[valid].float()
            denominator = valid.sum()
        else:
            logits = preds.permute(0, 2, 3, 1).reshape(-1, self.nc)
            target = flat.long()
            denominator = valid.sum() if self.ce.weight is None else self.ce.weight[target[valid]].sum()
        return self.ce(logits, target) / denominator.clamp_min(1)

    def _dice_loss(self, preds, masks, valid):
        """Compute Dice loss excluding ignore pixels."""
        if self.nc == 1:
            return self._binary_dice_loss(preds, masks, valid)
        flat_target = masks.reshape(-1)
        pred_soft = F.softmax(preds, dim=1)
        target = flat_target[valid].long()
        flat_pred = pred_soft.float().permute(0, 2, 3, 1).reshape(-1, self.nc)[valid]
        intersection = torch.zeros(self.nc, device=preds.device, dtype=torch.float32)
        intersection.scatter_add_(0, target, flat_pred.gather(1, target[:, None]).squeeze(1))
        pred_sum = flat_pred.sum(dim=0)
        target_sum = torch.bincount(target, minlength=self.nc).to(device=preds.device, dtype=torch.float32)
        cardinality = pred_sum + target_sum
        return (1.0 - (2.0 * intersection + 1.0) / (cardinality + 1.0)).mean()

    def _binary_dice_loss(self, preds, masks, valid):
        """Compute Dice loss for single-class (binary) segmentation.

        Pixels with value 255 are excluded from Dice terms to match BCE valid-pixel filtering.
        """
        valid = valid.reshape_as(masks).float()
        pred_soft = preds.squeeze(1).sigmoid()
        target = (masks == 1).float()
        intersection = (pred_soft * target * valid).sum()
        cardinality = ((pred_soft + target) * valid).sum()
        return 1.0 - (2.0 * intersection + 1.0) / (cardinality + 1.0)

    def forward(self, preds, batch):
        """Compute semantic segmentation loss with optional auxiliary loss.

        Args:
            preds (torch.Tensor | tuple): Main logits [B, nc, H', W'], or (main, aux) tuple.
            batch (dict): Batch dict with 'semantic_mask' [B, H, W] containing class IDs (255=ignore).

        Returns:
            (tuple[torch.Tensor, dict[str, torch.Tensor]]): Total loss * batch_size and a dict of detached loss items
                (ce_loss, dice_loss, aux_loss).
        """
        # Unpack auxiliary logits when present.
        aux_logits = None
        if isinstance(preds, tuple):
            preds, aux_logits = preds

        masks = batch["semantic_mask"].to(preds.device)
        valid = masks.reshape(-1) != 255
        if preds.shape[2:] != masks.shape[1:]:
            preds = F.interpolate(preds, size=masks.shape[1:], mode="bilinear", align_corners=False)

        # Main cross-entropy and Dice loss.
        ce_loss = self._ce_loss(preds, masks, valid)
        dice_loss = self._dice_loss(preds, masks, valid)
        total = ce_loss + dice_loss

        # Auxiliary cross-entropy loss. Match ce_loss dtype so adding to total succeeds under AMP.
        aux_loss = torch.tensor(0.0, device=preds.device, dtype=ce_loss.dtype)
        if aux_logits is not None:
            if aux_logits.shape[2:] != masks.shape[1:]:
                aux_logits = F.interpolate(aux_logits, size=masks.shape[1:], mode="bilinear", align_corners=False)
            aux_loss = self._ce_loss(aux_logits, masks, valid) * 0.4
            total += aux_loss

        # Форк 8.3.6: BaseTrainer/BaseValidator ждут ТЕНЗОР loss_items, не dict (upstream 8.4.x умеет dict).
        # Порядок должен совпадать с SemanticSegmentationTrainer.loss_names.
        loss_items = torch.stack([ce_loss.detach(), dice_loss.detach(), aux_loss.detach()])
        return total * preds.shape[0], loss_items
class E2ELoss:
    """Criterion class for computing training losses for end-to-end detection with decay schedule."""

    def __init__(self, model, loss_fn=v8DetectionLoss, **kwargs):
        """Initialize E2ELoss with one-to-many and one-to-one detection losses."""
        self.one2many = loss_fn(model, tal_topk=10, **kwargs)
        self.one2one = loss_fn(model, tal_topk=7, tal_topk2=1, verbose=False, **kwargs)
        self.updates = 0
        self.total = 1.0
        self.o2m = 0.8
        self.o2o = self.total - self.o2m
        self.o2m_copy = self.o2m
        self.final_o2m = 0.1

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        preds = self.one2many.parse_output(preds)
        one2many, one2one = preds["one2many"], preds["one2one"]
        loss_one2many = self.one2many.loss(one2many, batch)
        loss_one2one = self.one2one.loss(one2one, batch)
        return loss_one2many[0] * self.o2m + loss_one2one[0] * self.o2o, loss_one2one[1]

    def update(self) -> None:
        """Update the weights for one-to-many and one-to-one losses based on the decay schedule."""
        self.updates += 1
        self.o2m = self.decay(self.updates)
        self.o2o = max(self.total - self.o2m, 0)

    def decay(self, x) -> float:
        """Calculate the decayed weight for one-to-many loss based on the current update step."""
        return max(1 - x / max(self.one2one.hyp.epochs - 1, 1), 0) * (self.o2m_copy - self.final_o2m) + self.final_o2m


class TVPDetectLoss:
    """Criterion class for computing training losses for text-visual prompt detection."""

    def __init__(self, model, tal_topk=10, tal_topk2: int | None = None):
        """Initialize TVPDetectLoss with task-prompt and visual-prompt criteria."""
        self.vp_criterion = v8DetectionLoss(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        self.hyp = self.vp_criterion.hyp
        self.ori_nc = self.vp_criterion.nc[0] if isinstance(self.vp_criterion.nc, list) else self.vp_criterion.nc
        self.ori_no = self.vp_criterion.no
        self.ori_reg_max = self.vp_criterion.reg_max

    def parse_output(self, preds) -> dict[str, torch.Tensor]:
        """Parse model predictions to extract features."""
        return self.vp_criterion.parse_output(preds)

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the loss for text-visual prompt detection."""
        return self.loss(self.parse_output(preds), batch)

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the loss for text-visual prompt detection."""
        if self.ori_nc == preds["scores"].shape[1]:
            loss = torch.zeros(3, device=self.vp_criterion.device, requires_grad=True)
            return loss, loss.detach()

        preds["scores"] = self._get_vp_features(preds)
        vp_loss = self.vp_criterion(preds, batch)
        box_loss = vp_loss[1][0]
        return box_loss, vp_loss[1]

    def _get_vp_features(self, preds: dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract visual-prompt features from the model output."""
        scores = preds["scores"]
        vnc = scores.shape[1]

        self.vp_criterion.nc = [vnc] if isinstance(self.vp_criterion.nc, list) else vnc
        self.vp_criterion.no = vnc + self.vp_criterion.reg_max * 4
        self.vp_criterion.assigner.num_classes = vnc
        return scores


class TVPSegmentLoss(TVPDetectLoss):
    """Criterion class for computing training losses for text-visual prompt segmentation."""

    def __init__(self, model, tal_topk=10):
        """Initialize TVPSegmentLoss with task-prompt and visual-prompt criteria."""
        super().__init__(model)
        self.vp_criterion = v8SegmentationLoss(model, tal_topk=tal_topk)
        self.hyp = self.vp_criterion.hyp

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the loss for text-visual prompt segmentation."""
        return self.loss(self.parse_output(preds), batch)

    def loss(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the loss for text-visual prompt detection."""
        ori_nc = self.vp_criterion.nc[0] if isinstance(self.vp_criterion.nc, list) else self.vp_criterion.nc
        if ori_nc == preds["scores"].shape[1]:
            loss = torch.zeros(4, device=self.vp_criterion.device, requires_grad=True)
            return loss, loss.detach()

        preds["scores"] = self._get_vp_features(preds)
        vp_loss = self.vp_criterion(preds, batch)
        cls_loss = vp_loss[0][2]
        return cls_loss, vp_loss[1]
