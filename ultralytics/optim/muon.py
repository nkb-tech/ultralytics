# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Muon and MuSGD optimizer implementations for YOLO26 training.

Incorporates Turbo-Muon improvements from https://hal.science/hal-05390446:
  - AOL preconditioning replaces Frobenius normalization for a better starting point
  - Dynamic per-iteration polynomial coefficients for faster convergence
  - 4 Newton-Schulz iterations instead of 5, saving ~20% compute at equal quality
"""

from __future__ import annotations

import torch
from torch import optim

# Try to load Triton-accelerated Newton-Schulz kernels
_USE_TRITON_NS = False
_triton_newton_schulz = None
try:
    from kernels import get_kernel
    _kern = get_kernel("tboissin/newton_schulz_triton")
    _triton_newton_schulz = _kern.newton_schulz
    _USE_TRITON_NS = True
except (ImportError, Exception):
    pass

# Per-iteration tuned quintic coefficients for AOL-preconditioned Newton-Schulz
_NS_COEFFS = [
    (4.0848, -6.8946, 2.9270),
    (3.9505, -6.3029, 2.6377),
    (3.7418, -5.5913, 2.3037),
    (2.8769, -3.1427, 1.2046),
    (2.8366, -3.0525, 1.2012),
]

# Legacy fixed coefficients (Frobenius normalization, 5 iters)
_NS_LEGACY_COEFF = (3.4445, -4.7750, 2.0315)


def _newton_schulz_pytorch(
    G: torch.Tensor,
    steps: int = 4,
    precondition: bool = True,
    eps: float = 1e-7,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Newton-Schulz iteration for orthogonalization.

    Args:
        G: Input 2D+ tensor to orthogonalize.
        steps: Number of NS iterations (4 recommended with AOL, 5 without).
        precondition: Use AOL preconditioning (True) or legacy Frobenius normalization (False).
        eps: Numerical stability epsilon.
        dtype: Compute dtype (bfloat16 recommended).

    Returns:
        Orthogonalized matrix with same shape as input.
    """
    assert G.ndim >= 2
    X = G.to(dtype=dtype)
    if G.size(-2) > G.size(-1):
        X = X.mT

    coeffs = _NS_COEFFS[-steps:] if precondition else [_NS_LEGACY_COEFF] * steps

    if not precondition:
        X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)

    for i, (a, b, c) in enumerate(coeffs):
        A = X @ X.mT
        if precondition and i == 0:
            s = torch.rsqrt(torch.clamp_min(A.abs().sum(dim=-1, keepdim=False), min=eps))
            X = X * s.unsqueeze(-1)
            A = A * s.unsqueeze(-1) * s.unsqueeze(-2)
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def newton_schulz(
    G: torch.Tensor,
    steps: int = 4,
    precondition: bool = True,
    eps: float = 1e-7,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Newton-Schulz orthogonalization with optional Triton acceleration.

    Uses AOL preconditioning by default for better convergence in fewer iterations.
    Falls back to pure PyTorch when Triton kernels are not available.
    """
    if _USE_TRITON_NS:
        try:
            return _triton_newton_schulz(G, iter=steps, precondition=precondition,
                                         epsilon=eps, dtype=dtype)
        except Exception:
            pass
    return _newton_schulz_pytorch(G, steps=steps, precondition=precondition, eps=eps, dtype=dtype)


def zeropower_via_newtonschulz5(G: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Legacy wrapper kept for backward compatibility. Uses turbo mode (AOL, 4 iters)."""
    return newton_schulz(G, steps=4, precondition=True, eps=eps)


def muon_update(
    grad: torch.Tensor,
    momentum: torch.Tensor,
    beta: float = 0.95,
    nesterov: bool = True,
    ns_steps: int = 4,
    precondition: bool = True,
) -> torch.Tensor:
    """Compute Muon optimizer update with momentum and orthogonalization.

    Args:
        grad: Gradient tensor (2D or 4D for conv filters; 1D skips orthogonalization).
        momentum: Momentum buffer tensor, modified in-place.
        beta: Momentum coefficient.
        nesterov: Whether to use Nesterov momentum.
        ns_steps: Number of Newton-Schulz iterations.
        precondition: Use AOL preconditioning.

    Returns:
        Orthogonalized update tensor.
    """
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim < 2:
        return update
    if update.ndim == 4:
        update = update.view(len(update), -1)
    update = newton_schulz(update, steps=ns_steps, precondition=precondition)
    update *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5
    return update


class MuSGD(optim.Optimizer):
    """Hybrid optimizer combining Muon and SGD updates for neural network training.

    Uses turbo mode (AOL preconditioning, 4 NS iterations) by default for ~20% faster
    optimizer steps compared to the legacy 5-iteration Frobenius approach.

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): Learning rate.
        momentum (float): Momentum factor.
        weight_decay (float): Weight decay (L2 penalty).
        nesterov (bool): Whether to use Nesterov momentum.
        use_muon (bool): Whether to enable Muon updates.
        muon (float): Scaling factor for Muon component.
        sgd (float): Scaling factor for SGD component.
        variant (str): 'turbo' (AOL, 4 iters) or 'standard' (Frobenius, 5 iters).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        use_muon: bool = False,
        muon: float = 0.5,
        sgd: float = 0.5,
        variant: str = "turbo",
    ):
        assert variant in ("turbo", "standard")
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            use_muon=use_muon,
        )
        super().__init__(params, defaults)
        self.muon = muon
        self.sgd = sgd
        self.ns_steps = 4 if variant == "turbo" else 5
        self.precondition = variant == "turbo"

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    lr = group["lr"]
                    if p.grad is None:
                        continue
                    grad = p.grad
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                        state["momentum_buffer_SGD"] = torch.zeros_like(p)

                    update = muon_update(
                        grad,
                        state["momentum_buffer"],
                        beta=group["momentum"],
                        nesterov=group["nesterov"],
                        ns_steps=self.ns_steps,
                        precondition=self.precondition,
                    )
                    p.add_(update.reshape(p.shape), alpha=-(lr * self.muon))

                    if group["weight_decay"] != 0:
                        grad = grad.add(p, alpha=group["weight_decay"])
                    state["momentum_buffer_SGD"].mul_(group["momentum"]).add_(grad)
                    sgd_update = (
                        grad.add(state["momentum_buffer_SGD"], alpha=group["momentum"])
                        if group["nesterov"]
                        else state["momentum_buffer_SGD"]
                    )
                    p.add_(sgd_update, alpha=-(lr * self.sgd))
            else:
                for p in group["params"]:
                    lr = group["lr"]
                    if p.grad is None:
                        continue
                    grad = p.grad
                    if group["weight_decay"] != 0:
                        grad = grad.add(p, alpha=group["weight_decay"])
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    state["momentum_buffer"].mul_(group["momentum"]).add_(grad)
                    update = (
                        grad.add(state["momentum_buffer"], alpha=group["momentum"])
                        if group["nesterov"]
                        else state["momentum_buffer"]
                    )
                    p.add_(update, alpha=-lr)
        return loss


class Muon(optim.Optimizer):
    """Muon optimizer for non-distributed settings.

    Uses turbo mode (AOL preconditioning, 4 NS iterations) by default.

    Args:
        params (iterable): Parameters to optimize.
        lr (float): Learning rate. Default: 0.02.
        weight_decay (float): Weight decay coefficient. Default: 0.
        momentum (float): Momentum coefficient. Default: 0.95.
        variant (str): 'turbo' (AOL, 4 iters) or 'standard' (Frobenius, 5 iters).
    """

    def __init__(self, params, lr: float = 0.02, weight_decay: float = 0, momentum: float = 0.95, variant: str = "turbo"):
        assert variant in ("turbo", "standard")
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        super().__init__(params, defaults)
        self.ns_steps = 4 if variant == "turbo" else 5
        self.precondition = variant == "turbo"

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                update = muon_update(
                    p.grad,
                    state["momentum_buffer"],
                    beta=group["momentum"],
                    ns_steps=self.ns_steps,
                    precondition=self.precondition,
                )
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.reshape(p.shape), alpha=-group["lr"])

        return loss
