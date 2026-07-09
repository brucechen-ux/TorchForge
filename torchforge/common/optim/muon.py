from __future__ import annotations

from typing import Any, Iterable

import torch
from torch import nn


class Muon(torch.optim.Optimizer):
    """Muon optimizer for matrix parameters.

    Muon applies Nesterov momentum followed by Newton-Schulz orthogonalization to
    2-D+ matrix parameters. Biases, norm scales, embeddings, and other non-matrix
    parameters should be optimized by AdamW via
    :func:`build_hybrid_optimizer_param_groups`.

    Args:
        params: Iterable of matrix parameters or parameter groups. Every parameter
            must have ``dim >= 2``.
        lr: Learning rate.
        momentum: Momentum coefficient used for the Nesterov update.
        ns_steps: Number of Newton-Schulz quintic iterations.
        weight_decay: Decoupled weight decay applied to matrix parameters.
        update_scale: Fixed RMS-matching scale applied after orthogonalization.
    """

    def __init__(
        self,
        params: Iterable[Any],
        *,
        lr: float = 2e-2,
        momentum: float = 0.95,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
        update_scale: float = 0.2,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"lr must be positive, got {lr!r}.")
        if not (0.0 <= momentum < 1.0):
            raise ValueError(f"momentum must be in [0, 1), got {momentum!r}.")
        if not isinstance(ns_steps, int) or ns_steps < 1:
            raise ValueError(f"ns_steps must be a positive int, got {ns_steps!r}.")
        if weight_decay < 0.0:
            raise ValueError(f"weight_decay must be non-negative, got {weight_decay!r}.")
        if update_scale <= 0.0:
            raise ValueError(f"update_scale must be positive, got {update_scale!r}.")
        defaults = dict(
            lr=lr,
            momentum=momentum,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
            update_scale=update_scale,
        )
        super().__init__(params, defaults)
        _validate_muon_param_groups(self.param_groups)

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr: float = group["lr"]
            momentum: float = group["momentum"]
            ns_steps: int = group["ns_steps"]
            weight_decay: float = group["weight_decay"]
            update_scale: float = group["update_scale"]

            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.dim() < 2:
                    raise ValueError("Muon only supports parameters with dim >= 2; use AdamW for vectors/scalars.")

                grad = param.grad
                state = self.state[param]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(param)

                buf: torch.Tensor = state["momentum_buffer"]
                buf.mul_(momentum).add_(grad)
                nesterov_update = grad.add(buf, alpha=momentum) if momentum != 0.0 else grad
                update = _newton_schulz_orthogonalize(nesterov_update, steps=ns_steps)
                update = _scale_muon_update(update, scale=update_scale)
                if weight_decay != 0.0:
                    param.mul_(1.0 - lr * weight_decay)
                param.add_(update, alpha=-lr)

        return loss


def build_hybrid_optimizer_param_groups(
    module: nn.Module,
    *,
    weight_decay: float = 0.1,
) -> dict[str, list[dict[str, Any]]]:
    """Split module parameters into Muon matrix groups and AdamW fallback groups.

    Returns:
        ``{"muon": [...], "adamw": [...]}``, where Muon groups contain only
        2-D+ trainable parameters and AdamW groups contain scalar/vector trainable
        parameters with zero weight decay.
    """

    if not isinstance(module, nn.Module):
        raise TypeError(f"module must be an nn.Module, got {type(module).__name__}.")
    if weight_decay < 0.0:
        raise ValueError(f"weight_decay must be non-negative, got {weight_decay!r}.")

    muon_params: list[nn.Parameter] = []
    adamw_params: list[nn.Parameter] = []
    seen: set[int] = set()
    for param in module.parameters():
        if not param.requires_grad or id(param) in seen:
            continue
        seen.add(id(param))
        if param.dim() >= 2:
            muon_params.append(param)
        else:
            adamw_params.append(param)

    if not muon_params and not adamw_params:
        raise ValueError("module has no trainable parameters.")

    groups: dict[str, list[dict[str, Any]]] = {"muon": [], "adamw": []}
    if muon_params:
        groups["muon"].append({"params": muon_params, "weight_decay": weight_decay})
    if adamw_params:
        groups["adamw"].append({"params": adamw_params, "weight_decay": 0.0})
    return groups


def _validate_muon_param_groups(param_groups: list[dict[str, Any]]) -> None:
    for group in param_groups:
        for param in group["params"]:
            if param.dim() < 2:
                raise ValueError("Muon only supports parameters with dim >= 2; use AdamW for vectors/scalars.")


def _newton_schulz_orthogonalize(matrix: torch.Tensor, *, steps: int = 5) -> torch.Tensor:
    """Map a matrix-like tensor to a near-orthogonal update.

    The input is flattened to 2-D (first dimension kept, rest merged). For wide
    matrices the iteration runs on the transpose and is transposed back. The
    output is not rescaled to the input Frobenius norm; Muon uses fixed
    shape-based scaling instead.
    """

    orig_shape = matrix.shape
    x = matrix.view(matrix.shape[0], -1).float()
    transposed = x.shape[0] < x.shape[1]
    if transposed:
        x = x.T

    x = x / x.norm().clamp_min(1e-12)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        xx_t = x @ x.T
        x = a * x + b * (xx_t @ x) + c * (xx_t @ xx_t @ x)

    if transposed:
        x = x.T
    return x.view(orig_shape).to(matrix.dtype)


def _scale_muon_update(update: torch.Tensor, *, scale: float = 0.2) -> torch.Tensor:
    rows = update.shape[0]
    cols = update.numel() // rows
    return update * (scale * (max(rows, cols) ** 0.5))


__all__ = ["Muon", "build_hybrid_optimizer_param_groups"]
