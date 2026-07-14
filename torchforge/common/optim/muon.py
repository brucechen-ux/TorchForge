from __future__ import annotations

import math
from typing import Any, Iterable

import torch
from torch import nn


class Muon(torch.optim.Optimizer):
    """Muon optimizer for matrix parameters.

    Muon applies Nesterov momentum followed by Newton-Schulz orthogonalization to
    2-D matrices and packed 3-D logical matrices. Biases, norm scales, embeddings, and other non-matrix
    parameters should be optimized by AdamW via
    :func:`build_hybrid_optimizer_param_groups`.

    Args:
        params: Iterable of matrix parameters or parameter groups. Every parameter
            must have dimension 2 or 3.
        lr: Learning rate.
        momentum: Momentum coefficient used for the Nesterov update.
        ns_steps: Number of Newton-Schulz iterations. DeepSeek-V4 Hybrid uses 10
            (8 aggressive convergence steps followed by 2 stabilizing steps).
        ns_method: ``"hybrid"`` for the report's 8+2 coefficients or
            ``"standard"`` for stabilizing coefficients at every iteration.
        nesterov: Whether to use ``N = momentum * M + G`` after updating
            ``M = momentum * M + G``.
        weight_decay: Decoupled weight decay applied to matrix parameters.
        update_scale: Fixed RMS-matching scale applied after orthogonalization.
    """

    def __init__(
        self,
        params: Iterable[Any],
        *,
        lr: float = 2e-2,
        momentum: float = 0.95,
        ns_steps: int = 10,
        ns_method: str = "hybrid",
        nesterov: bool = True,
        weight_decay: float = 0.0,
        update_scale: float = 0.18,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"lr must be positive, got {lr!r}.")
        if not (0.0 <= momentum < 1.0):
            raise ValueError(f"momentum must be in [0, 1), got {momentum!r}.")
        if not isinstance(ns_steps, int) or ns_steps < 1:
            raise ValueError(f"ns_steps must be a positive int, got {ns_steps!r}.")
        if ns_method not in {"hybrid", "standard"}:
            raise ValueError(f"ns_method must be 'hybrid' or 'standard', got {ns_method!r}.")
        if weight_decay < 0.0:
            raise ValueError(f"weight_decay must be non-negative, got {weight_decay!r}.")
        if update_scale <= 0.0:
            raise ValueError(f"update_scale must be positive, got {update_scale!r}.")
        defaults = dict(
            lr=lr,
            momentum=momentum,
            ns_steps=ns_steps,
            ns_method=ns_method,
            nesterov=bool(nesterov),
            weight_decay=weight_decay,
            update_scale=update_scale,
        )
        super().__init__(params, defaults)
        _validate_muon_param_groups(self.param_groups)
        self.last_step_metrics: dict[str, float | int] = {}

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        update_sum_squares: torch.Tensor | None = None
        update_numel = 0
        logical_matrix_count = 0
        for group in self.param_groups:
            lr: float = group["lr"]
            momentum: float = group["momentum"]
            ns_steps: int = group["ns_steps"]
            ns_method: str = group["ns_method"]
            weight_decay: float = group["weight_decay"]
            update_scale: float = group["update_scale"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.dim() not in {2, 3}:
                    raise ValueError("Muon supports 2-D matrices and packed 3-D logical matrices only.")

                grad = param.grad
                state = self.state[param]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(param)

                buf: torch.Tensor = state["momentum_buffer"]
                buf.mul_(momentum).add_(grad)
                nesterov_update = grad.add(buf, alpha=momentum) if group["nesterov"] else buf
                logical_inputs = nesterov_update.unbind(0) if nesterov_update.dim() == 3 else (nesterov_update,)
                logical_updates = []
                for logical_input in logical_inputs:
                    logical_update = _newton_schulz_orthogonalize(
                        logical_input,
                        steps=ns_steps,
                        method=ns_method,
                    )
                    logical_update = _scale_muon_update(logical_update, scale=update_scale)
                    logical_updates.append(logical_update)
                    logical_sum_squares = logical_update.float().square().sum()
                    update_sum_squares = (
                        logical_sum_squares
                        if update_sum_squares is None
                        else update_sum_squares + logical_sum_squares
                    )
                    update_numel += logical_update.numel()
                    logical_matrix_count += 1
                update = torch.stack(logical_updates) if nesterov_update.dim() == 3 else logical_updates[0]
                if weight_decay != 0.0:
                    param.mul_(1.0 - lr * weight_decay)
                param.add_(update, alpha=-lr)

        self.last_step_metrics = {
            "muon_update_rms": (
                math.sqrt(float(update_sum_squares.item()) / update_numel)
                if update_sum_squares is not None and update_numel
                else 0.0
            ),
            "logical_matrix_count": logical_matrix_count,
        }

        return loss


def build_hybrid_optimizer_param_groups(
    module: nn.Module,
    *,
    weight_decay: float = 0.1,
) -> dict[str, list[dict[str, Any]]]:
    """Split module parameters into Muon matrix groups and AdamW fallback groups.

    Assignment follows DeepSeek-V4 (paper Section 2.4) by *module role* rather than
    by tensor rank alone: the embedding module, the prediction head, all
    normalization scales, and the static biases and gating factors of mHC modules
    are optimized by AdamW; every other 2-D+ matrix parameter is optimized by Muon.
    This matters because some AdamW-owned tensors (e.g. embedding and LM-head
    weights, the mHC residual-mapping bias) are 2-D and would otherwise be
    mis-assigned to Muon by a rank-only split.

    Returns:
        ``{"muon": [...], "adamw": [...]}``, where Muon groups contain matrix
        parameters and AdamW groups contain the role-forced and scalar/vector
        parameters with zero weight decay.
    """

    if not isinstance(module, nn.Module):
        raise TypeError(f"module must be an nn.Module, got {type(module).__name__}.")
    if weight_decay < 0.0:
        raise ValueError(f"weight_decay must be non-negative, got {weight_decay!r}.")

    adamw_forced: set[int] = _collect_adamw_forced_param_ids(module)

    muon_params: list[nn.Parameter] = []
    adamw_params: list[nn.Parameter] = []
    seen: set[int] = set()
    for param in module.parameters():
        if not param.requires_grad or id(param) in seen:
            continue
        seen.add(id(param))
        if id(param) not in adamw_forced and param.dim() in {2, 3}:
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


# Module class names whose *entire* parameter subtree is AdamW-owned. Matched by
# name to avoid importing (and circularly depending on) the model components.
_ADAMW_ROLE_NAMES = frozenset({"Embedding", "LMHead", "RMSNorm", "UnweightedRMSNorm"})
# Modules whose *direct* parameters (static biases + gating factors) are
# AdamW-owned, while their submodule weight matrices remain Muon-eligible.
_ADAMW_DIRECT_PARAM_NAMES = frozenset({"ManifoldConstrainedHyperConnection"})


def _collect_adamw_forced_param_ids(module: nn.Module) -> set[int]:
    forced: set[int] = set()
    for submodule in module.modules():
        name = type(submodule).__name__
        if (
            isinstance(submodule, (nn.Embedding, nn.LayerNorm, nn.GroupNorm))
            or name in _ADAMW_ROLE_NAMES
            or name.endswith("Router")
        ):
            for param in submodule.parameters(recurse=True):
                forced.add(id(param))
        elif name in _ADAMW_DIRECT_PARAM_NAMES:
            for param in submodule.parameters(recurse=False):
                forced.add(id(param))
    return forced


def _validate_muon_param_groups(param_groups: list[dict[str, Any]]) -> None:
    for group in param_groups:
        for param in group["params"]:
            if param.dim() not in {2, 3}:
                raise ValueError("Muon supports 2-D matrices and packed 3-D logical matrices only.")


# DeepSeek-V4 hybrid Newton-Schulz coefficients (paper Section 2.4). The first
# stage drives rapid convergence toward singular values near 1; the final stage
# stabilizes them precisely at 1.
_NS_CONVERGE_COEFFS = (3.4445, -4.7750, 2.0315)
_NS_STABILIZE_COEFFS = (2.0, -1.5, 0.5)
_NS_STABILIZE_STEPS = 2


def _newton_schulz_orthogonalize(
    matrix: torch.Tensor,
    *,
    steps: int = 10,
    method: str = "hybrid",
) -> torch.Tensor:
    """Map a 2-D matrix to a near-orthogonal update via Newton-Schulz.

    For tall matrices the iteration runs on the transpose and is transposed back, so
    ``X @ X.T`` uses the smaller matrix dimension. The output is not rescaled
    to the input Frobenius norm; Muon uses fixed shape-based scaling instead.

    Following DeepSeek-V4, the iteration runs in two stages: the last
    ``min(2, steps)`` iterations use stabilizing coefficients ``(2, -1.5, 0.5)``
    that pin the singular values to 1, and the earlier iterations use the
    aggressive convergence coefficients ``(3.4445, -4.7750, 2.0315)``.
    """

    if matrix.dim() != 2:
        raise ValueError(f"Newton-Schulz expects a 2-D logical matrix, got shape={tuple(matrix.shape)}.")
    if method not in {"hybrid", "standard"}:
        raise ValueError(f"method must be 'hybrid' or 'standard', got {method!r}.")
    if not torch.isfinite(matrix).all():
        raise FloatingPointError("Newton-Schulz input contains NaN or Inf.")

    work = matrix.float()
    max_abs = work.abs().max()
    frobenius_norm = (work / max_abs).norm() * max_abs if max_abs > 0 else max_abs
    x = matrix.to(dtype=torch.bfloat16 if matrix.is_cuda else torch.float32)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T

    x = x / frobenius_norm.clamp_min(1e-7).to(dtype=x.dtype)
    stabilize_steps = min(_NS_STABILIZE_STEPS, steps) if method == "hybrid" else steps
    converge_steps = steps - stabilize_steps
    for step in range(steps):
        a, b, c = _NS_CONVERGE_COEFFS if step < converge_steps else _NS_STABILIZE_COEFFS
        xx_t = x @ x.T
        x = a * x + (b * xx_t + c * (xx_t @ xx_t)) @ x

    if transposed:
        x = x.T
    return x.to(matrix.dtype)


def _scale_muon_update(update: torch.Tensor, *, scale: float = 0.18) -> torch.Tensor:
    if update.dim() != 2:
        raise ValueError("Muon update scaling expects one 2-D logical matrix.")
    rows = update.shape[0]
    cols = update.shape[1]
    return update * (scale * (max(rows, cols) ** 0.5))


__all__ = ["Muon", "build_hybrid_optimizer_param_groups"]
