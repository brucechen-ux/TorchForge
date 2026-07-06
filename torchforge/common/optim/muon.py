from __future__ import annotations

from typing import Any, Iterable

import torch
from torch import nn


class Muon(torch.optim.Optimizer):
    """Muon optimizer: Nesterov momentum with Newton-Schulz orthogonalization.

    For matrix parameters (``dim >= 2``), the momentum buffer is
    orthogonalized via Newton-Schulz iterations before being applied as the
    weight update.  This keeps the effective update on the manifold of
    matrices with bounded spectral norm and empirically improves
    optimization of transformer hidden layers.

    One-dimensional parameters (biases, RMSNorm scales, …) receive plain
    SGD with Nesterov momentum — orthogonalization is undefined for vectors.

    Reference:
        Keller Jordan, "Muon: An optimizer for hidden layers in neural
        networks", 2024.  https://github.com/KellerJordan/modded-nanogpt

    Args:
        params: Iterable of parameters or parameter groups.
        lr: Learning rate.
        momentum: Nesterov momentum coefficient (``beta`` in SGD notation).
        ns_steps: Number of Newton-Schulz quintic polynomial iterations used
            to orthogonalize each gradient matrix.
        weight_decay: Decoupled weight-decay coefficient applied after the
            orthogonalized update (only to matrix parameters).
    """

    def __init__(
        self,
        params: Iterable[Any],
        *,
        lr: float = 2e-2,
        momentum: float = 0.95,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"lr must be positive, got {lr!r}.")
        if not (0.0 <= momentum < 1.0):
            raise ValueError(f"momentum must be in [0, 1), got {momentum!r}.")
        if not isinstance(ns_steps, int) or ns_steps < 1:
            raise ValueError(f"ns_steps must be a positive int, got {ns_steps!r}.")
        if weight_decay < 0.0:
            raise ValueError(f"weight_decay must be non-negative, got {weight_decay!r}.")
        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:  # type: ignore[override]
        """Perform a single optimization step.

        Args:
            closure: Optional closure that re-evaluates the model and returns
                the loss (same convention as :class:`torch.optim.Optimizer`).

        Returns:
            Loss returned by the closure, or ``None``.
        """

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr: float = group["lr"]
            momentum: float = group["momentum"]
            ns_steps: int = group["ns_steps"]
            weight_decay: float = group["weight_decay"]

            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad

                state = self.state[param]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(param)

                buf: torch.Tensor = state["momentum_buffer"]
                buf.mul_(momentum).add_(grad)

                if param.dim() >= 2:
                    update = _newton_schulz_orthogonalize(buf, steps=ns_steps)
                    if weight_decay != 0.0:
                        param.mul_(1.0 - lr * weight_decay)
                else:
                    # Biases / norm scales: plain Nesterov SGD, no orthogonalization.
                    update = buf.add(grad, alpha=momentum)

                param.add_(update, alpha=-lr)

        return loss


def _newton_schulz_orthogonalize(matrix: torch.Tensor, *, steps: int = 5) -> torch.Tensor:
    """Map a matrix to a near-orthogonal matrix via Newton-Schulz iterations.

    Uses the quintic polynomial ``X <- a*X + b*(X X^T)X + c*(X X^T)^2 X``
    with coefficients (a, b, c) = (3.4445, -4.7750, 2.0315), which converges
    to the orthogonal polar factor of the input matrix.

    The input is flattened to 2-D (first dimension kept, rest merged) before
    iteration and reshaped back on return.  For wide matrices (cols > rows)
    the iteration runs on the transpose and the result is transposed back.

    Args:
        matrix: Tensor with ``dim >= 2``.
        steps: Number of quintic polynomial iterations.

    Returns:
        Tensor of the same shape as ``matrix`` with approximately orthonormal
        rows (or columns for wide matrices), scaled so the output has the same
        Frobenius norm as the input.
    """

    orig_shape = matrix.shape
    # Flatten to 2-D: (rows, cols)
    g = matrix.view(matrix.shape[0], -1).float()
    transposed = g.shape[0] < g.shape[1]
    if transposed:
        g = g.T  # work on (cols, rows) so rows <= cols

    frob = g.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    x = g / frob

    # Quintic NS coefficients (Jordan 2024)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        a_ = x @ x.T          # (rows, rows)
        x = a * x + b * (a_ @ x) + c * (a_ @ a_ @ x)

    if transposed:
        x = x.T

    # Rescale to preserve the original Frobenius norm so that the effective
    # update magnitude is controlled purely by the learning rate.
    orig_frob = matrix.view(matrix.shape[0], -1).float().norm()
    out = x * (orig_frob / x.norm().clamp_min(1e-12))
    return out.view(orig_shape).to(matrix.dtype)


__all__ = ["Muon"]
