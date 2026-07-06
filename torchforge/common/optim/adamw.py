from __future__ import annotations

from typing import Any, Iterable

import torch
from torch import nn


class AdamW:
    """Thin wrapper around :class:`torch.optim.AdamW` with training-friendly defaults.

    The wrapper keeps the assembly style of the component library: keyword-only
    construction with validated arguments, delegating the numerics to PyTorch.

    Args:
        params: An iterable of parameters or parameter groups to optimize.
        lr: Learning rate.
        betas: Coefficients for running averages of gradient and its square.
        eps: Term added to the denominator for numerical stability.
        weight_decay: Decoupled weight-decay coefficient.
    """

    def __init__(
        self,
        params: Iterable[Any],
        *,
        lr: float = 3e-4,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"lr must be positive, got {lr!r}.")
        if not (isinstance(betas, tuple) and len(betas) == 2):
            raise ValueError("betas must be a 2-tuple.")
        if not all(0.0 <= b < 1.0 for b in betas):
            raise ValueError("betas must each be in [0, 1).")
        if eps <= 0.0:
            raise ValueError(f"eps must be positive, got {eps!r}.")
        if weight_decay < 0.0:
            raise ValueError(f"weight_decay must be non-negative, got {weight_decay!r}.")
        self._optimizer = torch.optim.AdamW(
            params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
        )

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return self._optimizer.param_groups

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        """The wrapped PyTorch optimizer."""

        return self._optimizer

    def step(self) -> None:
        self._optimizer.step()

    def zero_grad(self, *, set_to_none: bool = True) -> None:
        self._optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict[str, Any]:
        return self._optimizer.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._optimizer.load_state_dict(state_dict)


def build_param_groups(module: nn.Module, *, weight_decay: float = 0.1) -> list[dict[str, Any]]:
    """Split a module's parameters into decay / no-decay groups.

    Following common LLM training practice, biases and 1-D parameters (norm
    scales) are excluded from weight decay while 2-D+ weights are decayed.

    Args:
        module: The module whose trainable parameters are grouped.
        weight_decay: Weight decay applied to the decay group.

    Returns:
        A list of two parameter-group dicts suitable for an optimizer.
    """

    if not isinstance(module, nn.Module):
        raise TypeError(f"module must be an nn.Module, got {type(module).__name__}.")
    if weight_decay < 0.0:
        raise ValueError(f"weight_decay must be non-negative, got {weight_decay!r}.")

    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    seen: set[int] = set()
    for param in module.parameters():
        if not param.requires_grad or id(param) in seen:
            continue
        seen.add(id(param))
        if param.dim() >= 2:
            decay.append(param)
        else:
            no_decay.append(param)

    groups: list[dict[str, Any]] = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    if not groups:
        raise ValueError("module has no trainable parameters.")
    return groups


__all__ = ["AdamW", "build_param_groups"]
