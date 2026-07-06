from __future__ import annotations

from typing import Any, Callable, Optional

import torch
from torch import nn

from torchforge.common.optim import AdamW


class TrainStep:
    """Orchestrate a single optimization step for a language model.

    Wires a forward function, a loss module, and an optimizer into one
    ``zero_grad -> forward -> loss -> backward -> (clip) -> step`` cycle. It
    performs no data loading or distribution, keeping the training loop minimal
    and assembled from the surrounding components.

    Args:
        forward_fn: Callable mapping ``input_ids`` to logits of shape
            ``(batch, sequence_length, vocab_size)``.
        loss_module: Module mapping ``(logits, labels)`` to a scalar loss.
        optimizer: An :class:`AdamW` (or compatible) optimizer over the model
            parameters.
        max_grad_norm: Gradient-norm clip threshold, or ``None`` to disable.
    """

    def __init__(
        self,
        *,
        forward_fn: Callable[[torch.Tensor], torch.Tensor],
        loss_module: nn.Module,
        optimizer: AdamW,
        max_grad_norm: Optional[float] = 1.0,
    ) -> None:
        if not callable(forward_fn):
            raise TypeError("forward_fn must be callable.")
        if not isinstance(loss_module, nn.Module):
            raise TypeError(f"loss_module must be an nn.Module, got {type(loss_module).__name__}.")
        if not hasattr(optimizer, "step") or not hasattr(optimizer, "zero_grad"):
            raise TypeError("optimizer must expose step() and zero_grad().")
        if max_grad_norm is not None and max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive or None.")
        self.forward_fn = forward_fn
        self.loss_module = loss_module
        self.optimizer = optimizer
        self.max_grad_norm = max_grad_norm

    def run(self, input_ids: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError(f"input_ids must be a torch.Tensor, got {type(input_ids).__name__}.")
        if not isinstance(labels, torch.Tensor):
            raise TypeError(f"labels must be a torch.Tensor, got {type(labels).__name__}.")

        self.optimizer.zero_grad()
        logits = self.forward_fn(input_ids)
        loss = self.loss_module(logits, labels)
        loss.backward()

        grad_norm = self._clip_gradients()
        self.optimizer.step()
        return {"loss": float(loss.detach()), "grad_norm": grad_norm}

    def _clip_gradients(self) -> float:
        params = [
            p
            for group in self.optimizer.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        if not params:
            return 0.0
        max_norm = self.max_grad_norm if self.max_grad_norm is not None else float("inf")
        total_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm)
        return float(total_norm)


__all__ = ["TrainStep"]
