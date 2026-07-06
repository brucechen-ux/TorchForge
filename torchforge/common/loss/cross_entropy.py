from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class CausalLMLoss(nn.Module):
    """Next-token cross-entropy loss for causal language models.

    Shifts ``logits`` and ``labels`` by one position so that each position
    predicts the following token, then computes token-level cross entropy.

    Args:
        ignore_index: Label value excluded from the loss (e.g. padding).
        label_smoothing: Label-smoothing factor in ``[0, 1)``.

    Forward:
        ``logits`` has shape ``(batch, sequence_length, vocab_size)`` and
        ``labels`` has shape ``(batch, sequence_length)`` with integer dtype.

    Returns:
        Scalar loss tensor.
    """

    def __init__(self, *, ignore_index: int = -100, label_smoothing: float = 0.0) -> None:
        super().__init__()
        if not isinstance(ignore_index, int):
            raise ValueError(f"ignore_index must be an int, got {ignore_index!r}.")
        if label_smoothing < 0.0 or label_smoothing >= 1.0:
            raise ValueError("label_smoothing must be in [0, 1).")
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        _validate_logits(logits)
        _validate_labels(labels)
        if logits.shape[:2] != labels.shape[:2]:
            raise ValueError(
                f"logits and labels must share (batch, sequence_length), "
                f"got {tuple(logits.shape[:2])} and {tuple(labels.shape[:2])}."
            )
        if logits.shape[1] < 2:
            raise ValueError("sequence_length must be at least 2 to form a next-token target.")

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1),
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
        )


def _validate_logits(logits: torch.Tensor) -> None:
    if not isinstance(logits, torch.Tensor):
        raise TypeError(f"logits must be a torch.Tensor, got {type(logits).__name__}.")
    if logits.dim() != 3:
        raise ValueError("logits must have shape (batch, sequence_length, vocab_size).")
    if not torch.is_floating_point(logits):
        raise ValueError("logits must have a floating-point dtype.")


def _validate_labels(labels: torch.Tensor) -> None:
    if not isinstance(labels, torch.Tensor):
        raise TypeError(f"labels must be a torch.Tensor, got {type(labels).__name__}.")
    if labels.dim() != 2:
        raise ValueError("labels must have shape (batch, sequence_length).")
    if labels.dtype not in {torch.int64, torch.int32, torch.int16, torch.int8, torch.long}:
        raise ValueError(f"labels must have an integer dtype, got {labels.dtype}.")


__all__ = ["CausalLMLoss"]
