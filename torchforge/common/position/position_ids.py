from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class PositionIds(nn.Module):
    """Build dense position ids for decoder-only models.

    Args:
        start: First position id used when ``past_length`` is zero.

    Forward:
        Either pass ``input_ids`` with shape ``(batch, sequence_length)`` or pass
        ``batch_size`` and ``seq_length`` explicitly.

    Returns:
        Position ids with shape ``(batch, sequence_length)``.
    """

    def __init__(self, *, start: int = 0) -> None:
        super().__init__()
        if not isinstance(start, int) or start < 0:
            raise ValueError("start must be a non-negative int.")
        self.start = start

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        *,
        batch_size: Optional[int] = None,
        seq_length: Optional[int] = None,
        past_length: int = 0,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        if input_ids is not None:
            if not isinstance(input_ids, torch.Tensor):
                raise TypeError(f"input_ids must be a torch.Tensor, got {type(input_ids).__name__}.")
            if input_ids.dim() != 2:
                raise ValueError("input_ids must have shape (batch, sequence_length).")
            batch_size, seq_length = input_ids.shape
            device = input_ids.device
        if batch_size is None or seq_length is None:
            raise ValueError("batch_size and seq_length are required when input_ids is omitted.")
        if batch_size <= 0 or seq_length <= 0:
            raise ValueError("batch_size and seq_length must be positive.")
        if past_length < 0:
            raise ValueError("past_length must be non-negative.")
        positions = torch.arange(
            self.start + past_length,
            self.start + past_length + seq_length,
            device=device,
            dtype=torch.long,
        )
        return positions.unsqueeze(0).expand(batch_size, -1)


__all__ = ["PositionIds"]
