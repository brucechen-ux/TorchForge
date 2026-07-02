from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class SlidingWindowCausalMask(nn.Module):
    """Build additive causal masks restricted to a local attention window.

    Args:
        window_size: Number of most recent key positions each query can attend.

    Forward:
        Either pass ``input_ids`` with shape ``(batch, sequence_length)`` or pass
        ``batch_size`` and ``seq_length`` explicitly.

    Returns:
        Additive mask with shape ``(batch, 1, sequence_length, sequence_length + past_length)``.
    """

    def __init__(self, *, window_size: int) -> None:
        super().__init__()
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError(f"window_size must be a positive int, got {window_size!r}.")
        self.window_size = window_size

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        *,
        batch_size: Optional[int] = None,
        seq_length: Optional[int] = None,
        past_length: int = 0,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
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

        key_length = seq_length + past_length
        query_positions = torch.arange(past_length, past_length + seq_length, device=device).unsqueeze(-1)
        key_positions = torch.arange(key_length, device=device).unsqueeze(0)
        future = key_positions > query_positions
        too_old = key_positions < (query_positions - self.window_size + 1)
        blocked = future | too_old
        mask = torch.zeros((seq_length, key_length), device=device, dtype=dtype)
        mask = mask.masked_fill(blocked, torch.finfo(dtype).min)
        return mask.view(1, 1, seq_length, key_length).expand(batch_size, 1, seq_length, key_length)


__all__ = ["SlidingWindowCausalMask"]
