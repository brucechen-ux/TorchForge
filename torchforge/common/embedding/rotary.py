from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    """Rotary position embedding table computed from position ids.

    Args:
        head_dim: Attention head dimension used to derive the rotary dimension.
        rope_theta: RoPE frequency base.
        partial_rotary_factor: Fraction of ``head_dim`` using rotary features.
        max_position_embeddings: Informational maximum sequence length.

    Forward:
        ``position_ids`` has shape ``(batch, sequence_length)``. When omitted,
        ``seq_length`` must be provided and a single-row position tensor is used.

    Returns:
        ``(cos, sin)`` tensors with shape ``(batch, sequence_length, rotary_dim // 2)``.
    """

    def __init__(
        self,
        *,
        head_dim: int,
        rope_theta: float = 10000.0,
        partial_rotary_factor: float = 1.0,
        max_position_embeddings: int = 4096,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(head_dim, int) or head_dim <= 0:
            raise ValueError(f"head_dim must be a positive int, got {head_dim!r}.")
        if rope_theta <= 0.0:
            raise ValueError("rope_theta must be positive.")
        if partial_rotary_factor <= 0.0 or partial_rotary_factor > 1.0:
            raise ValueError("partial_rotary_factor must be in (0, 1].")
        if not isinstance(max_position_embeddings, int) or max_position_embeddings <= 0:
            raise ValueError("max_position_embeddings must be a positive int.")
        rotary_dim = int(head_dim * partial_rotary_factor)
        if rotary_dim <= 0:
            raise ValueError("computed rotary_dim must be positive.")
        if rotary_dim % 2 != 0:
            rotary_dim -= 1
        if rotary_dim <= 0:
            raise ValueError("computed rotary_dim must be at least 2 after even rounding.")

        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        self.rope_theta = rope_theta
        self.partial_rotary_factor = partial_rotary_factor
        self.max_position_embeddings = max_position_embeddings
        inv_freq = 1.0 / (
            rope_theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device) / rotary_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        position_ids: Optional[torch.Tensor] = None,
        *,
        seq_length: Optional[int] = None,
        device: torch.device | str | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if position_ids is None:
            if seq_length is None:
                raise ValueError("Either position_ids or seq_length must be provided.")
            if not isinstance(seq_length, int) or seq_length <= 0:
                raise ValueError("seq_length must be a positive int.")
            position_ids = torch.arange(seq_length, device=device or self.inv_freq.device).unsqueeze(0)
        if not isinstance(position_ids, torch.Tensor):
            raise TypeError(f"position_ids must be a torch.Tensor, got {type(position_ids).__name__}.")
        if position_ids.dim() != 2:
            raise ValueError("position_ids must have shape (batch, sequence_length).")
        inv_freq = self.inv_freq.to(position_ids.device)
        freqs = torch.einsum("bs,d->bsd", position_ids.float(), inv_freq)
        return torch.cos(freqs), torch.sin(freqs)


__all__ = ["RotaryEmbedding"]

