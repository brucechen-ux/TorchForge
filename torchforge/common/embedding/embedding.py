from __future__ import annotations

import torch
from torch import nn


class Embedding(nn.Module):
    """Token embedding layer.

    Args:
        vocab_size: Number of tokens in the vocabulary.
        hidden_size: Size of each token embedding.

    Forward:
        ``input_ids`` has shape ``(...)`` and integer dtype.

    Returns:
        Tensor with shape ``(..., hidden_size)``.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_size: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(vocab_size, int) or vocab_size <= 0:
            raise ValueError(f"vocab_size must be a positive int, got {vocab_size!r}.")
        if not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ValueError(f"hidden_size must be a positive int, got {hidden_size!r}.")
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size, device=device, dtype=dtype)

    @property
    def weight(self) -> nn.Parameter:
        return self.embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError(f"input_ids must be a torch.Tensor, got {type(input_ids).__name__}.")
        return self.embedding(input_ids)


__all__ = ["Embedding"]

