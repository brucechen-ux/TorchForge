from __future__ import annotations

import torch
from torch import nn


class LMHead(nn.Module):
    """Language-model output projection.

    Args:
        hidden_size: Size of the input hidden-state dimension.
        vocab_size: Size of the output vocabulary dimension.
        bias: Whether the projection uses bias.

    Forward:
        ``hidden_states`` has shape ``(..., hidden_size)``.

    Returns:
        Logits with shape ``(..., vocab_size)``.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        vocab_size: int,
        bias: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ValueError(f"hidden_size must be a positive int, got {hidden_size!r}.")
        if not isinstance(vocab_size, int) or vocab_size <= 0:
            raise ValueError(f"vocab_size must be a positive int, got {vocab_size!r}.")
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.proj = nn.Linear(hidden_size, vocab_size, bias=bias, device=device, dtype=dtype)

    @property
    def weight(self) -> nn.Parameter:
        return self.proj.weight

    def tie_weights(self, embedding: nn.Module) -> None:
        """Share output projection weights with an embedding module."""

        weight = getattr(embedding, "weight", None)
        if weight is None:
            raise TypeError("embedding must expose a weight parameter.")
        if tuple(weight.shape) != (self.vocab_size, self.hidden_size):
            raise ValueError(
                f"embedding weight must have shape {(self.vocab_size, self.hidden_size)}, "
                f"got {tuple(weight.shape)}."
            )
        self.proj.weight = weight

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not isinstance(hidden_states, torch.Tensor):
            raise TypeError(f"hidden_states must be a torch.Tensor, got {type(hidden_states).__name__}.")
        if hidden_states.dim() < 1:
            raise ValueError("hidden_states must have at least 1 dimension.")
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(f"hidden_states last dimension must be {self.hidden_size}, got {hidden_states.shape[-1]}.")
        return self.proj(hidden_states)


__all__ = ["LMHead"]

