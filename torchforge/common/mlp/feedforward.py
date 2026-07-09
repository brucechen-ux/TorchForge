from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn

from torchforge.common.nn import GEGLU, SwiGLU


class FeedForward(nn.Module):
    """Two-layer feed-forward network.

    Args:
        hidden_size: Size of the input and output hidden-state dimension.
        intermediate_size: Size of the intermediate feed-forward dimension.
        activation: Activation function: ``"silu"``, ``"gelu"``, ``"relu"``,
            ``"swiglu"``, or ``"geglu"``.
        dropout: Dropout probability applied after activation and output projection.
        bias: Whether projection layers use bias.

    Forward:
        ``hidden_states`` has shape ``(..., hidden_size)``.

    Returns:
        Tensor with the same shape as ``hidden_states``.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        activation: str = "silu",
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        _validate_sizes(hidden_size, intermediate_size)
        _validate_dropout(dropout)
        if activation not in {"silu", "gelu", "relu", "swiglu", "geglu"}:
            raise ValueError(f"Unsupported activation: {activation!r}.")
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.activation = activation
        self.dropout = dropout
        if activation in {"swiglu", "geglu"}:
            self.up_proj = nn.Linear(hidden_size, 2 * intermediate_size, bias=bias)
            self.gated_activation = SwiGLU() if activation == "swiglu" else GEGLU()
        else:
            self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
            self.gated_activation = None
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        _validate_hidden_states(hidden_states, self.hidden_size)
        hidden = self.up_proj(hidden_states)
        if self.gated_activation is not None:
            hidden = self.gated_activation(hidden)
        else:
            hidden = _activation_fn(self.activation)(hidden)
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        output = self.down_proj(hidden)
        return F.dropout(output, p=self.dropout, training=self.training)


def _activation_fn(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    if name == "silu":
        return F.silu
    if name == "gelu":
        return F.gelu
    if name == "relu":
        return F.relu
    if name == "tanh":
        return torch.tanh
    raise ValueError(f"Unsupported activation: {name!r}.")


def _validate_sizes(hidden_size: int, intermediate_size: int) -> None:
    _validate_positive_int("hidden_size", hidden_size)
    _validate_positive_int("intermediate_size", intermediate_size)


def _validate_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive int, got {value!r}.")


def _validate_dropout(dropout: float) -> None:
    if dropout < 0.0 or dropout >= 1.0:
        raise ValueError("dropout must be in [0, 1).")


def _validate_hidden_states(hidden_states: torch.Tensor, hidden_size: int) -> None:
    _validate_tensor_last_dim("hidden_states", hidden_states, hidden_size)


def _validate_tensor_last_dim(name: str, value: torch.Tensor, expected: int) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}.")
    if value.dim() < 1:
        raise ValueError(f"{name} must have at least 1 dimension.")
    if value.shape[-1] != expected:
        raise ValueError(f"{name} last dimension must be {expected}, got {value.shape[-1]}.")


__all__ = ["FeedForward"]
