from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn


class MLP(nn.Module):
    """Configurable multilayer perceptron.

    Args:
        input_size: Size of the input feature dimension.
        hidden_size: Size of hidden layers.
        output_size: Size of the output feature dimension.
        num_layers: Number of linear layers.
        activation: Hidden activation: ``"silu"``, ``"gelu"``, ``"relu"``, or ``"tanh"``.
        dropout: Dropout probability after hidden activations.
        bias: Whether linear layers use bias.

    Forward:
        ``inputs`` has shape ``(..., input_size)``.

    Returns:
        Tensor with shape ``(..., output_size)``.
    """

    def __init__(
        self,
        *,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 2,
        activation: str = "gelu",
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        _validate_positive_int("input_size", input_size)
        _validate_positive_int("hidden_size", hidden_size)
        _validate_positive_int("output_size", output_size)
        _validate_positive_int("num_layers", num_layers)
        _validate_dropout(dropout)
        if activation not in {"silu", "gelu", "relu", "tanh"}:
            raise ValueError(f"Unsupported activation: {activation!r}.")
        self.input_size = input_size
        self.output_size = output_size
        self.dropout = dropout
        self.activation = activation
        if num_layers == 1:
            layers = [nn.Linear(input_size, output_size, bias=bias)]
        else:
            layers = [nn.Linear(input_size, hidden_size, bias=bias)]
            layers.extend(nn.Linear(hidden_size, hidden_size, bias=bias) for _ in range(num_layers - 2))
            layers.append(nn.Linear(hidden_size, output_size, bias=bias))
        self.layers = nn.ModuleList(layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _validate_tensor_last_dim("inputs", inputs, self.input_size)
        hidden = inputs
        activation = _activation_fn(self.activation)
        for layer in self.layers[:-1]:
            hidden = activation(layer(hidden))
            hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        return self.layers[-1](hidden)


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


def _validate_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive int, got {value!r}.")


def _validate_dropout(dropout: float) -> None:
    if dropout < 0.0 or dropout >= 1.0:
        raise ValueError("dropout must be in [0, 1).")


def _validate_tensor_last_dim(name: str, value: torch.Tensor, expected: int) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}.")
    if value.dim() < 1:
        raise ValueError(f"{name} must have at least 1 dimension.")
    if value.shape[-1] != expected:
        raise ValueError(f"{name} last dimension must be {expected}, got {value.shape[-1]}.")


__all__ = ["MLP"]
