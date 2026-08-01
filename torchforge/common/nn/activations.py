from __future__ import annotations

from typing import Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn


class SwiGLU(nn.Module):
    """SwiGLU gated activation.

    Forward:
        Accepts either ``(gate, value)`` tensors or a single tensor whose last
        dimension is split evenly into gate and value halves.

    Returns:
        ``silu(gate) * value``.
    """

    def forward(self, inputs: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        gate, value = _split_or_pair(inputs, "SwiGLU")
        return F.silu(gate) * value


class GEGLU(nn.Module):
    """GEGLU gated activation.

    Forward:
        Accepts either ``(gate, value)`` tensors or a single tensor whose last
        dimension is split evenly into gate and value halves.

    Returns:
        ``gelu(gate) * value``.
    """

    def forward(self, inputs: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        gate, value = _split_or_pair(inputs, "GEGLU")
        return F.gelu(gate) * value


class SiTUGLU(nn.Module):
    """Sigmoid Tanh Unit GLU used by Kimi-K3.

    The gate's linear factor and the value branch are independently soft-capped
    before multiplication. With the paper defaults the output magnitude is
    bounded by ``beta_gate * beta_up``.
    """

    def __init__(self, beta_gate: float = 4.0, beta_up: float = 25.0) -> None:
        super().__init__()
        if beta_gate <= 0.0 or beta_up <= 0.0:
            raise ValueError("beta_gate and beta_up must be positive.")
        self.beta_gate = float(beta_gate)
        self.beta_up = float(beta_up)

    def forward(self, inputs: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        gate, value = _split_or_pair(inputs, "SiTUGLU")
        capped_gate = self.beta_gate * torch.tanh(gate / self.beta_gate)
        capped_value = self.beta_up * torch.tanh(value / self.beta_up)
        return capped_gate * torch.sigmoid(gate) * capped_value


def _split_or_pair(inputs: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], component: str) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(inputs, tuple):
        if len(inputs) != 2:
            raise TypeError(f"{component} tuple input must contain exactly two tensors.")
        gate, value = inputs
        if not isinstance(gate, torch.Tensor) or not isinstance(value, torch.Tensor):
            raise TypeError(f"{component} tuple input must contain tensors.")
        if gate.shape != value.shape:
            raise ValueError(f"{component} gate and value tensors must have matching shapes.")
        return gate, value
    if not isinstance(inputs, torch.Tensor):
        raise TypeError(f"{component} input must be a torch.Tensor or a tuple of tensors.")
    if inputs.shape[-1] % 2 != 0:
        raise ValueError(f"{component} single-tensor input last dimension must be even.")
    return inputs.chunk(2, dim=-1)


__all__ = ["GEGLU", "SiTUGLU", "SwiGLU"]
