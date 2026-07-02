from __future__ import annotations

import torch
from torch import nn

from .expert import ExpertMLP


class SharedExpertMLP(nn.Module):
    """Shared expert applied to every token in a DeepSeekMoE block."""

    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        activation: str = "silu",
        gated: bool = True,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.expert = ExpertMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation=activation,
            gated=gated,
            bias=bias,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.expert(hidden_states)


__all__ = ["SharedExpertMLP"]
