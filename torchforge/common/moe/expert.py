from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ExpertMLP(nn.Module):
    """Feed-forward expert MLP.

    Args:
        hidden_size: Size of the input and output hidden-state dimension.
        intermediate_size: Size of the intermediate feed-forward dimension.
        activation: Activation function, one of ``"silu"``, ``"gelu"``, or ``"relu"``.
        gated: Whether to use a gated MLP path.
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
        gated: bool = True,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")
        if intermediate_size <= 0:
            raise ValueError("intermediate_size must be positive.")
        if activation not in {"silu", "gelu", "relu"}:
            raise ValueError(f"Unsupported activation: {activation!r}.")

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.activation = activation
        self.gated = gated
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias) if gated else None
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def _activate(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "silu":
            return F.silu(x)
        if self.activation == "gelu":
            return F.gelu(x)
        if self.activation == "relu":
            return F.relu(x)
        raise ValueError(f"Unsupported activation: {self.activation!r}.")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not isinstance(hidden_states, torch.Tensor):
            raise TypeError(f"hidden_states must be a torch.Tensor, got {type(hidden_states).__name__}.")
        if hidden_states.dim() < 2:
            raise ValueError("hidden_states must have at least 2 dimensions.")
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(f"hidden_states last dimension must be {self.hidden_size}, got {hidden_states.shape[-1]}.")
        up = self.up_proj(hidden_states)
        if self.gated:
            if self.gate_proj is None:
                raise RuntimeError("gated ExpertMLP requires gate_proj.")
            hidden = self._activate(self.gate_proj(hidden_states)) * up
        else:
            hidden = self._activate(up)
        return self.down_proj(hidden)


__all__ = ["ExpertMLP"]
