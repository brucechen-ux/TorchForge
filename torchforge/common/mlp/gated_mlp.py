from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class GatedMLP(nn.Module):
    """DeepSeek-style gated feed-forward network.

    Computes ``down_proj(act(gate_proj(x)) * up_proj(x))`` when gated, or
    ``down_proj(act(up_proj(x)))`` otherwise. This is the canonical dense FFN
    structure used by DeepSeek-V3/V4 dense layers, routed experts, and shared
    experts.

    Args:
        hidden_size: Size of the input and output hidden-state dimension.
        intermediate_size: Size of the intermediate feed-forward dimension.
        activation: Activation function, one of ``"silu"``, ``"gelu"``, or ``"relu"``.
        gated: Whether to use the gated MLP path.
        bias: Whether projection layers use bias.
        clamp_limit: When set, applies DeepSeek-V4 SwiGLU clamping (paper Section
            4.2.3): the linear component is clamped to ``[-clamp_limit, clamp_limit]``
            and the gate component's upper bound is capped at ``clamp_limit``.

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
        clamp_limit: float | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ValueError(f"hidden_size must be a positive int, got {hidden_size!r}.")
        if not isinstance(intermediate_size, int) or intermediate_size <= 0:
            raise ValueError(f"intermediate_size must be a positive int, got {intermediate_size!r}.")
        if activation not in {"silu", "gelu", "relu"}:
            raise ValueError(f"Unsupported activation: {activation!r}.")
        if clamp_limit is not None and (not isinstance(clamp_limit, (int, float)) or clamp_limit <= 0.0):
            raise ValueError(f"clamp_limit must be a positive number or None, got {clamp_limit!r}.")

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.activation = activation
        self.gated = gated
        self.clamp_limit = None if clamp_limit is None else float(clamp_limit)
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
                raise RuntimeError("gated GatedMLP requires gate_proj.")
            gate = self.gate_proj(hidden_states)
            if self.clamp_limit is not None:
                # DeepSeek-V4 SwiGLU clamping: bound the linear branch symmetrically
                # and cap only the upper bound of the gate branch.
                up = up.clamp(-self.clamp_limit, self.clamp_limit)
                gate = gate.clamp(max=self.clamp_limit)
            hidden = self._activate(gate) * up
        else:
            if self.clamp_limit is not None:
                up = up.clamp(-self.clamp_limit, self.clamp_limit)
            hidden = self._activate(up)
        return self.down_proj(hidden)


__all__ = ["GatedMLP"]
