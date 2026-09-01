from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch import nn

from torchforge.common.nn import RMSNorm


@dataclass
class GatedDeltaNetState:
    """Explicit recurrent matrix carried between Gated DeltaNet calls."""

    recurrent_state: torch.Tensor


class Qwen3NextGatedDeltaNet(nn.Module):
    """Reference Qwen3-Next-style Gated DeltaNet recurrent layer."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        value_head_dim: Optional[int] = None,
        beta_bias: float = 0.0,
        decay_bias: float = 0.0,
        rms_norm_eps: float = 1.0e-6,
        output_gate: bool = True,
        output_projection: bool = True,
        backend: str = "reference",
        bias: bool = False,
    ) -> None:
        super().__init__()
        for name, value in {
            "hidden_size": hidden_size,
            "num_heads": num_heads,
            "head_dim": head_dim,
        }.items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive int, got {value!r}.")
        value_head_dim = head_dim if value_head_dim is None else value_head_dim
        if not isinstance(value_head_dim, int) or value_head_dim <= 0:
            raise ValueError(f"value_head_dim must be a positive int, got {value_head_dim!r}.")
        if not isinstance(beta_bias, (int, float)):
            raise TypeError("beta_bias must be a number.")
        if not isinstance(decay_bias, (int, float)):
            raise TypeError("decay_bias must be a number.")
        if rms_norm_eps <= 0.0:
            raise ValueError("rms_norm_eps must be positive.")
        if backend != "reference":
            raise ValueError("backend must be 'reference'.")
        if not output_projection and num_heads * value_head_dim != hidden_size:
            raise ValueError(
                "num_heads * value_head_dim must equal hidden_size when output_projection=False."
            )

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.value_head_dim = value_head_dim
        self.rms_norm_eps = float(rms_norm_eps)
        self.backend = backend

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, num_heads * value_head_dim, bias=bias)
        self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        self.beta_proj = nn.Linear(hidden_size, num_heads, bias=True)
        self.decay_proj = nn.Linear(hidden_size, num_heads, bias=True)
        self.output_gate = (
            nn.Linear(hidden_size, num_heads * value_head_dim, bias=bias) if output_gate else None
        )
        self.output_proj = (
            nn.Linear(num_heads * value_head_dim, hidden_size, bias=bias)
            if output_projection
            else None
        )
        nn.init.constant_(self.beta_proj.bias, float(beta_bias))
        nn.init.constant_(self.decay_proj.bias, float(decay_bias))

    def _initial_recurrent_state(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            hidden_states.shape[0],
            self.num_heads,
            self.head_dim,
            self.value_head_dim,
            device=hidden_states.device,
            dtype=torch.float32,
        )

    def _shape_query_or_key(self, projected: torch.Tensor) -> torch.Tensor:
        return projected.view(projected.shape[0], projected.shape[1], self.num_heads, self.head_dim)

    def _shape_value(self, projected: torch.Tensor) -> torch.Tensor:
        return projected.view(
            projected.shape[0], projected.shape[1], self.num_heads, self.value_head_dim
        )

    def _reference_scan(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        recurrent_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = []
        current_state = recurrent_state
        for token_index in range(query.shape[1]):
            q_t = query[:, token_index]
            k_t = key[:, token_index]
            v_t = value[:, token_index]
            prediction = torch.einsum("bhkv,bhk->bhv", current_state, k_t)
            error = v_t - prediction
            current_state = alpha[:, token_index].unsqueeze(-1).unsqueeze(-1) * current_state
            current_state = current_state + torch.einsum(
                "bh,bhk,bhv->bhkv", beta[:, token_index], k_t, error
            )
            outputs.append(torch.einsum("bhkv,bhk->bhv", current_state, q_t))
        if outputs:
            return torch.stack(outputs, dim=1), current_state
        empty = value.new_empty(value.shape[0], 0, self.num_heads, self.value_head_dim)
        return empty, current_state

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        state: Optional[GatedDeltaNetState] = None,
        recurrent_state: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **_: Any,
    ) -> Any:
        del position_ids
        _validate_hidden_states(hidden_states, self.hidden_size)
        if state is not None and recurrent_state is not None:
            raise ValueError("Pass only one of state or recurrent_state.")
        if state is not None:
            if not isinstance(state, GatedDeltaNetState):
                raise TypeError("state must be a GatedDeltaNetState.")
            recurrent_state = state.recurrent_state
        if recurrent_state is None:
            recurrent_state = self._initial_recurrent_state(hidden_states)
        _validate_recurrent_state(recurrent_state, hidden_states, self)

        query = self.q_norm(self._shape_query_or_key(self.q_proj(hidden_states))).float()
        key = self.k_norm(self._shape_query_or_key(self.k_proj(hidden_states))).float()
        value = self._shape_value(self.v_proj(hidden_states)).float()
        gate_eps = torch.finfo(torch.float32).eps
        beta = torch.sigmoid(self.beta_proj(hidden_states).float()).clamp(gate_eps, 1.0 - gate_eps)
        alpha = torch.sigmoid(self.decay_proj(hidden_states).float()).clamp(gate_eps, 1.0 - gate_eps)
        output, next_recurrent_state = self._reference_scan(
            query, key, value, alpha, beta, recurrent_state
        )
        if self.output_gate is not None:
            gate = torch.sigmoid(self.output_gate(hidden_states).float()).view_as(output)
            output = output * gate
        output = output.flatten(2)
        if self.output_proj is not None:
            output = self.output_proj(output.to(self.output_proj.weight.dtype))
        output = output.to(hidden_states.dtype)
        next_state = GatedDeltaNetState(recurrent_state=next_recurrent_state)

        if not return_dict:
            return output, next_state
        return {
            "hidden_states": output,
            "state": next_state,
            "recurrent_state": next_recurrent_state,
            "alpha": alpha,
            "beta": beta,
        }


def _validate_hidden_states(hidden_states: torch.Tensor, hidden_size: int) -> None:
    if not isinstance(hidden_states, torch.Tensor):
        raise TypeError("hidden_states must be a torch.Tensor.")
    if hidden_states.dim() != 3 or hidden_states.shape[-1] != hidden_size:
        raise ValueError(
            f"hidden_states must have shape (batch, sequence, {hidden_size}), "
            f"got {tuple(hidden_states.shape)}."
        )
    if not torch.is_floating_point(hidden_states):
        raise TypeError("hidden_states must have a floating-point dtype.")


def _validate_recurrent_state(
    recurrent_state: torch.Tensor,
    hidden_states: torch.Tensor,
    module: Qwen3NextGatedDeltaNet,
) -> None:
    if not isinstance(recurrent_state, torch.Tensor):
        raise TypeError("recurrent_state must be a torch.Tensor.")
    expected_shape = (
        hidden_states.shape[0],
        module.num_heads,
        module.head_dim,
        module.value_head_dim,
    )
    if tuple(recurrent_state.shape) != expected_shape:
        raise ValueError(
            f"recurrent_state must have shape {expected_shape}, got {tuple(recurrent_state.shape)}."
        )
    if recurrent_state.device != hidden_states.device:
        raise ValueError(
            f"recurrent_state must be on {hidden_states.device}, got {recurrent_state.device}."
        )
    if recurrent_state.dtype != torch.float32:
        raise ValueError(
            f"recurrent_state must have dtype torch.float32, got {recurrent_state.dtype}."
        )


GatedDeltaNet = Qwen3NextGatedDeltaNet


__all__ = ["GatedDeltaNet", "GatedDeltaNetState", "Qwen3NextGatedDeltaNet"]
