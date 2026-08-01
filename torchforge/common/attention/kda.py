from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
import math
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class KDAState:
    """Recurrent and short-convolution state for incremental KDA execution."""

    recurrent_state: torch.Tensor
    query_history: torch.Tensor
    key_history: torch.Tensor
    value_history: torch.Tensor


class KimiDeltaAttention(nn.Module):
    """Kimi-K3 Delta Attention with channel-wise lower-bounded decay.

    ``backend="reference"`` evaluates the recurrence token by token and is the
    correctness oracle. ``backend="fla"`` delegates only the recurrent scan to
    Flash Linear Attention while retaining TorchForge-owned projections.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        value_head_dim: Optional[int] = None,
        short_conv_kernel_size: int = 4,
        decay_rank: int = 64,
        g_min: float = -5.0,
        chunk_size: int = 64,
        tile_size: int = 16,
        rms_norm_eps: float = 1.0e-6,
        backend: str = "reference",
        bias: bool = False,
    ) -> None:
        super().__init__()
        values = {
            "hidden_size": hidden_size,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "short_conv_kernel_size": short_conv_kernel_size,
            "decay_rank": decay_rank,
            "chunk_size": chunk_size,
            "tile_size": tile_size,
        }
        for name, value in values.items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive int, got {value!r}.")
        if chunk_size % tile_size != 0:
            raise ValueError("chunk_size must be divisible by tile_size.")
        if g_min >= 0.0:
            raise ValueError("g_min must be negative.")
        if rms_norm_eps <= 0.0:
            raise ValueError("rms_norm_eps must be positive.")
        if backend not in {"reference", "fla"}:
            raise ValueError("backend must be either 'reference' or 'fla'.")
        if backend == "fla" and chunk_size not in {32, 64}:
            raise ValueError("backend='fla' requires chunk_size to be 32 or 64.")
        if backend == "fla" and tile_size != 16:
            raise ValueError("backend='fla' requires the Kimi-K3 16-token tile size.")
        if backend == "fla" and g_min < -5.0:
            raise ValueError("backend='fla' requires g_min in the numerically safe range [-5, 0).")

        value_head_dim = head_dim if value_head_dim is None else value_head_dim
        if not isinstance(value_head_dim, int) or value_head_dim <= 0:
            raise ValueError("value_head_dim must be a positive int.")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.value_head_dim = value_head_dim
        self.short_conv_kernel_size = short_conv_kernel_size
        self.decay_rank = decay_rank
        self.g_min = float(g_min)
        self.chunk_size = chunk_size
        self.tile_size = tile_size
        self.rms_norm_eps = float(rms_norm_eps)
        self.backend = backend

        self.q_weight = nn.Parameter(torch.empty(num_heads, head_dim, hidden_size))
        self.k_weight = nn.Parameter(torch.empty(num_heads, head_dim, hidden_size))
        self.v_weight = nn.Parameter(torch.empty(num_heads, value_head_dim, hidden_size))
        self.q_bias = nn.Parameter(torch.zeros(num_heads, head_dim)) if bias else None
        self.k_bias = nn.Parameter(torch.zeros(num_heads, head_dim)) if bias else None
        self.v_bias = nn.Parameter(torch.zeros(num_heads, value_head_dim)) if bias else None
        self.q_conv_weight = nn.Parameter(torch.empty(num_heads, head_dim, short_conv_kernel_size))
        self.k_conv_weight = nn.Parameter(torch.empty(num_heads, head_dim, short_conv_kernel_size))
        self.v_conv_weight = nn.Parameter(torch.empty(num_heads, value_head_dim, short_conv_kernel_size))
        self.beta_proj = nn.Linear(hidden_size, num_heads, bias=True)
        self.decay_a_proj = nn.Linear(hidden_size, decay_rank, bias=False)
        self.decay_b_weight = nn.Parameter(torch.empty(num_heads, head_dim, decay_rank))
        self.decay_bias = nn.Parameter(torch.zeros(num_heads, head_dim))
        self.decay_log_scale = nn.Parameter(torch.zeros(num_heads))
        self.output_norm_weight = nn.Parameter(torch.ones(num_heads, value_head_dim))
        self.output_gate = nn.Linear(hidden_size, num_heads * value_head_dim, bias=bias)
        self.output_proj = nn.Linear(num_heads * value_head_dim, hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for weight in (self.q_weight, self.k_weight, self.v_weight, self.decay_b_weight):
            nn.init.kaiming_uniform_(weight.flatten(0, 1), a=math.sqrt(5))
        for weight in (self.q_conv_weight, self.k_conv_weight, self.v_conv_weight):
            nn.init.zeros_(weight)
            weight.data[..., -1] = 1.0
        self.beta_proj.reset_parameters()
        self.decay_a_proj.reset_parameters()
        self.output_gate.reset_parameters()
        self.output_proj.reset_parameters()

    def _project(self, hidden_states: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]) -> torch.Tensor:
        projected = F.linear(
            hidden_states,
            weight.flatten(0, 1),
            None if bias is None else bias.flatten(),
        )
        return projected.view(*hidden_states.shape[:-1], weight.shape[0], weight.shape[1])

    def _initial_state(self, hidden_states: torch.Tensor) -> KDAState:
        batch_size = hidden_states.shape[0]
        history_length = self.short_conv_kernel_size - 1
        return KDAState(
            recurrent_state=torch.zeros(
                batch_size,
                self.num_heads,
                self.head_dim,
                self.value_head_dim,
                dtype=torch.float32,
                device=hidden_states.device,
            ),
            query_history=hidden_states.new_zeros((batch_size, history_length, self.num_heads, self.head_dim)),
            key_history=hidden_states.new_zeros((batch_size, history_length, self.num_heads, self.head_dim)),
            value_history=hidden_states.new_zeros(
                (batch_size, history_length, self.num_heads, self.value_head_dim)
            ),
        )

    def _short_conv(
        self,
        projected: torch.Tensor,
        weight: torch.Tensor,
        history: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat((history, projected), dim=1)
        batch_size, _, num_heads, head_dim = combined.shape
        flattened = combined.permute(0, 2, 3, 1).reshape(batch_size, num_heads * head_dim, -1)
        filters = weight.reshape(num_heads * head_dim, 1, self.short_conv_kernel_size)
        convolved = F.conv1d(flattened, filters, groups=num_heads * head_dim)
        convolved = convolved.view(batch_size, num_heads, head_dim, projected.shape[1]).permute(0, 3, 1, 2)
        history_length = self.short_conv_kernel_size - 1
        return convolved, combined[:, -history_length:] if history_length else combined[:, :0]

    def _prepare_inputs(
        self,
        hidden_states: torch.Tensor,
        state: KDAState,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        KDAState,
    ]:
        query, query_history = self._short_conv(
            self._project(hidden_states, self.q_weight, self.q_bias),
            self.q_conv_weight,
            state.query_history,
        )
        key, key_history = self._short_conv(
            self._project(hidden_states, self.k_weight, self.k_bias),
            self.k_conv_weight,
            state.key_history,
        )
        value, value_history = self._short_conv(
            self._project(hidden_states, self.v_weight, self.v_bias),
            self.v_conv_weight,
            state.value_history,
        )
        query = F.normalize(F.silu(query).float(), p=2.0, dim=-1, eps=1.0e-6)
        key = F.normalize(F.silu(key).float(), p=2.0, dim=-1, eps=1.0e-6)
        value = F.silu(value).float()
        beta_logits = self.beta_proj(hidden_states).float()
        beta = torch.sigmoid(beta_logits)
        decay_hidden = self.decay_a_proj(hidden_states)
        decay_input = self._project(decay_hidden, self.decay_b_weight, None).float()
        decay_logits = decay_input + self.decay_bias.float()
        scale = self.decay_log_scale.exp().view(1, 1, self.num_heads, 1)
        log_decay = self.g_min * torch.sigmoid(scale * decay_logits)
        next_state = KDAState(
            recurrent_state=state.recurrent_state,
            query_history=query_history,
            key_history=key_history,
            value_history=value_history,
        )
        return query, key, value, beta, log_decay, beta_logits, decay_input, next_state

    def _reference_scan(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        beta: torch.Tensor,
        log_decay: torch.Tensor,
        initial_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        recurrent_state = initial_state.float()
        outputs = []
        # Keep the reference scan chunked so its state-carry contract mirrors
        # the accelerated prefill path, while retaining token-level math.
        for chunk_start in range(0, query.shape[1], self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, query.shape[1])
            for token_index in range(chunk_start, chunk_end):
                q_t = query[:, token_index]
                k_t = key[:, token_index]
                v_t = value[:, token_index]
                beta_t = beta[:, token_index]
                alpha_t = log_decay[:, token_index].exp()
                decayed = recurrent_state * alpha_t.unsqueeze(-1)
                prediction = torch.einsum("bhkv,bhk->bhv", decayed, k_t)
                residual = v_t - prediction
                recurrent_state = decayed + torch.einsum(
                    "bh,bhk,bhv->bhkv", beta_t, k_t, residual
                )
                outputs.append(torch.einsum("bhkv,bhk->bhv", recurrent_state, q_t))
        output = torch.stack(outputs, dim=1) if outputs else value.new_empty((*value.shape[:-1], self.value_head_dim))
        return output, recurrent_state

    def _fla_scan(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        beta_logits: torch.Tensor,
        decay_input: torch.Tensor,
        initial_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not query.is_cuda:
            raise RuntimeError("KDA backend='fla' requires CUDA tensors.")
        try:
            operation = getattr(import_module("fla.ops.kda"), "chunk_kda")
        except (ImportError, AttributeError, OSError, RuntimeError) as exc:
            raise RuntimeError(
                "KDA backend='fla' requires a Flash Linear Attention build containing fla.ops.kda.chunk_kda."
            ) from exc
        try:
            result = operation(
                q=query.to(dtype=torch.bfloat16),
                k=key.to(dtype=torch.bfloat16),
                v=value.to(dtype=torch.bfloat16),
                g=decay_input.to(dtype=torch.bfloat16),
                beta=beta_logits.to(dtype=torch.bfloat16),
                scale=1.0,
                initial_state=initial_state,
                output_final_state=True,
                use_beta_sigmoid_in_kernel=True,
                use_gate_in_kernel=True,
                safe_gate=True,
                lower_bound=self.g_min,
                A_log=self.decay_log_scale.float(),
                dt_bias=self.decay_bias.float().flatten(),
                chunk_size=self.chunk_size,
            )
        except TypeError as exc:
            raise RuntimeError("Installed FLA KDA API is incompatible with TorchForge's adapter.") from exc
        except (RuntimeError, ValueError) as exc:
            raise RuntimeError(f"FLA KDA execution failed: {exc}") from exc
        if not isinstance(result, tuple) or len(result) != 2:
            raise RuntimeError("FLA chunk_kda must return (output, final_state).")
        output, final_state = result
        if not isinstance(output, torch.Tensor) or not isinstance(final_state, torch.Tensor):
            raise RuntimeError("FLA chunk_kda must return tensor output and final_state values.")
        return output.float(), final_state.float()

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        state: Optional[KDAState] = None,
        return_dict: bool = True,
        **_: Any,
    ) -> Any:
        if not isinstance(hidden_states, torch.Tensor):
            raise TypeError("hidden_states must be a torch.Tensor.")
        if hidden_states.dim() != 3 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"hidden_states must have shape (batch, sequence, {self.hidden_size}), got {tuple(hidden_states.shape)}."
            )
        state = self._initial_state(hidden_states) if state is None else state
        _validate_state(state, hidden_states, self)
        query, key, value, beta, log_decay, beta_logits, decay_input, next_state = self._prepare_inputs(
            hidden_states, state
        )
        if self.backend == "reference":
            attention_output, recurrent_state = self._reference_scan(
                query,
                key,
                value,
                beta,
                log_decay,
                state.recurrent_state,
            )
        else:
            attention_output, recurrent_state = self._fla_scan(
                query,
                key,
                value,
                beta_logits,
                decay_input,
                state.recurrent_state,
            )
        normalized = attention_output * torch.rsqrt(
            attention_output.square().mean(-1, keepdim=True) + self.rms_norm_eps
        )
        normalized = normalized * self.output_norm_weight.float()
        gate = torch.sigmoid(self.output_gate(hidden_states).float()).view_as(normalized)
        gated = (gate * normalized).flatten(2)
        output = F.linear(
            gated,
            self.output_proj.weight.float(),
            None if self.output_proj.bias is None else self.output_proj.bias.float(),
        ).to(hidden_states.dtype)
        next_state.recurrent_state = recurrent_state
        if not return_dict:
            return output, next_state
        return {
            "hidden_states": output,
            "state": next_state,
            "log_decay": log_decay,
        }


def _validate_state(state: KDAState, hidden_states: torch.Tensor, module: KimiDeltaAttention) -> None:
    if not isinstance(state, KDAState):
        raise TypeError("state must be a KDAState.")
    batch_size = hidden_states.shape[0]
    history_length = module.short_conv_kernel_size - 1
    expected = {
        "recurrent_state": (batch_size, module.num_heads, module.head_dim, module.value_head_dim),
        "query_history": (batch_size, history_length, module.num_heads, module.head_dim),
        "key_history": (batch_size, history_length, module.num_heads, module.head_dim),
        "value_history": (batch_size, history_length, module.num_heads, module.value_head_dim),
    }
    for name, shape in expected.items():
        value = getattr(state, name)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"KDAState.{name} must be a torch.Tensor.")
        if tuple(value.shape) != shape:
            raise ValueError(f"KDAState.{name} must have shape {shape}, got {tuple(value.shape)}.")
        if value.device != hidden_states.device:
            raise ValueError(f"KDAState.{name} must be on {hidden_states.device}, got {value.device}.")
    for name in ("query_history", "key_history", "value_history"):
        value = getattr(state, name)
        if value.dtype != hidden_states.dtype:
            raise ValueError(
                f"KDAState.{name} must have dtype {hidden_states.dtype}, got {value.dtype}."
            )


__all__ = ["KDAState", "KimiDeltaAttention"]
