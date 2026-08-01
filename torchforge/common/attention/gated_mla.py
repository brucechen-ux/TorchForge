from __future__ import annotations

import math
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn

from torchforge.common.nn import RMSNorm


class GatedMLA(nn.Module):
    """NoPE Multi-head Latent Attention with Kimi-K3 output gating."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        head_dim: int,
        value_head_dim: Optional[int] = None,
        rms_norm_eps: float = 1.0e-6,
        attention_dropout: float = 0.0,
        attention_backend: str = "sdpa",
        bias: bool = False,
    ) -> None:
        super().__init__()
        values = {
            "hidden_size": hidden_size,
            "num_heads": num_heads,
            "q_lora_rank": q_lora_rank,
            "kv_lora_rank": kv_lora_rank,
            "head_dim": head_dim,
        }
        for name, value in values.items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive int, got {value!r}.")
        value_head_dim = head_dim if value_head_dim is None else value_head_dim
        if not isinstance(value_head_dim, int) or value_head_dim <= 0:
            raise ValueError("value_head_dim must be a positive int.")
        if not 0.0 <= attention_dropout < 1.0:
            raise ValueError("attention_dropout must be in [0, 1).")
        if attention_backend not in {"reference", "sdpa"}:
            raise ValueError("attention_backend must be either 'reference' or 'sdpa'.")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.head_dim = head_dim
        self.value_head_dim = value_head_dim
        self.attention_dropout = float(attention_dropout)
        self.attention_backend = attention_backend
        self.scaling = head_dim**-0.5

        self.q_a_proj = nn.Linear(hidden_size, q_lora_rank, bias=bias)
        self.q_a_norm = RMSNorm(q_lora_rank, eps=rms_norm_eps)
        self.q_b_weight = nn.Parameter(torch.empty(num_heads, head_dim, q_lora_rank))
        self.kv_a_proj = nn.Linear(hidden_size, kv_lora_rank, bias=bias)
        self.kv_a_norm = RMSNorm(kv_lora_rank, eps=rms_norm_eps)
        self.k_weight = nn.Parameter(torch.empty(num_heads, head_dim, kv_lora_rank))
        self.v_weight = nn.Parameter(torch.empty(num_heads, value_head_dim, kv_lora_rank))
        self.output_gate = nn.Linear(hidden_size, num_heads * value_head_dim, bias=bias)
        self.output_proj = nn.Linear(num_heads * value_head_dim, hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.q_a_proj.reset_parameters()
        self.kv_a_proj.reset_parameters()
        for weight in (self.q_b_weight, self.k_weight, self.v_weight):
            nn.init.kaiming_uniform_(weight.flatten(0, 1), a=math.sqrt(5))
        self.output_gate.reset_parameters()
        self.output_proj.reset_parameters()

    @staticmethod
    def _packed_linear(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        output = F.linear(hidden_states, weight.flatten(0, 1))
        return output.view(*hidden_states.shape[:-1], weight.shape[0], weight.shape[1]).transpose(1, 2)

    def _reference_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        output_attentions: bool,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        scores = torch.matmul(query.float(), key.float().transpose(-2, -1)) * self.scaling
        sequence_length = query.shape[-2]
        causal = torch.ones((sequence_length, sequence_length), device=query.device, dtype=torch.bool).triu(1)
        scores = scores.masked_fill(causal, float("-inf"))
        if attention_mask is not None:
            attention_mask = self._normalize_attention_mask(attention_mask, query.shape[0], sequence_length)
            if attention_mask.dtype == torch.bool:
                scores = scores.masked_fill(~attention_mask, float("-inf"))
            else:
                scores = scores + attention_mask.float()
        probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32)
        probabilities = F.dropout(
            probabilities,
            p=self.attention_dropout if self.training else 0.0,
            training=self.training,
        )
        output = torch.matmul(probabilities, value.float()).transpose(1, 2)
        return output, probabilities if output_attentions else None

    def _sdpa_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if attention_mask is None:
            mask = None
            is_causal = True
        else:
            sequence_length = query.shape[-2]
            attention_mask = self._normalize_attention_mask(attention_mask, query.shape[0], sequence_length)
            causal = torch.full(
                (sequence_length, sequence_length),
                float("-inf"),
                device=query.device,
                dtype=query.dtype,
            ).triu(1)
            if attention_mask.dtype == torch.bool:
                supplied = torch.zeros_like(attention_mask, dtype=query.dtype).masked_fill(
                    ~attention_mask, float("-inf")
                )
            else:
                supplied = attention_mask.to(query.dtype)
            mask = supplied + causal
            is_causal = False
        output = F.scaled_dot_product_attention(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            attn_mask=mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
            scale=self.scaling,
        )
        return output.transpose(1, 2).float()

    @staticmethod
    def _normalize_attention_mask(
        attention_mask: torch.Tensor,
        batch_size: int,
        sequence_length: int,
    ) -> torch.Tensor:
        if not isinstance(attention_mask, torch.Tensor):
            raise TypeError("attention_mask must be a torch.Tensor.")
        if attention_mask.dim() == 2:
            if tuple(attention_mask.shape) == (batch_size, sequence_length):
                return attention_mask[:, None, None, :]
            if tuple(attention_mask.shape) == (sequence_length, sequence_length):
                return attention_mask[None, None, :, :]
        elif attention_mask.dim() == 3:
            if tuple(attention_mask.shape[-2:]) == (sequence_length, sequence_length):
                return attention_mask[:, None, :, :]
        elif attention_mask.dim() == 4:
            if tuple(attention_mask.shape[-2:]) == (sequence_length, sequence_length):
                return attention_mask
        raise ValueError(
            "attention_mask must have shape (batch, sequence), (sequence, sequence), "
            "(batch, sequence, sequence), or (batch, heads, sequence, sequence)."
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        return_dict: bool = True,
        **_: Any,
    ) -> Any:
        del position_ids  # Kimi-K3 MLA intentionally uses no positional encoding.
        if not isinstance(hidden_states, torch.Tensor):
            raise TypeError("hidden_states must be a torch.Tensor.")
        if hidden_states.dim() != 3 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"hidden_states must have shape (batch, sequence, {self.hidden_size}), got {tuple(hidden_states.shape)}."
            )
        query_latent = self.q_a_norm(self.q_a_proj(hidden_states))
        kv_latent = self.kv_a_norm(self.kv_a_proj(hidden_states))
        query = self._packed_linear(query_latent, self.q_b_weight)
        key = self._packed_linear(kv_latent, self.k_weight)
        value = self._packed_linear(kv_latent, self.v_weight)

        if self.attention_backend == "reference":
            attention_output, attentions = self._reference_attention(
                query,
                key,
                value,
                attention_mask,
                output_attentions,
            )
        else:
            if output_attentions:
                raise ValueError("output_attentions=True requires attention_backend='reference'.")
            attention_output = self._sdpa_attention(query, key, value, attention_mask)
            attentions = None

        gate = torch.sigmoid(self.output_gate(hidden_states).float()).view_as(attention_output)
        gated = (gate * attention_output).flatten(2)
        output = F.linear(
            gated,
            self.output_proj.weight.float(),
            None if self.output_proj.bias is None else self.output_proj.bias.float(),
        ).to(hidden_states.dtype)
        if not return_dict:
            return output, attentions
        return {
            "hidden_states": output,
            "attentions": attentions,
            "kv_latent": kv_latent,
        }


__all__ = ["GatedMLA"]
