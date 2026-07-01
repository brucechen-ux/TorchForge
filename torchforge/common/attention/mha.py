from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _rotate_half_interleaved(x: torch.Tensor) -> torch.Tensor:
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


def _apply_rotary_standard(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    query = (query * cos) + (_rotate_half(query) * sin)
    key = (key * cos) + (_rotate_half(key) * sin)
    return query, key


def _apply_rotary_interleaved(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if cos.shape[-1] == query.shape[-1]:
        cos = cos[..., : cos.shape[-1] // 2]
        sin = sin[..., : sin.shape[-1] // 2]
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q1, q2 = query[..., 0::2], query[..., 1::2]
    k1, k2 = key[..., 0::2], key[..., 1::2]
    query = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
    key = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)
    return query, key


def _apply_partial_trailing_rotary_interleaved(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.repeat_interleave(2, dim=-1).unsqueeze(1)
    sin = sin.repeat_interleave(2, dim=-1).unsqueeze(1)
    rotary_dim = cos.shape[-1]
    q_pass, q_rot = query[..., :-rotary_dim], query[..., -rotary_dim:]
    k_pass, k_rot = key[..., :-rotary_dim], key[..., -rotary_dim:]
    q_rot = (q_rot.float() * cos) + (_rotate_half_interleaved(q_rot).float() * sin)
    k_rot = (k_rot.float() * cos) + (_rotate_half_interleaved(k_rot).float() * sin)
    return (
        torch.cat([q_pass, q_rot.to(query.dtype)], dim=-1),
        torch.cat([k_pass, k_rot.to(key.dtype)], dim=-1),
    )


class MHA(nn.Module):
    """Multi-head attention as a directly instantiable PyTorch module.

    Args:
        hidden_size: Size of the input and output hidden-state dimension.
        num_attention_heads: Number of attention heads.
        head_dim: Per-head query/key dimension. Defaults to ``hidden_size // num_attention_heads``.
        value_head_dim: Per-head value dimension. Defaults to ``head_dim``.
        attention_dropout: Dropout probability applied to attention weights during training.
        bias: Whether projection layers use bias.
        rotary: Whether to apply RoPE to query/key states.
        rotary_layout: RoPE layout, either ``"standard"`` or ``"interleaved"``.
        rotary_application: RoPE application mode, either ``"full"`` or ``"partial_trailing"``.
        softmax_dtype: Dtype used for attention softmax.

    Forward:
        ``hidden_states`` has shape ``(batch, sequence_length, hidden_size)``.
        ``attention_mask`` is broadcast over attention logits.
        ``position_embeddings`` is a ``(cos, sin)`` tuple when ``rotary=True``.

    Returns:
        With ``return_dict=True``, returns ``{"hidden_states", "attentions"}``.
        With ``return_dict=False``, returns ``(hidden_states, attentions)``.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        head_dim: Optional[int] = None,
        value_head_dim: Optional[int] = None,
        attention_dropout: float = 0.0,
        bias: bool = False,
        rotary: bool = True,
        rotary_layout: str = "standard",
        rotary_application: str = "full",
        softmax_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")
        if num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be positive.")
        if attention_dropout < 0.0 or attention_dropout >= 1.0:
            raise ValueError("attention_dropout must be in [0, 1).")

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_attention_heads
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_attention_heads
        self.value_head_dim = value_head_dim if value_head_dim is not None else self.head_dim
        self.attention_dropout = attention_dropout
        self.rotary = rotary
        self.rotary_layout = rotary_layout
        self.rotary_application = rotary_application
        self.softmax_dtype = softmax_dtype
        self.scaling = self.head_dim**-0.5

        if self.head_dim <= 0:
            raise ValueError("head_dim must be positive.")
        if self.value_head_dim <= 0:
            raise ValueError("value_head_dim must be positive.")
        if rotary_layout not in {"standard", "interleaved"}:
            raise ValueError("rotary_layout must be 'standard' or 'interleaved'.")
        if rotary_application not in {"full", "partial_trailing"}:
            raise ValueError("rotary_application must be 'full' or 'partial_trailing'.")

        self.q_proj = nn.Linear(hidden_size, num_attention_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, num_attention_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, num_attention_heads * self.value_head_dim, bias=bias)
        self.o_proj = nn.Linear(num_attention_heads * self.value_head_dim, hidden_size, bias=bias)

    def _shape_q(self, query: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length = query.shape[:-1]
        return query.view(batch_size, seq_length, self.num_attention_heads, self.head_dim).transpose(1, 2)

    def _shape_k(self, key: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length = key.shape[:-1]
        return key.view(batch_size, seq_length, self.num_attention_heads, self.head_dim).transpose(1, 2)

    def _shape_v(self, value: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length = value.shape[:-1]
        return value.view(batch_size, seq_length, self.num_attention_heads, self.value_head_dim).transpose(1, 2)

    def _apply_rotary(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        position_embeddings: Optional[Any],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.rotary:
            return query, key
        if position_embeddings is None:
            raise ValueError("MHA requires position_embeddings when rotary=True.")
        if not isinstance(position_embeddings, tuple) or len(position_embeddings) != 2:
            raise TypeError("position_embeddings must be a tuple of (cos, sin).")
        cos, sin = position_embeddings

        if self.rotary_application == "partial_trailing":
            if self.rotary_layout != "interleaved":
                raise ValueError("partial_trailing rotary requires rotary_layout='interleaved'.")
            return _apply_partial_trailing_rotary_interleaved(query, key, cos, sin)
        if self.rotary_layout == "interleaved":
            return _apply_rotary_interleaved(query, key, cos, sin)
        return _apply_rotary_standard(query, key, cos, sin)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Any] = None,
        output_attentions: bool = False,
        return_dict: bool = True,
        **kwargs: Any,
    ) -> Any:
        if not isinstance(hidden_states, torch.Tensor):
            raise TypeError(f"hidden_states must be a torch.Tensor, got {type(hidden_states).__name__}.")
        if hidden_states.dim() < 2:
            raise ValueError("hidden_states must have at least 2 dimensions.")

        query = self._shape_q(self.q_proj(hidden_states))
        key = self._shape_k(self.k_proj(hidden_states))
        value = self._shape_v(self.v_proj(hidden_states))
        query, key = self._apply_rotary(query, key, position_embeddings)

        attention_weights = torch.matmul(query, key.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            attention_weights = attention_weights + attention_mask

        attention_weights = F.softmax(attention_weights, dim=-1, dtype=self.softmax_dtype).to(query.dtype)
        attention_weights = F.dropout(
            attention_weights,
            p=0.0 if not self.training else self.attention_dropout,
            training=self.training,
        ).to(value.dtype)
        attention_output = torch.matmul(attention_weights, value).transpose(1, 2).contiguous()
        batch_size, seq_length = attention_output.shape[:2]
        hidden_states = self.o_proj(attention_output.reshape(batch_size, seq_length, -1))

        if not return_dict:
            return hidden_states, attention_weights if output_attentions else None
        return {
            "hidden_states": hidden_states,
            "attentions": attention_weights if output_attentions else None,
        }


__all__ = ["MHA"]
