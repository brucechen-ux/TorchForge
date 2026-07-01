from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .config import DSV3ReferenceConfig, validate_config


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    input_dtype = x.dtype
    x_fp32 = x.to(torch.float32)
    x_fp32 = x_fp32 * torch.rsqrt(x_fp32.square().mean(-1, keepdim=True) + eps)
    return weight * x_fp32.to(input_dtype)


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


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, seq_len, head_dim)


class ReferenceRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return _rms_norm(hidden_states, self.weight, self.eps)


class ReferenceDenseFFN(nn.Module):
    def __init__(self, config: DSV3ReferenceConfig) -> None:
        super().__init__()
        self.up_proj = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.dropout = config.hidden_dropout

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate, value = self.up_proj(hidden_states).chunk(2, dim=-1)
        hidden_states = F.silu(gate) * value
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = self.down_proj(hidden_states)
        return F.dropout(hidden_states, p=self.dropout, training=self.training)


class ReferenceTopKRouter(nn.Module):
    def __init__(self, config: DSV3ReferenceConfig) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.proj = nn.Linear(config.hidden_size, config.num_experts, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.proj(hidden_states.float())
        scores = F.softmax(logits, dim=-1)
        routing_weights, selected_experts = torch.topk(scores, k=self.top_k, dim=-1)
        if self.top_k > 1:
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-9)
        return routing_weights.to(hidden_states.dtype), selected_experts


class ReferenceExpertMLP(nn.Module):
    def __init__(self, config: DSV3ReferenceConfig) -> None:
        super().__init__()
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        return self.down_proj(hidden)


class ReferenceMoE(nn.Module):
    def __init__(self, config: DSV3ReferenceConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.router = ReferenceTopKRouter(config)
        self.experts = nn.ModuleList(ReferenceExpertMLP(config) for _ in range(config.num_experts))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape
        flat = hidden_states.reshape(-1, self.hidden_size)
        routing_weights, selected_experts = self.router(flat)
        routed = torch.zeros_like(flat)
        for expert_id, expert in enumerate(self.experts):
            token_mask = selected_experts == expert_id
            if not token_mask.any():
                routed = routed + expert(flat[:1]).sum() * 0.0
                continue
            token_pos, route_pos = token_mask.nonzero(as_tuple=True)
            expert_output = expert(flat[token_pos]).to(flat.dtype)
            weight = routing_weights[token_pos, route_pos].unsqueeze(-1).to(flat.dtype)
            routed.index_add_(0, token_pos, expert_output * weight)
        return routed.reshape(original_shape)


class ReferenceMLA(nn.Module):
    def __init__(self, config: DSV3ReferenceConfig) -> None:
        super().__init__()
        self.config = config
        self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=config.attention_bias)
        self.q_b_proj = nn.Linear(
            config.q_lora_rank,
            config.num_attention_heads * config.qk_head_dim,
            bias=False,
        )
        self.q_a_norm_weight = nn.Parameter(torch.ones(config.q_lora_rank))
        self.kv_a_proj_with_mqa = nn.Linear(
            config.hidden_size,
            config.kv_lora_rank + config.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        self.kv_a_norm_weight = nn.Parameter(torch.ones(config.kv_lora_rank))
        self.kv_b_proj = nn.Linear(
            config.kv_lora_rank,
            config.num_attention_heads * (config.qk_nope_head_dim + config.v_head_dim),
            bias=False,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * config.v_head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.scaling = config.qk_head_dim**-0.5

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        config = self.config
        batch_size, seq_length = hidden_states.shape[:2]
        cos, sin = position_embeddings

        q_latent = _rms_norm(self.q_a_proj(hidden_states), self.q_a_norm_weight, config.rms_norm_eps)
        query = self.q_b_proj(q_latent)
        query = query.view(batch_size, seq_length, config.num_attention_heads, config.qk_head_dim).transpose(1, 2)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_latent, k_rot = torch.split(compressed_kv, [config.kv_lora_rank, config.qk_rope_head_dim], dim=-1)
        k_latent = _rms_norm(k_latent, self.kv_a_norm_weight, config.rms_norm_eps)
        key_value = self.kv_b_proj(k_latent)
        key_value = key_value.view(
            batch_size,
            seq_length,
            config.num_attention_heads,
            config.qk_nope_head_dim + config.v_head_dim,
        ).transpose(1, 2)
        k_pass, value = torch.split(key_value, [config.qk_nope_head_dim, config.v_head_dim], dim=-1)
        k_rot = k_rot.view(batch_size, 1, seq_length, config.qk_rope_head_dim).expand(*k_pass.shape[:-1], -1)
        key = torch.cat((k_pass, k_rot), dim=-1)

        q_pass, q_rot = torch.split(query, [config.qk_nope_head_dim, config.qk_rope_head_dim], dim=-1)
        k_pass, k_rot = torch.split(key, [config.qk_nope_head_dim, config.qk_rope_head_dim], dim=-1)
        q_rot, k_rot = _apply_rotary_interleaved(q_rot, k_rot, cos, sin)
        query = torch.cat((q_pass, q_rot), dim=-1)
        key = torch.cat((k_pass, k_rot), dim=-1)

        attention_weights = torch.matmul(query, key.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            attention_weights = attention_weights + attention_mask
        attention_weights = F.softmax(attention_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attention_weights = F.dropout(attention_weights, p=config.attention_dropout, training=self.training)
        attention_output = torch.matmul(attention_weights, value).transpose(1, 2).contiguous()
        return self.o_proj(attention_output.reshape(batch_size, seq_length, -1))


class ReferenceDecoderLayer(nn.Module):
    def __init__(self, config: DSV3ReferenceConfig) -> None:
        super().__init__()
        self.input_norm = ReferenceRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = ReferenceMLA(config)
        self.post_attention_norm = ReferenceRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = ReferenceMoE(config) if config.ffn_type == "moe" else ReferenceDenseFFN(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attn_input = self.input_norm(hidden_states)
        hidden_states = hidden_states + self.self_attn(
            attn_input,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )
        hidden_states = hidden_states + self.mlp(self.post_attention_norm(hidden_states))
        return hidden_states


class RotaryEmbedding(nn.Module):
    def __init__(self, rotary_dim: int, max_position_embeddings: int = 4096, base: float = 10000.0) -> None:
        super().__init__()
        if rotary_dim <= 0 or rotary_dim % 2 != 0:
            raise ValueError("rotary_dim must be a positive even integer.")
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_position_embeddings = max_position_embeddings

    def forward(self, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        freqs = torch.einsum("bs,d->bsd", position_ids.float(), self.inv_freq.to(position_ids.device))
        return torch.cos(freqs), torch.sin(freqs)


class DSV3ReferenceModel(nn.Module):
    """Single-card PyTorch DeepSeek-V3-style causal language model."""

    def __init__(self, config: DSV3ReferenceConfig) -> None:
        super().__init__()
        validate_config(config)
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rotary_emb = RotaryEmbedding(config.qk_rope_head_dim)
        self.layers = nn.ModuleList(ReferenceDecoderLayer(config) for _ in range(config.num_hidden_layers))
        self.norm = ReferenceRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Any:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must have shape (batch, sequence_length).")
        batch_size, seq_length = input_ids.shape
        position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.rotary_emb(position_ids)
        hidden_states = self.embed_tokens(input_ids)

        if attention_mask is None:
            attention_mask = _make_causal_mask(batch_size, seq_length, hidden_states.device, hidden_states.dtype)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
            )
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.reshape(-1, self.config.vocab_size), labels.reshape(-1))

        if return_dict:
            return {"loss": loss, "logits": logits, "hidden_states": hidden_states}
        return (loss, logits) if loss is not None else (logits,)


def _make_causal_mask(batch_size: int, seq_length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    mask = torch.full((seq_length, seq_length), torch.finfo(dtype).min, device=device, dtype=dtype)
    mask = torch.triu(mask, diagonal=1)
    return mask.view(1, 1, seq_length, seq_length).expand(batch_size, 1, seq_length, seq_length)


__all__ = [
    "DSV3ReferenceConfig",
    "DSV3ReferenceModel",
    "ReferenceDecoderLayer",
    "ReferenceDenseFFN",
    "ReferenceMLA",
    "ReferenceMoE",
    "ReferenceRMSNorm",
]
