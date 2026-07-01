from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from torchforge.common.attention import MLA


@dataclass(frozen=True)
class TinyDSV3BlockConfig:
    hidden_size: int = 16
    num_attention_heads: int = 2
    num_key_value_heads: int = 2
    q_lora_rank: int = 4
    kv_lora_rank: int = 4
    qk_nope_head_dim: int = 4
    qk_rope_head_dim: int = 4
    v_head_dim: int = 8
    intermediate_size: int = 32
    rms_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    attention_bias: bool = False

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    input_dtype = x.dtype
    x_fp32 = x.to(torch.float32)
    x_fp32 = x_fp32 * torch.rsqrt(x_fp32.square().mean(-1, keepdim=True) + eps)
    return weight * x_fp32.to(input_dtype)


def _rotate_half_interleaved(x: torch.Tensor) -> torch.Tensor:
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


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


class TinyRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return _rms_norm(hidden_states, self.weight, self.eps)


class TinyFFN(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class ReferenceDSV3Attention(nn.Module):
    def __init__(self, config: TinyDSV3BlockConfig) -> None:
        super().__init__()
        self.config = config
        self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=config.attention_bias)
        self.q_a_norm_weight = nn.Parameter(torch.ones(config.q_lora_rank))
        self.q_b_proj = nn.Linear(
            config.q_lora_rank,
            config.num_attention_heads * config.qk_head_dim,
            bias=False,
        )
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

    def forward(self, hidden_states: torch.Tensor, position_embeddings: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
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
        attention_weights = F.softmax(attention_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attention_output = torch.matmul(attention_weights, value).transpose(1, 2).contiguous()
        return self.o_proj(attention_output.reshape(batch_size, seq_length, -1))


class ReferenceDSV3Block(nn.Module):
    def __init__(self, config: TinyDSV3BlockConfig) -> None:
        super().__init__()
        self.input_norm = TinyRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attention = ReferenceDSV3Attention(config)
        self.post_attention_norm = TinyRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn = TinyFFN(config.hidden_size, config.intermediate_size)

    def forward(self, hidden_states: torch.Tensor, position_embeddings: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        hidden_states = hidden_states + self.attention(self.input_norm(hidden_states), position_embeddings)
        return hidden_states + self.ffn(self.post_attention_norm(hidden_states))


class TorchForgeDSV3Block(nn.Module):
    def __init__(self, config: TinyDSV3BlockConfig) -> None:
        super().__init__()
        self.input_norm = TinyRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attention = MLA(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            attention_dropout=config.attention_dropout,
            attention_bias=config.attention_bias,
            rms_norm_eps=config.rms_norm_eps,
            query_projection_type="low_rank",
            query_pre_norm="rmsnorm",
            kv_projection_type="latent_kv_with_rope",
            kv_latent_norm="rmsnorm",
            rotary_layout="interleaved",
            rotary_application="explicit_split",
            position_source="tuple",
            attention_scaling="qk_head_dim",
            output_projection_type="linear",
        )
        self.post_attention_norm = TinyRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn = TinyFFN(config.hidden_size, config.intermediate_size)

    def forward(self, hidden_states: torch.Tensor, position_embeddings: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        attention_out = self.attention(self.input_norm(hidden_states), position_embeddings=position_embeddings)
        hidden_states = hidden_states + attention_out["hidden_states"]
        return hidden_states + self.ffn(self.post_attention_norm(hidden_states))


def _copy_reference_weights_to_torchforge(reference: ReferenceDSV3Block, target: TorchForgeDSV3Block) -> None:
    target.input_norm.load_state_dict(reference.input_norm.state_dict())
    target.post_attention_norm.load_state_dict(reference.post_attention_norm.state_dict())
    target.ffn.load_state_dict(reference.ffn.state_dict())

    source_attention = reference.attention
    target_attention = target.attention
    target_attention.query_projection.q_a_proj.load_state_dict(source_attention.q_a_proj.state_dict())
    target_attention.query_projection.q_a_norm_weight.data.copy_(source_attention.q_a_norm_weight.data)
    target_attention.query_projection.q_b_proj.load_state_dict(source_attention.q_b_proj.state_dict())
    target_attention.kv_projection.kv_a_proj_with_mqa.load_state_dict(source_attention.kv_a_proj_with_mqa.state_dict())
    target_attention.kv_projection.kv_a_norm_weight.data.copy_(source_attention.kv_a_norm_weight.data)
    target_attention.kv_projection.kv_b_proj.load_state_dict(source_attention.kv_b_proj.state_dict())
    target_attention.output_projection.o_proj.load_state_dict(source_attention.o_proj.state_dict())


def test_incremental_component_replacement_dsv3_mla_stage_matches_reference() -> None:
    torch.manual_seed(1234)
    config = TinyDSV3BlockConfig()
    reference = ReferenceDSV3Block(config)
    replacement = TorchForgeDSV3Block(config)
    _copy_reference_weights_to_torchforge(reference, replacement)
    reference.eval()
    replacement.eval()

    batch_size = 2
    seq_length = 3
    hidden_states = torch.randn(batch_size, seq_length, config.hidden_size)
    cos = torch.randn(batch_size, seq_length, config.qk_rope_head_dim)
    sin = torch.randn(batch_size, seq_length, config.qk_rope_head_dim)

    reference_output = reference(hidden_states, (cos, sin))
    replacement_output = replacement(hidden_states, (cos, sin))

    assert reference_output.shape == replacement_output.shape == (batch_size, seq_length, config.hidden_size)
    torch.testing.assert_close(replacement_output, reference_output, rtol=1e-5, atol=1e-5)
