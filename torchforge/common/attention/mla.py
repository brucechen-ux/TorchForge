from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .rotary import (
    apply_rotary_interleaved as _apply_rotary_interleaved,
    apply_rotary_interleaved_single as _apply_rotary_interleaved_single,
    apply_rotary_standard as _apply_rotary_standard,
)


@dataclass(frozen=True)
class MLAConfig:
    """Configuration for the public MLA component.

    The base dimensions are model-neutral. Strategy and policy fields configure
    common MLA submodules without requiring model-specific projection modules.
    """

    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    q_lora_rank: Optional[int]
    kv_lora_rank: Optional[int]
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    attention_dropout: float = 0.0
    attention_bias: bool = False
    rms_norm_eps: float = 1e-6

    query_projection_type: str = "auto"
    query_pre_norm: str = "rmsnorm"
    query_post_norm: str = "none"
    query_store_residual: bool = False

    kv_projection_type: str = "auto"
    kv_latent_norm: str = "rmsnorm"
    kv_final_norm: str = "none"
    kv_value_mode: str = "projected_value"
    kv_rope_mode: str = "split_from_first_projection"
    kv_rope_k_broadcast: bool = True

    rotary_layout: str = "standard"
    rotary_application: str = "explicit_split"
    position_source: str = "tuple"
    position_key: Optional[str] = None
    store_latest_position: bool = False

    kv_value_policy: str = "keep_original_value"
    attention_bias_policy: str = "pass_through"
    pad_attention_bias_to_kv_length: bool = False

    repeat_kv: bool = False
    attention_sinks: bool = False
    attention_scaling: str = "qk_head_dim"
    attention_implementation: str = "eager"
    softmax_dtype: torch.dtype = torch.float32

    output_projection_type: str = "linear"
    pre_output_transform: str = "none"
    o_groups: int = 1
    o_lora_rank: Optional[int] = None

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim


@dataclass
class MLATensors:
    """Container passed between MLA lifecycle stages."""

    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    attention_bias: Optional[torch.Tensor] = None
    state: Optional[dict[str, Any]] = None


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    input_dtype = x.dtype
    x_fp32 = x.to(torch.float32)
    x_fp32 = x_fp32 * torch.rsqrt(x_fp32.square().mean(-1, keepdim=True) + eps)
    return weight * x_fp32.to(input_dtype)


def _unweighted_rms_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps).to(x.dtype)


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, seq_len, head_dim)


def _yarn_get_mscale(scale: float = 1.0, mscale: float = 1.0) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class QueryProjection(nn.Module):
    """Common MLA query projection."""

    def __init__(self, config: MLAConfig) -> None:
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.head_dim = config.qk_head_dim
        self.q_lora_rank = config.q_lora_rank
        self.projection_type = (
            "direct" if config.query_projection_type == "auto" and config.q_lora_rank is None else
            "low_rank" if config.query_projection_type == "auto" else
            config.query_projection_type
        )
        self.pre_norm = config.query_pre_norm
        self.post_norm = config.query_post_norm
        self.store_residual = config.query_store_residual
        self.rms_norm_eps = config.rms_norm_eps
        self.last_q_residual: Optional[torch.Tensor] = None

        if self.projection_type == "direct":
            self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        elif self.projection_type == "low_rank":
            if self.q_lora_rank is None:
                raise ValueError("low_rank query projection requires q_lora_rank.")
            self.q_a_proj = nn.Linear(config.hidden_size, self.q_lora_rank, bias=config.attention_bias)
            self.q_b_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.head_dim, bias=False)
            if self.pre_norm == "rmsnorm":
                self.q_a_norm_weight = nn.Parameter(torch.ones(self.q_lora_rank))
        else:
            raise ValueError(f"Unsupported query_projection_type: {self.projection_type!r}.")

    def forward(self, hidden_states: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        query_shape = (*input_shape, self.num_heads, self.head_dim)

        if self.projection_type == "direct":
            query = self.q_proj(hidden_states)
        else:
            residual = self.q_a_proj(hidden_states)
            if self.pre_norm == "rmsnorm":
                residual = _rms_norm(residual, self.q_a_norm_weight, self.rms_norm_eps)
            elif self.pre_norm != "none":
                raise ValueError(f"Unsupported query pre_norm: {self.pre_norm!r}.")
            self.last_q_residual = residual if self.store_residual else None
            query = self.q_b_proj(residual)

        query = query.view(query_shape).transpose(1, 2)
        if self.post_norm == "unweighted_rmsnorm":
            query = _unweighted_rms_norm(query, self.rms_norm_eps)
        elif self.post_norm != "none":
            raise ValueError(f"Unsupported query post_norm: {self.post_norm!r}.")
        return query


class KVProjection(nn.Module):
    """Common MLA key/value projection."""

    def __init__(self, config: MLAConfig) -> None:
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = config.qk_head_dim
        self.v_head_dim = config.v_head_dim
        self.rms_norm_eps = config.rms_norm_eps
        self.projection_type = (
            "latent_kv_with_rope" if config.kv_projection_type == "auto" and config.kv_lora_rank is not None else
            "direct_kv" if config.kv_projection_type == "auto" else
            config.kv_projection_type
        )

        if self.projection_type == "latent_kv_with_rope":
            if self.kv_lora_rank is None:
                raise ValueError("latent_kv_with_rope projection requires kv_lora_rank.")
            self.kv_a_proj_with_mqa = nn.Linear(
                config.hidden_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
                bias=config.attention_bias,
            )
            if config.kv_latent_norm == "rmsnorm":
                self.kv_a_norm_weight = nn.Parameter(torch.ones(self.kv_lora_rank))
            self.kv_b_proj = nn.Linear(
                self.kv_lora_rank,
                self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
                bias=False,
            )
        elif self.projection_type == "direct_kv":
            if config.kv_value_mode == "shared_with_key":
                self.kv_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.qk_head_dim, bias=False)
            elif config.kv_value_mode == "projected_value":
                self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.qk_head_dim, bias=False)
                self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.v_head_dim, bias=False)
            else:
                raise ValueError(f"Unsupported kv value_mode: {config.kv_value_mode!r}.")
            if config.kv_final_norm == "rmsnorm":
                self.kv_norm_weight = nn.Parameter(torch.ones(self.qk_head_dim))
        else:
            raise ValueError(f"Unsupported kv_projection_type: {self.projection_type!r}.")

    def forward(self, hidden_states: torch.Tensor, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]

        if self.projection_type == "direct_kv":
            if self.config.kv_value_mode == "shared_with_key":
                kv = self.kv_proj(hidden_states).view(*input_shape, self.num_key_value_heads, self.qk_head_dim)
                if self.config.kv_final_norm == "rmsnorm":
                    kv = _rms_norm(kv, self.kv_norm_weight, self.rms_norm_eps)
                elif self.config.kv_final_norm != "none":
                    raise ValueError(f"Unsupported kv final norm: {self.config.kv_final_norm!r}.")
                kv = kv.transpose(1, 2)
                return kv, kv

            key = self.k_proj(hidden_states).view(*input_shape, self.num_key_value_heads, self.qk_head_dim)
            value = self.v_proj(hidden_states).view(*input_shape, self.num_key_value_heads, self.v_head_dim)
            if self.config.kv_final_norm == "rmsnorm":
                key = _rms_norm(key, self.kv_norm_weight, self.rms_norm_eps)
            elif self.config.kv_final_norm != "none":
                raise ValueError(f"Unsupported kv final norm: {self.config.kv_final_norm!r}.")
            return key.transpose(1, 2), value.transpose(1, 2)

        batch_size, seq_length = hidden_states.shape[:-1]
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_latent, k_rot = torch.split(
            compressed_kv,
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1,
        )
        if self.config.kv_latent_norm == "rmsnorm":
            k_latent = _rms_norm(k_latent, self.kv_a_norm_weight, self.rms_norm_eps)
        elif self.config.kv_latent_norm != "none":
            raise ValueError(f"Unsupported kv latent norm: {self.config.kv_latent_norm!r}.")

        key_value_shape = (
            batch_size,
            seq_length,
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        key_value = self.kv_b_proj(k_latent).view(key_value_shape).transpose(1, 2)
        k_pass, value = torch.split(key_value, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        if self.config.kv_rope_mode != "split_from_first_projection":
            raise ValueError(f"Unsupported kv rope mode: {self.config.kv_rope_mode!r}.")
        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)
        if self.config.kv_rope_k_broadcast:
            k_rot = k_rot.expand(*k_pass.shape[:-1], -1)
        key = torch.cat((k_pass, k_rot), dim=-1)
        return key, value


class RotaryPosition(nn.Module):
    """Common partial/full RoPE application."""

    def __init__(self, config: MLAConfig) -> None:
        super().__init__()
        self.config = config
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.layout = config.rotary_layout
        self.application = config.rotary_application
        self.position_source = config.position_source
        self.position_key = config.position_key
        self.latest_position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def _select_embeddings(self, position_embeddings: Optional[Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        if position_embeddings is None:
            raise ValueError("MLA rotary position requires position_embeddings.")
        if self.position_source == "dict":
            if not isinstance(position_embeddings, dict):
                raise TypeError("position_embeddings must be a dict for position_source='dict'.")
            if self.position_key is None:
                raise ValueError("position_key is required for position_source='dict'.")
            return position_embeddings[self.position_key]
        if self.position_source == "tuple":
            if not isinstance(position_embeddings, tuple) or len(position_embeddings) != 2:
                raise TypeError("position_embeddings must be a tuple of (cos, sin).")
            return position_embeddings
        raise ValueError(f"Unsupported position_source: {self.position_source!r}.")

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        *,
        position_ids: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Any] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cos, sin = self._select_embeddings(position_embeddings)
        if self.config.store_latest_position:
            self.latest_position_embeddings = (cos, sin)

        if self.application == "explicit_split":
            q_pass, q_rot = torch.split(query, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
            k_pass, k_rot = torch.split(key, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
            if self.layout == "interleaved":
                q_rot, k_rot = _apply_rotary_interleaved(q_rot, k_rot, cos, sin)
            elif self.layout == "standard":
                q_rot, k_rot = _apply_rotary_standard(q_rot, k_rot, cos, sin)
            else:
                raise ValueError(f"Unsupported rotary layout: {self.layout!r}.")
            return torch.cat((q_pass, q_rot), dim=-1), torch.cat((k_pass, k_rot), dim=-1)

        if self.application == "partial_trailing":
            if self.layout != "interleaved":
                raise ValueError("partial_trailing rotary currently requires interleaved layout.")
            return _apply_rotary_interleaved_single(query, cos, sin), _apply_rotary_interleaved_single(key, cos, sin)

        raise ValueError(f"Unsupported rotary application: {self.application!r}.")


class KVAugment(nn.Module):
    """Common KV augmentation stage with optional specialized compressor."""

    def __init__(
        self,
        config: MLAConfig,
        *,
        compressor: Optional[nn.Module] = None,
        query_projection: Optional[QueryProjection] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.compressor = compressor
        self.query_projection = query_projection

    def forward(
        self,
        tensors: MLATensors,
        *,
        hidden_states: torch.Tensor,
        cache: Optional[Any] = None,
        **kwargs: Any,
    ) -> MLATensors:
        if self.config.kv_value_policy == "value_equals_key_after_position":
            tensors.value = tensors.key
        elif self.config.kv_value_policy != "keep_original_value":
            raise ValueError(f"Unsupported kv value policy: {self.config.kv_value_policy!r}.")

        if self.compressor is None:
            return tensors
        position_ids = kwargs.get("position_ids")
        if position_ids is None:
            raise ValueError("Compressed KV augmentation requires position_ids.")
        if self.query_projection is None or self.query_projection.last_q_residual is None:
            raise RuntimeError("Compressed KV augmentation requires query residual from QueryProjection.")

        original_len = tensors.key.shape[2]
        compressed_kv, block_bias = self.compressor(hidden_states, self.query_projection.last_q_residual, position_ids)
        tensors.key = torch.cat([tensors.key, compressed_kv], dim=2)
        tensors.value = tensors.key
        tensors.state = {
            **(tensors.state or {}),
            "block_bias": block_bias,
            "original_kv_len": original_len,
            "compressed_kv_len": compressed_kv.shape[2],
        }
        return tensors


class AttentionBias(nn.Module):
    """Common attention-bias merge stage."""

    def __init__(self, config: MLAConfig) -> None:
        super().__init__()
        self.config = config

    def forward(
        self,
        tensors: MLATensors,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> MLATensors:
        block_bias = (tensors.state or {}).get("block_bias")
        compressed_len = (tensors.state or {}).get("compressed_kv_len", 0)
        if self.config.attention_bias_policy == "append_block_bias" and compressed_len and block_bias is not None:
            if attention_mask is not None:
                tensors.attention_bias = torch.cat([attention_mask, block_bias.to(attention_mask.dtype)], dim=-1)
            else:
                batch, _, query_len, _ = block_bias.shape
                original_len = (tensors.state or {})["original_kv_len"]
                zeros = block_bias.new_zeros((batch, 1, query_len, original_len))
                tensors.attention_bias = torch.cat([zeros, block_bias], dim=-1)
        elif (
            self.config.pad_attention_bias_to_kv_length
            and attention_mask is not None
            and tensors.key.shape[2] > attention_mask.shape[-1]
        ):
            tensors.attention_bias = F.pad(attention_mask, (0, tensors.key.shape[2] - attention_mask.shape[-1]), value=0.0)
        else:
            tensors.attention_bias = attention_mask
        return tensors


class AttentionBackend(nn.Module):
    """Common eager attention backend with optional GQA repeat and sinks."""

    def __init__(self, config: MLAConfig, *, rope_parameters: Optional[dict[str, Any]] = None) -> None:
        super().__init__()
        self.config = config
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.attention_dropout = config.attention_dropout
        if config.attention_scaling == "qk_head_dim":
            self.scaling = config.qk_head_dim ** -0.5
        elif config.attention_scaling == "v_head_dim":
            self.scaling = config.v_head_dim ** -0.5
        else:
            self.scaling = float(config.attention_scaling)

        rope_parameters = rope_parameters or {}
        if rope_parameters.get("rope_type", "default") != "default":
            mscale_all_dim = rope_parameters.get("mscale_all_dim", 0)
            scaling_factor = rope_parameters["factor"]
            if mscale_all_dim:
                mscale = _yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.scaling = self.scaling * mscale * mscale

        if config.attention_sinks:
            self.sinks = nn.Parameter(torch.zeros(config.num_attention_heads))
        else:
            self.register_parameter("sinks", None)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        attention_bias: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        key_states = _repeat_kv(key, self.num_key_value_groups) if self.config.repeat_kv else key
        value_states = _repeat_kv(value, self.num_key_value_groups) if self.config.repeat_kv else value
        if self.config.attention_implementation == "sdpa":
            if self.sinks is None:
                output = F.scaled_dot_product_attention(
                    query.contiguous(),
                    key_states.contiguous(),
                    value_states.contiguous(),
                    attn_mask=attention_bias,
                    dropout_p=self.attention_dropout if self.training else 0.0,
                    scale=self.scaling,
                )
                return output.transpose(1, 2).contiguous(), None

            batch, heads, _, head_dim = query.shape
            padded_head_dim = ((head_dim + 1 + 7) // 8) * 8
            query_states = query.new_zeros((*query.shape[:-1], padded_head_dim))
            query_states[..., :head_dim] = query
            query_states[..., head_dim] = 1.0

            padded_key_states = key_states.new_zeros((*key_states.shape[:-1], padded_head_dim))
            padded_key_states[..., :head_dim] = key_states
            sink_key = key_states.new_zeros((batch, heads, 1, padded_head_dim))
            sink_key[..., head_dim] = (self.sinks / self.scaling).to(key_states.dtype).reshape(1, heads, 1)
            padded_key_states = torch.cat([padded_key_states, sink_key], dim=2)
            sink_value = value_states.new_zeros((batch, heads, 1, value_states.shape[-1]))
            value_states = torch.cat([value_states, sink_value], dim=2)
            if attention_bias is not None:
                sink_bias = attention_bias.new_zeros((*attention_bias.shape[:-1], 1))
                attention_bias = torch.cat([attention_bias, sink_bias], dim=-1)
            output = F.scaled_dot_product_attention(
                query_states.contiguous(),
                padded_key_states.contiguous(),
                value_states.contiguous(),
                attn_mask=attention_bias,
                dropout_p=self.attention_dropout if self.training else 0.0,
                scale=self.scaling,
            )
            return output.transpose(1, 2).contiguous(), None

        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * self.scaling
        if attention_bias is not None:
            attn_weights = attn_weights + attention_bias

        if self.sinks is not None:
            sinks = self.sinks.reshape(1, -1, 1, 1).expand(query.shape[0], -1, query.shape[-2], -1)
            combined_logits = torch.cat([attn_weights, sinks], dim=-1)
            combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
            probs = F.softmax(combined_logits, dim=-1, dtype=combined_logits.dtype)
            attn_weights = probs[..., :-1]
        else:
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=self.config.softmax_dtype).to(query.dtype)

        attn_weights = F.dropout(
            attn_weights,
            p=0.0 if not self.training else self.attention_dropout,
            training=self.training,
        ).to(value_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        return attn_output.transpose(1, 2).contiguous(), attn_weights if output_attentions else None


class GroupedLinear(nn.Module):
    def __init__(self, in_features_per_group: int, out_features: int, n_groups: int, bias: bool = False) -> None:
        super().__init__()
        self.n_groups = n_groups
        self.weight = nn.Parameter(torch.empty(out_features, in_features_per_group))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)  # same as nn.Linear default

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape[:-2]
        hidden_dim = x.shape[-1]
        weight = self.weight.view(self.n_groups, -1, hidden_dim).transpose(1, 2)
        output = torch.bmm(x.reshape(-1, self.n_groups, hidden_dim).transpose(0, 1), weight).transpose(0, 1)
        output = output.reshape(*input_shape, self.n_groups, -1)
        if self.bias is not None:
            output = output + self.bias.view(self.n_groups, -1)
        return output


class OutputProjection(nn.Module):
    """Common MLA output projection."""

    def __init__(self, config: MLAConfig, *, position: Optional[RotaryPosition] = None) -> None:
        super().__init__()
        self.config = config
        self.position = position
        self.projection_type = config.output_projection_type
        if self.projection_type == "linear":
            self.o_proj = nn.Linear(
                config.num_attention_heads * config.v_head_dim,
                config.hidden_size,
                bias=config.attention_bias,
            )
        elif self.projection_type == "grouped_low_rank":
            if config.o_lora_rank is None:
                raise ValueError("grouped_low_rank output projection requires o_lora_rank.")
            self.o_a_proj = GroupedLinear(
                config.num_attention_heads * config.qk_head_dim // config.o_groups,
                config.o_groups * config.o_lora_rank,
                config.o_groups,
            )
            self.o_b_proj = nn.Linear(config.o_groups * config.o_lora_rank, config.hidden_size, bias=False)
        else:
            raise ValueError(f"Unsupported output_projection_type: {self.projection_type!r}.")

    def forward(self, attention_output: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        if self.config.pre_output_transform == "inverse_rope":
            if self.position is None or self.position.latest_position_embeddings is None:
                raise RuntimeError("inverse_rope output transform requires RotaryPosition to run first.")
            cos, sin = self.position.latest_position_embeddings
            attention_output = _apply_rotary_interleaved_single(attention_output.transpose(1, 2), cos, -sin).transpose(1, 2)
        elif self.config.pre_output_transform != "none":
            raise ValueError(f"Unsupported pre_output_transform: {self.config.pre_output_transform!r}.")

        batch_size, seq_length = attention_output.shape[:2]
        if self.projection_type == "linear":
            return self.o_proj(attention_output.reshape(batch_size, seq_length, -1).contiguous())

        grouped = attention_output.reshape(batch_size, seq_length, self.config.o_groups, -1)
        grouped = self.o_a_proj(grouped).flatten(2)
        return self.o_b_proj(grouped)


def _validate_config(config: MLAConfig) -> None:
    positive_fields = (
        "hidden_size",
        "num_attention_heads",
        "num_key_value_heads",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "v_head_dim",
        "o_groups",
    )
    for name in positive_fields:
        value = getattr(config, name)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"MLAConfig.{name} must be a positive int, got {value!r}.")
    if config.q_lora_rank is not None and config.q_lora_rank <= 0:
        raise ValueError("MLAConfig.q_lora_rank must be positive when set.")
    if config.kv_lora_rank is not None and config.kv_lora_rank <= 0:
        raise ValueError("MLAConfig.kv_lora_rank must be positive when set.")
    if config.o_lora_rank is not None and config.o_lora_rank <= 0:
        raise ValueError("MLAConfig.o_lora_rank must be positive when set.")
    if config.attention_dropout < 0.0 or config.attention_dropout >= 1.0:
        raise ValueError("MLAConfig.attention_dropout must be in [0, 1).")
    if config.num_attention_heads % config.num_key_value_heads != 0:
        raise ValueError("num_attention_heads must be divisible by num_key_value_heads.")
    if config.attention_implementation not in {"eager", "sdpa"}:
        raise ValueError("MLAConfig.attention_implementation must be 'eager' or 'sdpa'.")


def _require_tensor(name: str, value: Any) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}.")
    return value


def _require_pair(name: str, value: Any) -> Tuple[Any, Any]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError(f"{name} must return a tuple of length 2.")
    return value


def _require_mla_tensors(stage: str, value: Any) -> MLATensors:
    if not isinstance(value, MLATensors):
        raise TypeError(f"{stage} must return MLATensors, got {type(value).__name__}.")
    _require_tensor(f"{stage}.query", value.query)
    _require_tensor(f"{stage}.key", value.key)
    _require_tensor(f"{stage}.value", value.value)
    if value.attention_bias is not None:
        _require_tensor(f"{stage}.attention_bias", value.attention_bias)
    if value.state is not None and not isinstance(value.state, dict):
        raise TypeError(f"{stage}.state must be a dict when set.")
    _validate_attention_shapes(stage, value.query, value.key, value.value)
    return value


def _validate_attention_shapes(
    stage: str,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> None:
    if query.dim() < 3:
        raise ValueError(f"{stage}.query must have at least 3 dimensions.")
    if key.dim() != query.dim() or value.dim() != query.dim():
        raise ValueError(
            f"{stage}.query/key/value must have the same rank, got "
            f"{query.dim()}, {key.dim()}, {value.dim()}."
        )
    if query.shape[0] != key.shape[0] or key.shape[0] != value.shape[0]:
        raise ValueError(f"{stage}.query/key/value batch dimensions must match.")
    if query.shape[-1] != key.shape[-1]:
        raise ValueError(
            f"{stage}.query and {stage}.key must share the attention head dimension, "
            f"got {query.shape[-1]} and {key.shape[-1]}."
        )
    if key.shape[:-1] != value.shape[:-1]:
        raise ValueError(
            f"{stage}.key and {stage}.value must match except for the last dimension, "
            f"got {tuple(key.shape)} and {tuple(value.shape)}."
        )


class MLA(nn.Module):
    """Multi-head Latent Attention as a directly instantiable PyTorch module.

    Args:
        config: Optional ``MLAConfig``. When omitted, keyword configuration is used.
        hidden_size: Size of the input and output hidden-state dimension.
        num_attention_heads: Number of query attention heads.
        num_key_value_heads: Number of key/value heads.
        q_lora_rank: Optional low-rank query projection rank.
        kv_lora_rank: Optional low-rank key/value projection rank.
        qk_nope_head_dim: Non-rotary query/key head dimension.
        qk_rope_head_dim: Rotary query/key head dimension.
        v_head_dim: Value head dimension.
        rope_parameters: Optional RoPE scaling parameters for attention scaling.
        kv_compressor: Optional KV compressor module used by compressed KV augmentation.
        **kwargs: Additional ``MLAConfig`` strategy and policy fields.

    Forward:
        ``hidden_states`` has shape ``(batch, sequence_length, hidden_size)``.
        ``attention_mask`` is broadcast over attention logits.
        ``position_embeddings`` is a ``(cos, sin)`` tuple or a dict selected by config.
        ``position_ids`` is required when a KV compressor is configured.

    Returns:
        With ``return_dict=True``, returns ``{"hidden_states", "attentions", "cache"}``.
        With ``return_dict=False``, returns ``(hidden_states, attentions)``.
    """

    def __init__(
        self,
        config: Optional[MLAConfig] = None,
        *,
        hidden_size: Optional[int] = None,
        num_attention_heads: Optional[int] = None,
        num_key_value_heads: Optional[int] = None,
        q_lora_rank: Optional[int] = None,
        kv_lora_rank: Optional[int] = None,
        qk_nope_head_dim: Optional[int] = None,
        qk_rope_head_dim: Optional[int] = None,
        v_head_dim: Optional[int] = None,
        rope_parameters: Optional[dict[str, Any]] = None,
        kv_compressor: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        if config is None:
            required = {
                "hidden_size": hidden_size,
                "num_attention_heads": num_attention_heads,
                "num_key_value_heads": num_key_value_heads,
                "qk_nope_head_dim": qk_nope_head_dim,
                "qk_rope_head_dim": qk_rope_head_dim,
                "v_head_dim": v_head_dim,
            }
            missing = [name for name, value in required.items() if value is None]
            if missing:
                raise TypeError(f"MLA missing required arguments: {', '.join(missing)}.")
            config = MLAConfig(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                q_lora_rank=q_lora_rank,
                kv_lora_rank=kv_lora_rank,
                qk_nope_head_dim=qk_nope_head_dim,
                qk_rope_head_dim=qk_rope_head_dim,
                v_head_dim=v_head_dim,
                **kwargs,
            )
        elif kwargs:
            raise TypeError("Pass either MLAConfig or keyword configuration, not both.")

        super().__init__()
        _validate_config(config)
        self.config = config
        self.query_projection = QueryProjection(config)
        self.kv_projection = KVProjection(config)
        self.rotary = RotaryPosition(config)
        self.kv_augment = KVAugment(config, compressor=kv_compressor, query_projection=self.query_projection)
        self.attention_bias = AttentionBias(config)
        self.attention_backend = AttentionBackend(config, rope_parameters=rope_parameters)
        self.output_projection = OutputProjection(config, position=self.rotary)

    def _require_tensor(self, name: str, value: Any) -> torch.Tensor:
        return _require_tensor(name, value)

    def _require_pair(self, name: str, value: Any) -> Tuple[Any, Any]:
        return _require_pair(name, value)

    def _require_mla_tensors(self, stage: str, value: Any) -> MLATensors:
        return _require_mla_tensors(stage, value)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Any] = None,
        cache: Optional[Any] = None,
        output_attentions: bool = False,
        return_dict: bool = True,
        **kwargs: Any,
    ) -> Any:
        self._require_tensor("hidden_states", hidden_states)
        if hidden_states.dim() < 2:
            raise ValueError("hidden_states must have at least 2 dimensions.")

        query = self.query_projection(hidden_states, **kwargs)
        query = self._require_tensor("query_projection output", query)

        key, value = self._require_pair("kv_projection", self.kv_projection(hidden_states, **kwargs))
        key = self._require_tensor("kv_projection key", key)
        value = self._require_tensor("kv_projection value", value)

        query, key = self._require_pair(
            "rotary",
            self.rotary(
                query,
                key,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                **kwargs,
            ),
        )
        query = self._require_tensor("rotary query", query)
        key = self._require_tensor("rotary key", key)

        tensors = MLATensors(query=query, key=key, value=value)
        self._require_mla_tensors("after_rotary", tensors)

        tensors = self.kv_augment(
            tensors,
            hidden_states=hidden_states,
            cache=cache,
            position_ids=position_ids,
            **kwargs,
        )
        tensors = self._require_mla_tensors("kv_augment", tensors)

        tensors = self.attention_bias(tensors, attention_mask=attention_mask, **kwargs)
        tensors = self._require_mla_tensors("attention_bias", tensors)

        attention_output, attention_weights = self._require_pair(
            "attention_backend",
            self.attention_backend(
                tensors.query,
                tensors.key,
                tensors.value,
                attention_bias=tensors.attention_bias,
                output_attentions=output_attentions,
                **kwargs,
            ),
        )
        attention_output = self._require_tensor("attention_backend output", attention_output)
        if attention_weights is not None:
            self._require_tensor("attention_backend attentions", attention_weights)

        output = self.output_projection(attention_output, **kwargs)
        output = self._require_tensor("output_projection output", output)

        if not return_dict:
            return output, attention_weights
        return {
            "hidden_states": output,
            "attentions": attention_weights,
            "cache": cache,
        }


__all__ = ["MLA"]
