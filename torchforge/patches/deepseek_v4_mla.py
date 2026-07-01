from __future__ import annotations

from typing import Any, Optional

import torch
from torch import nn

from torchforge.common.attention.mla import MLA, MLAConfig
from torchforge.common.kv import CSACompressor as _CSACompressor
from torchforge.common.kv import HCACompressor as _HCACompressor


class _MLAKVCompressorAdapter(nn.Module):
    def __init__(self, compressor: nn.Module, *, uses_q_residual: bool) -> None:
        super().__init__()
        self.compressor = compressor
        self.uses_q_residual = uses_q_residual

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Any:
        if self.uses_q_residual:
            return self.compressor(hidden_states, q_residual=q_residual, position_ids=position_ids)
        return self.compressor(hidden_states, position_ids=position_ids)


class DSV4MLAPatch:
    """Adapt DeepSeek V4 MLA config to the public common MLA component."""

    @classmethod
    def adapt_config(cls, model_config: Any, *, layer_idx: int = 0) -> MLAConfig:
        layer_type = model_config.layer_types[layer_idx]
        qk_rope_head_dim = getattr(
            model_config,
            "qk_rope_head_dim",
            int(model_config.head_dim * model_config.partial_rotary_factor),
        )
        return MLAConfig(
            hidden_size=model_config.hidden_size,
            num_attention_heads=model_config.num_attention_heads,
            num_key_value_heads=getattr(model_config, "num_key_value_heads", 1),
            q_lora_rank=model_config.q_lora_rank,
            kv_lora_rank=None,
            qk_nope_head_dim=model_config.head_dim - qk_rope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=model_config.head_dim,
            attention_dropout=getattr(model_config, "attention_dropout", 0.0),
            attention_bias=False,
            rms_norm_eps=getattr(model_config, "rms_norm_eps", 1e-6),
            query_projection_type="low_rank",
            query_pre_norm="rmsnorm",
            query_post_norm="unweighted_rmsnorm",
            query_store_residual=True,
            kv_projection_type="direct_kv",
            kv_latent_norm="none",
            kv_final_norm="rmsnorm",
            kv_value_mode="shared_with_key",
            kv_rope_mode="none",
            kv_rope_k_broadcast=False,
            rotary_layout="interleaved",
            rotary_application="partial_trailing",
            position_source="dict",
            position_key="main" if layer_type == "sliding_attention" else "compress",
            store_latest_position=True,
            kv_value_policy="value_equals_key_after_position",
            attention_bias_policy="append_block_bias",
            pad_attention_bias_to_kv_length=True,
            repeat_kv=True,
            attention_sinks=True,
            attention_scaling="qk_head_dim",
            output_projection_type="grouped_low_rank",
            pre_output_transform="inverse_rope",
            o_groups=model_config.o_groups,
            o_lora_rank=model_config.o_lora_rank,
        )

    @classmethod
    def _build_hca_compressor(cls, model_config: Any) -> _HCACompressor:
        return _HCACompressor(
            hidden_size=model_config.hidden_size,
            head_dim=model_config.head_dim,
            compress_rate=model_config.compress_rates["heavily_compressed_attention"],
            partial_rotary_factor=model_config.partial_rotary_factor,
            rope_theta=model_config.compress_rope_theta,
            rms_norm_eps=model_config.rms_norm_eps,
        )

    @classmethod
    def _build_csa_compressor(cls, model_config: Any) -> _CSACompressor:
        return _CSACompressor(
            hidden_size=model_config.hidden_size,
            q_lora_rank=model_config.q_lora_rank,
            head_dim=model_config.head_dim,
            index_num_heads=model_config.index_n_heads,
            index_head_dim=model_config.index_head_dim,
            index_top_k=model_config.index_topk,
            compress_rate=model_config.compress_rates["compressed_sparse_attention"],
            partial_rotary_factor=model_config.partial_rotary_factor,
            rope_theta=model_config.compress_rope_theta,
            rms_norm_eps=model_config.rms_norm_eps,
        )

    @classmethod
    def build(cls, model_config: Any, *, layer_idx: int = 0) -> MLA:
        config = cls.adapt_config(model_config, layer_idx=layer_idx)
        layer_type = model_config.layer_types[layer_idx]
        compressor: Optional[nn.Module]
        if layer_type == "heavily_compressed_attention":
            compressor = _MLAKVCompressorAdapter(cls._build_hca_compressor(model_config), uses_q_residual=False)
        elif layer_type == "compressed_sparse_attention":
            compressor = _MLAKVCompressorAdapter(cls._build_csa_compressor(model_config), uses_q_residual=True)
        else:
            compressor = None
        return MLA(config, kv_compressor=compressor)


__all__ = ["DSV4MLAPatch"]
