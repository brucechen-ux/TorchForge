from __future__ import annotations

from typing import Any

from torchforge.common.attention.mla import MLA, MLAConfig


class DSV3MLAPatch:
    """Adapt DeepSeek V3 MLA config to the public common MLA component."""

    @classmethod
    def adapt_config(cls, model_config: Any, *, layer_idx: int = 0) -> MLAConfig:
        q_lora_rank = getattr(model_config, "q_lora_rank", None)
        return MLAConfig(
            hidden_size=model_config.hidden_size,
            num_attention_heads=model_config.num_attention_heads,
            num_key_value_heads=getattr(
                model_config,
                "num_key_value_heads",
                model_config.num_attention_heads,
            ),
            q_lora_rank=q_lora_rank,
            kv_lora_rank=model_config.kv_lora_rank,
            qk_nope_head_dim=model_config.qk_nope_head_dim,
            qk_rope_head_dim=model_config.qk_rope_head_dim,
            v_head_dim=model_config.v_head_dim,
            attention_dropout=getattr(model_config, "attention_dropout", 0.0),
            attention_bias=getattr(model_config, "attention_bias", False),
            rms_norm_eps=getattr(model_config, "rms_norm_eps", 1e-6),
            query_projection_type="direct" if q_lora_rank is None else "low_rank",
            query_pre_norm="none" if q_lora_rank is None else "rmsnorm",
            query_post_norm="none",
            query_store_residual=False,
            kv_projection_type="latent_kv_with_rope",
            kv_latent_norm="rmsnorm",
            kv_final_norm="none",
            kv_value_mode="projected_value",
            kv_rope_mode="split_from_first_projection",
            kv_rope_k_broadcast=True,
            rotary_layout="interleaved" if getattr(model_config, "rope_interleave", False) else "standard",
            rotary_application="explicit_split",
            position_source="tuple",
            kv_value_policy="keep_original_value",
            attention_bias_policy="pass_through",
            pad_attention_bias_to_kv_length=False,
            repeat_kv=False,
            attention_sinks=False,
            attention_scaling="qk_head_dim",
            output_projection_type="linear",
            pre_output_transform="none",
        )

    @classmethod
    def build(cls, model_config: Any, *, layer_idx: int = 0) -> MLA:
        config = cls.adapt_config(model_config, layer_idx=layer_idx)
        return MLA(
            config,
            rope_parameters=getattr(model_config, "rope_parameters", None),
        )


__all__ = ["DSV3MLAPatch"]
