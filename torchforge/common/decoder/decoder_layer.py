from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
from torch import nn

from torchforge.common.attention import MLA
from torchforge.common.moe import MoE
from torchforge.common.nn import FeedForward, RMSNorm


class DecoderLayer(nn.Module):
    """DeepSeek-V3-style decoder layer assembled from common components.

    Structure:
        RMSNorm -> MLA -> residual -> RMSNorm -> FeedForward/MoE -> residual.

    Args:
        hidden_size: Hidden-state dimension.
        num_attention_heads: Number of attention heads.
        num_key_value_heads: Number of key/value heads.
        intermediate_size: Feed-forward intermediate dimension.
        q_lora_rank: Query low-rank projection rank.
        kv_lora_rank: Key/value low-rank projection rank.
        qk_nope_head_dim: Non-rotary query/key head dimension.
        qk_rope_head_dim: Rotary query/key head dimension.
        v_head_dim: Value head dimension.
        ffn_type: ``"dense"`` or ``"moe"``.
        num_experts: Number of experts when ``ffn_type="moe"``.
        num_experts_per_tok: Number of selected experts per token.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        intermediate_size: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        ffn_type: str = "dense",
        num_experts: int = 4,
        num_experts_per_tok: int = 2,
        rms_norm_eps: float = 1e-6,
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        attention_bias: bool = False,
    ) -> None:
        super().__init__()
        if ffn_type not in {"dense", "moe"}:
            raise ValueError("ffn_type must be either 'dense' or 'moe'.")
        self.input_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = MLA(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            attention_dropout=attention_dropout,
            attention_bias=attention_bias,
            rms_norm_eps=rms_norm_eps,
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
        self.post_attention_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        if ffn_type == "moe":
            self.mlp = MoE(
                hidden_size=hidden_size,
                num_experts=num_experts,
                top_k=num_experts_per_tok,
                expert_intermediate_size=intermediate_size,
                router_score_function="softmax",
                normalize_topk=True,
                expert_activation="silu",
                expert_gated=True,
                bias=False,
            )
        else:
            self.mlp = FeedForward(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                activation="swiglu",
                dropout=hidden_dropout,
                bias=False,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs: Any,
    ) -> Any:
        attention_output = self.self_attn(
            self.input_norm(hidden_states),
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            **kwargs,
        )["hidden_states"]
        hidden_states = hidden_states + attention_output
        mlp_output = self.mlp(self.post_attention_norm(hidden_states))
        if isinstance(mlp_output, dict):
            mlp_output = mlp_output["hidden_states"]
        hidden_states = hidden_states + mlp_output
        if return_dict:
            return {"hidden_states": hidden_states}
        return hidden_states


__all__ = ["DecoderLayer"]

