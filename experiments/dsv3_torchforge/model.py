from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from experiments.dsv3_reference.config import DSV3ReferenceConfig, validate_config
from experiments.dsv3_reference.model import (
    DSV3ReferenceModel,
    ReferenceDecoderLayer,
    ReferenceDenseFFN,
    ReferenceMLA,
    ReferenceMoE,
    ReferenceRMSNorm,
    RotaryEmbedding,
    _make_causal_mask,
)
from torchforge.common.attention import MLA
from torchforge.common.moe import MoE
from torchforge.common.nn import FeedForward, RMSNorm


class ComponentConfig:
    def __init__(
        self,
        *,
        attention: str = "pytorch",
        norm: str = "pytorch",
        ffn: str = "pytorch",
        kv: str = "pytorch",
    ) -> None:
        self.attention = attention
        self.norm = norm
        self.ffn = ffn
        self.kv = kv
        self._validate()

    def _validate(self) -> None:
        if self.attention not in {"pytorch", "torchforge"}:
            raise ValueError("attention must be either 'pytorch' or 'torchforge'.")
        if self.norm not in {"pytorch", "torchforge"}:
            raise ValueError("norm must be either 'pytorch' or 'torchforge'.")
        if self.ffn not in {"pytorch", "torchforge", "moe"}:
            raise ValueError("ffn must be one of 'pytorch', 'torchforge', or 'moe'.")
        if self.kv not in {"pytorch", "torchforge"}:
            raise ValueError("kv must be either 'pytorch' or 'torchforge'.")

    def to_dict(self) -> dict[str, str]:
        return {
            "attention": self.attention,
            "norm": self.norm,
            "ffn": self.ffn,
            "kv": self.kv,
        }


def variant_name(components: ComponentConfig) -> str:
    return (
        f"attention_{components.attention}"
        f"__ffn_{components.ffn}"
        f"__norm_{components.norm}"
        f"__kv_{components.kv}"
    )


class TorchForgeMLAWrapper(nn.Module):
    def __init__(self, config: DSV3ReferenceConfig) -> None:
        super().__init__()
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
        )["hidden_states"]


class TorchForgeMoEWrapper(nn.Module):
    def __init__(self, config: DSV3ReferenceConfig) -> None:
        super().__init__()
        self.moe = MoE(
            hidden_size=config.hidden_size,
            num_experts=config.num_experts,
            top_k=config.top_k,
            expert_intermediate_size=config.intermediate_size,
            router_score_function="softmax",
            normalize_topk=True,
            expert_activation="silu",
            expert_gated=True,
            bias=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.moe(hidden_states)["hidden_states"]


class TorchForgeDecoderLayer(nn.Module):
    def __init__(self, config: DSV3ReferenceConfig, components: ComponentConfig) -> None:
        super().__init__()
        self.input_norm = _build_norm(config, components.norm)
        self.self_attn = _build_attention(config, components.attention)
        self.post_attention_norm = _build_norm(config, components.norm)
        self.mlp = _build_ffn(config, components.ffn)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(
            self.input_norm(hidden_states),
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )
        hidden_states = hidden_states + self.mlp(self.post_attention_norm(hidden_states))
        return hidden_states


class DSV3TorchForgeModel(nn.Module):
    """DeepSeek-V3-style causal LM with configurable TorchForge replacements."""

    def __init__(self, config: DSV3ReferenceConfig, components: ComponentConfig) -> None:
        super().__init__()
        validate_config(config)
        if components.kv != "pytorch":
            # Reserved switch. KV is still part of the selected attention implementation.
            pass
        self.config = config
        self.components = components
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rotary_emb = RotaryEmbedding(config.qk_rope_head_dim)
        self.layers = nn.ModuleList(TorchForgeDecoderLayer(config, components) for _ in range(config.num_hidden_layers))
        self.norm = _build_norm(config, components.norm)
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


def build_model(config: DSV3ReferenceConfig, components: ComponentConfig) -> nn.Module:
    if components.to_dict() == {"attention": "pytorch", "norm": "pytorch", "ffn": "pytorch", "kv": "pytorch"}:
        return DSV3ReferenceModel(config)
    return DSV3TorchForgeModel(config, components)


def _build_norm(config: DSV3ReferenceConfig, implementation: str) -> nn.Module:
    if implementation == "pytorch":
        return ReferenceRMSNorm(config.hidden_size, config.rms_norm_eps)
    if implementation == "torchforge":
        return RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    raise ValueError(f"Unsupported norm implementation: {implementation!r}.")


def _build_attention(config: DSV3ReferenceConfig, implementation: str) -> nn.Module:
    if implementation == "pytorch":
        return ReferenceMLA(config)
    if implementation == "torchforge":
        return TorchForgeMLAWrapper(config)
    raise ValueError(f"Unsupported attention implementation: {implementation!r}.")


def _build_ffn(config: DSV3ReferenceConfig, implementation: str) -> nn.Module:
    if implementation == "pytorch":
        if config.ffn_type == "moe":
            return ReferenceMoE(config)
        return ReferenceDenseFFN(config)
    if implementation == "torchforge":
        return FeedForward(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            activation="swiglu",
            dropout=config.hidden_dropout,
            bias=False,
        )
    if implementation == "moe":
        return TorchForgeMoEWrapper(config)
    raise ValueError(f"Unsupported ffn implementation: {implementation!r}.")


__all__ = [
    "ComponentConfig",
    "DSV3TorchForgeModel",
    "TorchForgeDecoderLayer",
    "build_model",
    "variant_name",
]

