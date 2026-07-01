from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class DSV3ReferenceConfig:
    """Small single-card DeepSeek-V3-style decoder configuration."""

    vocab_size: int = 128
    hidden_size: int = 32
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    num_key_value_heads: int = 4
    q_lora_rank: int = 8
    kv_lora_rank: int = 8
    qk_nope_head_dim: int = 4
    qk_rope_head_dim: int = 4
    v_head_dim: int = 8
    intermediate_size: int = 64
    rms_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    attention_bias: bool = False
    ffn_type: str = "dense"
    num_experts: int = 4
    top_k: int = 2
    tie_word_embeddings: bool = False

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_config(config: DSV3ReferenceConfig) -> None:
    positive_int_fields = (
        "vocab_size",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "q_lora_rank",
        "kv_lora_rank",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "v_head_dim",
        "intermediate_size",
    )
    for name in positive_int_fields:
        value = getattr(config, name)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive int, got {value!r}.")
    if config.num_attention_heads % config.num_key_value_heads != 0:
        raise ValueError("num_attention_heads must be divisible by num_key_value_heads.")
    if config.ffn_type not in {"dense", "moe"}:
        raise ValueError("ffn_type must be either 'dense' or 'moe'.")
    if config.num_experts <= 0:
        raise ValueError("num_experts must be positive.")
    if config.top_k <= 0 or config.top_k > config.num_experts:
        raise ValueError("top_k must be in the range [1, num_experts].")
    if config.rms_norm_eps <= 0.0:
        raise ValueError("rms_norm_eps must be positive.")
    for name in ("attention_dropout", "hidden_dropout"):
        value = getattr(config, name)
        if value < 0.0 or value >= 1.0:
            raise ValueError(f"{name} must be in [0, 1).")

