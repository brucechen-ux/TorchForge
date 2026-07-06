from __future__ import annotations

import argparse
from typing import Any

import torch
from torch import nn

from torchforge.common.attention import MLA
from torchforge.common.embedding import Embedding, RotaryEmbedding
from torchforge.common.lm_head import LMHead
from torchforge.common.loss import CausalLMLoss
from torchforge.common.mask import CausalMask
from torchforge.common.moe import MoE, SharedExpertMLP
from torchforge.common.nn import FeedForward, RMSNorm
from torchforge.common.optim import AdamW, build_param_groups
from torchforge.common.position import PositionIds
from torchforge.common.residual import ResidualAdd
from torchforge.common.train import TrainStep, random_token_batches


def tiny_deepseek_v3_config() -> dict[str, Any]:
    return {
        "vocab_size": 128,
        "hidden_size": 32,
        "num_hidden_layers": 4,
        "first_k_dense_replace": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "dense_intermediate_size": 64,
        "moe_intermediate_size": 16,
        "n_routed_experts": 4,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "q_lora_rank": 8,
        "kv_lora_rank": 8,
        "qk_nope_head_dim": 4,
        "qk_rope_head_dim": 4,
        "v_head_dim": 8,
        "rms_norm_eps": 1.0e-6,
        "rope_theta": 10000.0,
        "max_position_embeddings": 4096,
        "attention_dropout": 0.0,
        "hidden_dropout": 0.0,
        "attention_bias": False,
        "router_score_function": "sigmoid",
        "normalize_topk": True,
        "routed_scaling_factor": 2.5,
        "tie_word_embeddings": False,
    }


def paper_scale_deepseek_v3_config() -> dict[str, Any]:
    return {
        **tiny_deepseek_v3_config(),
        "vocab_size": 129280,
        "hidden_size": 7168,
        "num_hidden_layers": 61,
        "first_k_dense_replace": 3,
        "num_attention_heads": 128,
        "num_key_value_heads": 128,
        "dense_intermediate_size": 18432,
        "moe_intermediate_size": 2048,
        "n_routed_experts": 256,
        "num_experts_per_tok": 8,
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
    }


def build_deepseek_v3_components(config: dict[str, Any]) -> nn.ModuleDict:
    validate_config(config)
    components = nn.ModuleDict(
        {
            "embed_tokens": Embedding(vocab_size=config["vocab_size"], hidden_size=config["hidden_size"]),
            "position_ids": PositionIds(),
            "rotary_emb": RotaryEmbedding(
                head_dim=config["qk_rope_head_dim"],
                rope_theta=config["rope_theta"],
                partial_rotary_factor=1.0,
                max_position_embeddings=config["max_position_embeddings"],
            ),
            "causal_mask": CausalMask(),
            "layers": nn.ModuleList(
                build_decoder_layer_components(config, layer_idx)
                for layer_idx in range(config["num_hidden_layers"])
            ),
            "final_norm": RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"]),
            "lm_head": LMHead(hidden_size=config["hidden_size"], vocab_size=config["vocab_size"], bias=False),
        }
    )
    if config["tie_word_embeddings"]:
        components["lm_head"].tie_weights(components["embed_tokens"])
    return components


def build_decoder_layer_components(config: dict[str, Any], layer_idx: int) -> nn.ModuleDict:
    return nn.ModuleDict(
        {
            "input_norm": RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"]),
            "self_attn": build_mla(config),
            "attention_residual": ResidualAdd(),
            "post_attention_norm": RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"]),
            "ffn": build_dense_ffn(config) if layer_idx < config["first_k_dense_replace"] else build_deepseek_moe(config),
            "ffn_residual": ResidualAdd(),
        }
    )


def build_mla(config: dict[str, Any]) -> MLA:
    return MLA(
        hidden_size=config["hidden_size"],
        num_attention_heads=config["num_attention_heads"],
        num_key_value_heads=config["num_key_value_heads"],
        q_lora_rank=config["q_lora_rank"],
        kv_lora_rank=config["kv_lora_rank"],
        qk_nope_head_dim=config["qk_nope_head_dim"],
        qk_rope_head_dim=config["qk_rope_head_dim"],
        v_head_dim=config["v_head_dim"],
        attention_dropout=config["attention_dropout"],
        attention_bias=config["attention_bias"],
        rms_norm_eps=config["rms_norm_eps"],
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


def build_dense_ffn(config: dict[str, Any]) -> FeedForward:
    return FeedForward(
        hidden_size=config["hidden_size"],
        intermediate_size=config["dense_intermediate_size"],
        activation="swiglu",
        dropout=config["hidden_dropout"],
        bias=False,
    )


def build_deepseek_moe(config: dict[str, Any]) -> MoE:
    shared_expert = SharedExpertMLP(
        hidden_size=config["hidden_size"],
        intermediate_size=config["n_shared_experts"] * config["moe_intermediate_size"],
        activation="silu",
        gated=True,
        bias=False,
    )
    return MoE(
        hidden_size=config["hidden_size"],
        num_experts=config["n_routed_experts"],
        top_k=config["num_experts_per_tok"],
        expert_intermediate_size=config["moe_intermediate_size"],
        shared_expert=shared_expert,
        router_score_function=config["router_score_function"],
        normalize_topk=config["normalize_topk"],
        route_scale=config["routed_scaling_factor"],
        expert_activation="silu",
        expert_gated=True,
        bias=False,
    )


def forward_deepseek_v3_components(components: nn.ModuleDict, input_ids: torch.Tensor) -> torch.Tensor:
    position_ids = components["position_ids"](input_ids)
    position_embeddings = components["rotary_emb"](position_ids)
    hidden_states = components["embed_tokens"](input_ids)
    attention_mask = components["causal_mask"](input_ids, dtype=hidden_states.dtype)
    for layer in components["layers"]:
        hidden_states = forward_decoder_layer_components(layer, hidden_states, position_embeddings, attention_mask)
    return components["lm_head"](components["final_norm"](hidden_states))


def forward_decoder_layer_components(
    layer: nn.ModuleDict,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    attention_output = layer["self_attn"](
        layer["input_norm"](hidden_states),
        attention_mask=attention_mask,
        position_embeddings=position_embeddings,
    )["hidden_states"]
    hidden_states = layer["attention_residual"](hidden_states, attention_output)
    ffn_output = layer["ffn"](layer["post_attention_norm"](hidden_states))
    if isinstance(ffn_output, dict):
        ffn_output = ffn_output["hidden_states"]
    return layer["ffn_residual"](hidden_states, ffn_output)


def validate_config(config: dict[str, Any]) -> None:
    if config["first_k_dense_replace"] > config["num_hidden_layers"]:
        raise ValueError("first_k_dense_replace must be <= num_hidden_layers.")
    if config["num_experts_per_tok"] > config["n_routed_experts"]:
        raise ValueError("num_experts_per_tok must be <= n_routed_experts.")


def train_deepseek_v3_components(
    components: nn.ModuleDict,
    config: dict[str, Any],
    *,
    batch_size: int,
    seq_length: int,
    num_steps: int,
    lr: float,
    seed: int = 0,
) -> None:
    """Run a minimal training loop assembled from torchforge.common components."""

    generator = torch.Generator().manual_seed(seed)
    loss_module = CausalLMLoss()
    optimizer = AdamW(build_param_groups(components, weight_decay=0.1), lr=lr)
    step = TrainStep(
        forward_fn=lambda input_ids: forward_deepseek_v3_components(components, input_ids),
        loss_module=loss_module,
        optimizer=optimizer,
    )
    components.train()
    for i, (input_ids, labels) in enumerate(
        random_token_batches(
            vocab_size=config["vocab_size"],
            batch_size=batch_size,
            seq_length=seq_length,
            num_steps=num_steps,
            generator=generator,
        )
    ):
        metrics = step.run(input_ids, labels)
        print(f"step {i:03d} | loss {metrics['loss']:.4f} | grad_norm {metrics['grad_norm']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble DeepSeek-V3 from torchforge.common components.")
    parser.add_argument("--paper-scale", action="store_true", help="Print the paper-scale component layout.")
    parser.add_argument("--train", action="store_true", help="Run a minimal training loop on random data.")
    parser.add_argument("--steps", type=int, default=20, help="Number of training steps when --train is set.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate when --train is set.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-length", type=int, default=8)
    args = parser.parse_args()
    if args.paper_scale:
        config = paper_scale_deepseek_v3_config()
        print("Embedding -> DecoderLayer x", config["num_hidden_layers"], "-> Final RMSNorm -> LMHead")
        print("dense decoder layers:", config["first_k_dense_replace"])
        print("moe decoder layers:", config["num_hidden_layers"] - config["first_k_dense_replace"])
        return
    config = tiny_deepseek_v3_config()
    components = build_deepseek_v3_components(config)
    if args.train:
        print("Embedding -> DecoderLayer x", len(components["layers"]), "-> Final RMSNorm -> LMHead")
        train_deepseek_v3_components(
            components,
            config,
            batch_size=args.batch_size,
            seq_length=args.seq_length,
            num_steps=args.steps,
            lr=args.lr,
        )
        return
    input_ids = torch.randint(0, config["vocab_size"], (args.batch_size, args.seq_length))
    logits = forward_deepseek_v3_components(components, input_ids)
    print("Embedding -> DecoderLayer x", len(components["layers"]), "-> Final RMSNorm -> LMHead")
    print("logits shape:", tuple(logits.shape))


if __name__ == "__main__":
    main()
