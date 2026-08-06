from __future__ import annotations

import argparse
from typing import Any, Optional

import torch
from torch import nn

from torchforge.common.attention import CausalMask, MLA
from torchforge.common.embedding import Embedding, RotaryEmbedding
from torchforge.common.lm_head import LMHead
from torchforge.common.loss import CausalLMLoss
from torchforge.common.mlp import FeedForward
from torchforge.common.moe import MoE, SharedExpertMLP
from torchforge.common.mtp import MultiTokenPredictionModule
from torchforge.common.nn import RMSNorm
from torchforge.common.optim import AdamW, build_param_groups
from torchforge.common.position import PositionIds
from torchforge.common.residual import ResidualAdd
from torchforge.common.train import random_token_batches


class _DecoderLayerBlockAdapter(nn.Module):
    """Adapt an assembled decoder layer to the shared MTP block interface."""

    def __init__(self, layer: nn.ModuleDict) -> None:
        super().__init__()
        self.layer = layer
        self.last_aux_loss: Optional[torch.Tensor] = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
        update_router_bias: bool = False,
        return_dict: bool = True,
        **_: Any,
    ) -> Any:
        output, aux_loss = forward_decoder_layer_components(
            self.layer,
            hidden_states,
            position_embeddings,
            attention_mask,
            output_aux_loss=True,
            update_router_bias=update_router_bias,
        )
        self.last_aux_loss = aux_loss
        if return_dict:
            result = {"hidden_states": output}
            if aux_loss is not None:
                result["aux_loss"] = aux_loss
            return result
        return output


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
        "mtp_depth": 1,
        "mtp_loss_weight": 0.3,
        "moe_aux_loss_alpha": 0.0001,
        "router_score_correction_bias": True,
        "router_bias_update_rate": 1.0e-3,
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
    if config.get("mtp_depth", 0) != 1:
        raise ValueError("This single-card DeepSeek-V3 assembly supports mtp_depth=1.")
    components["mtp"] = MultiTokenPredictionModule(
        hidden_size=config["hidden_size"],
        embedding=components["embed_tokens"],
        transformer_block=_DecoderLayerBlockAdapter(
            build_decoder_layer_components(config, config["num_hidden_layers"])
        ),
        lm_head=components["lm_head"],
        bias=False,
        rms_norm_eps=config["rms_norm_eps"],
    )
    components._dsv3_config = dict(config)
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
        router_score_correction_bias=bool(config.get("router_score_correction_bias", False)),
        router_bias_update_rate=config.get("router_bias_update_rate", 1.0e-3),
        return_aux_loss=config.get("moe_aux_loss_alpha", 0.0) > 0.0,
        aux_loss_alpha=config.get("moe_aux_loss_alpha", 0.0),
        expert_activation="silu",
        expert_gated=True,
        bias=False,
    )


def forward_deepseek_v3_components(
    components: nn.ModuleDict,
    input_ids: torch.Tensor,
    *,
    labels: Optional[torch.Tensor] = None,
    return_dict: bool = False,
    update_router_bias: bool = False,
) -> Any:
    position_ids = components["position_ids"](input_ids)
    position_embeddings = components["rotary_emb"](position_ids)
    hidden_states = components["embed_tokens"](input_ids)
    attention_mask = components["causal_mask"](input_ids, dtype=hidden_states.dtype)
    moe_aux_loss = hidden_states.new_zeros(())
    for layer in components["layers"]:
        hidden_states, layer_aux_loss = forward_decoder_layer_components(
            layer,
            hidden_states,
            position_embeddings,
            attention_mask,
            output_aux_loss=True,
            update_router_bias=update_router_bias,
        )
        if layer_aux_loss is not None:
            moe_aux_loss = moe_aux_loss + layer_aux_loss
    final_hidden_states = components["final_norm"](hidden_states)
    logits = components["lm_head"](final_hidden_states)

    mtp_outputs = None
    mtp_aux_loss = hidden_states.new_zeros(())
    if "mtp" in components and (return_dict or labels is not None):
        mtp_outputs = components["mtp"](
            hidden_states,
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            update_router_bias=update_router_bias,
        )
        block_aux = getattr(components["mtp"].transformer_block, "last_aux_loss", None)
        if block_aux is not None:
            mtp_aux_loss = block_aux

    loss = lm_loss = mtp_loss = None
    if labels is not None:
        lm_loss = CausalLMLoss()(logits, labels)
        mtp_loss = (
            CausalLMLoss()(mtp_outputs["logits"], labels[:, 1:])
            if mtp_outputs is not None
            else logits.new_zeros(())
        )
        loss = (
            lm_loss
            + moe_aux_loss
            + mtp_aux_loss
            + mtp_loss * float(_config_value(components, "mtp_loss_weight", 0.3))
        )

    if not return_dict:
        return logits
    result = {
        "logits": logits,
        "hidden_states": final_hidden_states,
        "moe_aux_loss": moe_aux_loss + mtp_aux_loss,
    }
    if mtp_outputs is not None:
        result["mtp_logits"] = mtp_outputs["logits"]
    if loss is not None:
        result["loss"] = loss
        result["lm_loss"] = lm_loss
        result["mtp_loss"] = mtp_loss
    return result


def forward_decoder_layer_components(
    layer: nn.ModuleDict,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor,
    *,
    output_aux_loss: bool = False,
    update_router_bias: bool = False,
) -> Any:
    attention_output = layer["self_attn"](
        layer["input_norm"](hidden_states),
        attention_mask=attention_mask,
        position_embeddings=position_embeddings,
    )["hidden_states"]
    hidden_states = layer["attention_residual"](hidden_states, attention_output)
    ffn_input = layer["post_attention_norm"](hidden_states)
    if isinstance(layer["ffn"], MoE):
        ffn_output = layer["ffn"](
            ffn_input,
            output_aux_loss=output_aux_loss,
            update_router_bias=update_router_bias,
        )
    else:
        ffn_output = layer["ffn"](ffn_input)
    aux_loss = None
    if isinstance(ffn_output, dict):
        aux_loss = ffn_output.get("aux_loss")
        ffn_output = ffn_output["hidden_states"]
    hidden_states = layer["ffn_residual"](hidden_states, ffn_output)
    if output_aux_loss:
        return hidden_states, aux_loss
    return hidden_states


def _config_value(components: nn.ModuleDict, key: str, default: Any) -> Any:
    return getattr(components, "_dsv3_config", {}).get(key, default)


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

    if config.get("mtp_depth", 0) > 0 and seq_length < 3:
        raise ValueError("seq_length must be at least 3 when MTP loss is enabled.")
    generator = torch.Generator().manual_seed(seed)
    optimizer = AdamW(build_param_groups(components, weight_decay=0.1), lr=lr)
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
        optimizer.zero_grad()
        outputs = forward_deepseek_v3_components(
            components,
            input_ids,
            labels=labels,
            return_dict=True,
            update_router_bias=True,
        )
        outputs["loss"].backward()
        params = [
            param
            for group in optimizer.param_groups
            for param in group["params"]
            if param.grad is not None
        ]
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)) if params else 0.0
        optimizer.step()
        print(
            f"step {i:03d} | loss {float(outputs['loss'].detach()):.4f} "
            f"| lm {float(outputs['lm_loss'].detach()):.4f} "
            f"| mtp {float(outputs['mtp_loss'].detach()):.4f} "
            f"| aux {float(outputs['moe_aux_loss'].detach()):.4f} "
            f"| grad_norm {grad_norm:.4f}"
        )


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
