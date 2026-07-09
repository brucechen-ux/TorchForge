from __future__ import annotations

import argparse
from typing import Any, Optional

import torch
from torch import nn

from torchforge.common.attention import MLA
from torchforge.common.embedding import Embedding, RotaryEmbedding
from torchforge.common.attention import CSACompressor, HCACompressor
from torchforge.common.lm_head import LMHead
from torchforge.common.loss import CausalLMLoss
from torchforge.common.mask import SlidingWindowCausalMask
from torchforge.common.moe import HashRouter, MoE, SharedExpertMLP
from torchforge.common.nn import RMSNorm
from torchforge.common.optim import AdamW, build_param_groups
from torchforge.common.position import PositionIds
from torchforge.common.residual import ManifoldConstrainedHyperConnection
from torchforge.common.train import TrainStep, random_token_batches


class _HashRouterAdapter(nn.Module):
    """Experiment-only adapter from token-id hash routing to MoE's router slot."""

    def __init__(self, router: HashRouter) -> None:
        super().__init__()
        self.router = router
        self.num_experts = router.num_experts
        self.top_k = router.top_k
        self._input_ids: Optional[torch.Tensor] = None

    def set_input_ids(self, input_ids: torch.Tensor) -> None:
        self._input_ids = input_ids

    def forward(self, hidden_states: torch.Tensor, *, return_dict: bool = True) -> Any:
        if self._input_ids is None:
            raise RuntimeError("HashRouterAdapter requires set_input_ids before MoE forward.")
        return self.router(self._input_ids.reshape(-1), dtype=hidden_states.dtype, return_dict=return_dict)


class _HCAAdapter(nn.Module):
    def __init__(self, compressor: HCACompressor) -> None:
        super().__init__()
        self.compressor = compressor

    def forward(self, hidden_states: torch.Tensor, q_residual: torch.Tensor, position_ids: torch.Tensor) -> Any:
        return self.compressor(hidden_states, position_ids=position_ids)


class _CSAAdapter(nn.Module):
    def __init__(self, compressor: CSACompressor) -> None:
        super().__init__()
        self.compressor = compressor

    def forward(self, hidden_states: torch.Tensor, q_residual: torch.Tensor, position_ids: torch.Tensor) -> Any:
        return self.compressor(hidden_states, q_residual=q_residual, position_ids=position_ids)


def tiny_deepseek_v4_config(*, variant: str = "flash") -> dict[str, Any]:
    return {
        "variant": variant,
        "vocab_size": 128,
        "hidden_size": 32,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "rope_head_dim": 4,
        "q_lora_rank": 8,
        "o_groups": 2,
        "o_lora_rank": 8,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "num_shared_experts": 1,
        "expert_intermediate_size": 16,
        "hash_routing_layers": 3,
        "window_size": 4,
        "csa_compress_rate": 4,
        "hca_compress_rate": 4,
        "index_num_heads": 2,
        "index_head_dim": 8,
        "index_top_k": 2,
        "rms_norm_eps": 1.0e-6,
        "rope_theta": 10000.0,
        "max_position_embeddings": 4096,
        "attention_dropout": 0.0,
        "mha_expansion_factor": 4,
        "sinkhorn_iters": 20,
        "tie_word_embeddings": False,
    }


def paper_scale_deepseek_v4_config(*, variant: str) -> dict[str, Any]:
    if variant == "flash":
        return {
            **tiny_deepseek_v4_config(variant="flash"),
            "vocab_size": 128000,
            "hidden_size": 4096,
            "num_hidden_layers": 43,
            "num_attention_heads": 64,
            "head_dim": 512,
            "rope_head_dim": 64,
            "num_experts": 256,
        }
    if variant == "pro":
        return {
            **tiny_deepseek_v4_config(variant="pro"),
            "vocab_size": 128000,
            "hidden_size": 7168,
            "num_hidden_layers": 61,
            "num_attention_heads": 128,
            "head_dim": 512,
            "rope_head_dim": 64,
            "num_experts": 384,
        }
    raise ValueError("variant must be either 'flash' or 'pro'.")


def attention_kind_for_layer(config: dict[str, Any], layer_idx: int) -> str:
    if config["variant"] == "flash" and layer_idx < 2:
        return "sliding"
    if config["variant"] == "pro" and layer_idx < 2:
        return "hca"
    return "csa" if layer_idx % 2 == 0 else "hca"


def build_deepseek_v4_components(config: dict[str, Any]) -> nn.ModuleDict:
    components = nn.ModuleDict(
        {
            "embed_tokens": Embedding(vocab_size=config["vocab_size"], hidden_size=config["hidden_size"]),
            "position_ids": PositionIds(),
            "rotary_emb": RotaryEmbedding(
                head_dim=config["head_dim"],
                rope_theta=config["rope_theta"],
                partial_rotary_factor=config["rope_head_dim"] / config["head_dim"],
                max_position_embeddings=config["max_position_embeddings"],
            ),
            "sliding_mask": SlidingWindowCausalMask(window_size=config["window_size"]),
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
    attention_kind = attention_kind_for_layer(config, layer_idx)
    compressor = build_kv_compressor(config, attention_kind)
    layer = nn.ModuleDict(
        {
            "attention_mhc": build_mhc(config),
            "input_norm": RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"]),
            "self_attn": build_v4_mla(config, compressor=compressor),
            "post_attention_mhc": build_mhc(config),
            "post_attention_norm": RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"]),
            "ffn": build_deepseek_moe(config, layer_idx),
        }
    )
    return layer


def build_mhc(config: dict[str, Any]) -> ManifoldConstrainedHyperConnection:
    return ManifoldConstrainedHyperConnection(
        hidden_size=config["hidden_size"],
        expansion_factor=config["mha_expansion_factor"],
        sinkhorn_iters=config["sinkhorn_iters"],
        dynamic=True,
    )


def build_v4_mla(config: dict[str, Any], *, compressor: Optional[nn.Module]) -> MLA:
    return MLA(
        hidden_size=config["hidden_size"],
        num_attention_heads=config["num_attention_heads"],
        num_key_value_heads=config["num_key_value_heads"],
        q_lora_rank=config["q_lora_rank"],
        kv_lora_rank=None,
        qk_nope_head_dim=config["head_dim"] - config["rope_head_dim"],
        qk_rope_head_dim=config["rope_head_dim"],
        v_head_dim=config["head_dim"],
        kv_compressor=compressor,
        attention_dropout=config["attention_dropout"],
        rms_norm_eps=config["rms_norm_eps"],
        query_projection_type="low_rank",
        query_pre_norm="rmsnorm",
        query_post_norm="unweighted_rmsnorm",
        query_store_residual=True,
        kv_projection_type="direct_kv",
        kv_final_norm="rmsnorm",
        kv_value_mode="shared_with_key",
        rotary_layout="interleaved",
        rotary_application="partial_trailing",
        position_source="tuple",
        store_latest_position=True,
        kv_value_policy="value_equals_key_after_position",
        attention_bias_policy="append_block_bias",
        pad_attention_bias_to_kv_length=True,
        repeat_kv=True,
        attention_sinks=True,
        attention_scaling="v_head_dim",
        output_projection_type="grouped_low_rank",
        pre_output_transform="inverse_rope",
        o_groups=config["o_groups"],
        o_lora_rank=config["o_lora_rank"],
    )


def build_kv_compressor(config: dict[str, Any], attention_kind: str) -> Optional[nn.Module]:
    rotary_factor = config["rope_head_dim"] / config["head_dim"]
    if attention_kind == "hca":
        return _HCAAdapter(
            HCACompressor(
                hidden_size=config["hidden_size"],
                head_dim=config["head_dim"],
                compress_rate=config["hca_compress_rate"],
                partial_rotary_factor=rotary_factor,
                rope_theta=config["rope_theta"],
                rms_norm_eps=config["rms_norm_eps"],
            )
        )
    if attention_kind == "csa":
        return _CSAAdapter(
            CSACompressor(
                hidden_size=config["hidden_size"],
                q_lora_rank=config["q_lora_rank"],
                head_dim=config["head_dim"],
                index_num_heads=config["index_num_heads"],
                index_head_dim=config["index_head_dim"],
                index_top_k=config["index_top_k"],
                compress_rate=config["csa_compress_rate"],
                partial_rotary_factor=rotary_factor,
                rope_theta=config["rope_theta"],
                rms_norm_eps=config["rms_norm_eps"],
            )
        )
    return None


def build_deepseek_moe(config: dict[str, Any], layer_idx: int) -> MoE:
    shared_expert = SharedExpertMLP(
        hidden_size=config["hidden_size"],
        intermediate_size=config["num_shared_experts"] * config["expert_intermediate_size"],
        activation="silu",
        gated=True,
        bias=False,
    )
    router = None
    if layer_idx < config["hash_routing_layers"]:
        router = _HashRouterAdapter(
            HashRouter(num_experts=config["num_experts"], top_k=config["num_experts_per_tok"], seed=layer_idx)
        )
    return MoE(
        hidden_size=config["hidden_size"],
        router=router,
        num_experts=config["num_experts"],
        top_k=config["num_experts_per_tok"],
        expert_intermediate_size=config["expert_intermediate_size"],
        shared_expert=shared_expert,
        router_score_function="sigmoid",
        normalize_topk=True,
        expert_activation="silu",
        expert_gated=True,
        bias=False,
    )


def forward_deepseek_v4_components(components: nn.ModuleDict, input_ids: torch.Tensor) -> torch.Tensor:
    position_ids = components["position_ids"](input_ids)
    position_embeddings = components["rotary_emb"](position_ids)
    hidden_states = components["embed_tokens"](input_ids)
    attention_mask = components["sliding_mask"](input_ids, dtype=hidden_states.dtype)
    residual_state = components["layers"][0]["attention_mhc"].init_state(hidden_states)
    for layer in components["layers"]:
        residual_state, hidden_states = forward_decoder_layer_components(
            layer,
            residual_state,
            input_ids=input_ids,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )
    return components["lm_head"](components["final_norm"](hidden_states))


def forward_decoder_layer_components(
    layer: nn.ModuleDict,
    residual_state: torch.Tensor,
    *,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    attention_input = layer["attention_mhc"].read(residual_state)
    attention_output = layer["self_attn"](
        layer["input_norm"](attention_input),
        attention_mask=attention_mask,
        position_ids=position_ids,
        position_embeddings=position_embeddings,
    )["hidden_states"]
    residual_state, hidden_states = layer["attention_mhc"](residual_state, attention_output, return_dict=False)
    ffn_input = layer["post_attention_mhc"].read(residual_state)
    prime_hash_router(layer["ffn"], input_ids)
    ffn_output = layer["ffn"](layer["post_attention_norm"](ffn_input))
    if isinstance(ffn_output, dict):
        ffn_output = ffn_output["hidden_states"]
    return layer["post_attention_mhc"](residual_state, ffn_output, return_dict=False)


def prime_hash_router(module: nn.Module, input_ids: torch.Tensor) -> None:
    router = getattr(module, "router", None)
    if isinstance(router, _HashRouterAdapter):
        router.set_input_ids(input_ids)


def describe_layout(config: dict[str, Any]) -> list[str]:
    lines = [f"variant: DeepSeek-V4-{config['variant']}"]
    lines.append(f"Embedding -> DecoderLayer x {config['num_hidden_layers']} -> Final RMSNorm -> LMHead")
    for layer_idx in range(config["num_hidden_layers"]):
        router = "hash" if layer_idx < config["hash_routing_layers"] else "learned-topk"
        lines.append(f"layer {layer_idx}: attention={attention_kind_for_layer(config, layer_idx)}, residual=mHC, moe_router={router}")
    return lines


def train_deepseek_v4_components(
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
        forward_fn=lambda input_ids: forward_deepseek_v4_components(components, input_ids),
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
    parser = argparse.ArgumentParser(description="Assemble DeepSeek-V4 from torchforge.common components.")
    parser.add_argument("--variant", choices=("flash", "pro"), default="flash")
    parser.add_argument("--paper-scale", action="store_true", help="Print the paper-scale component layout.")
    parser.add_argument("--train", action="store_true", help="Run a minimal training loop on random data.")
    parser.add_argument("--steps", type=int, default=20, help="Number of training steps when --train is set.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate when --train is set.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-length", type=int, default=8)
    args = parser.parse_args()
    if args.paper_scale:
        print("\n".join(describe_layout(paper_scale_deepseek_v4_config(variant=args.variant))))
        return
    config = tiny_deepseek_v4_config(variant=args.variant)
    components = build_deepseek_v4_components(config)
    print("\n".join(describe_layout(config)))
    if args.train:
        train_deepseek_v4_components(
            components,
            config,
            batch_size=args.batch_size,
            seq_length=args.seq_length,
            num_steps=args.steps,
            lr=args.lr,
        )
        return
    input_ids = torch.randint(0, config["vocab_size"], (args.batch_size, args.seq_length))
    logits = forward_deepseek_v4_components(components, input_ids)
    print("logits shape:", tuple(logits.shape))


if __name__ == "__main__":
    main()
