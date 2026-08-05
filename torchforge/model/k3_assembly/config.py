from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


def small_k3_config() -> dict[str, Any]:
    return {
        "model": {
            "name": "torchforge_k3_small_1b",
            "vocab_size": 49152,
            "hidden_size": 768,
            "num_attention_heads": 12,
            "head_dim": 64,
            "num_hybrid_blocks": 6,
            "attnres_block_size": 8,
            "dense_intermediate_size": 2048,
            "rms_norm_eps": 1.0e-6,
            "tie_word_embeddings": False,
        },
        "kda": {
            "backend": "fla",
            "short_conv_kernel_size": 4,
            "decay_rank": 64,
            "g_min": -5.0,
            "chunk_size": 64,
            "tile_size": 16,
        },
        "mla": {
            "q_lora_rank": 192,
            "kv_lora_rank": 64,
            "attention_backend": "sdpa",
            "attention_dropout": 0.0,
        },
        "moe": {
            "latent_size": 384,
            "num_routed_experts": 224,
            "num_experts_per_token": 4,
            "num_shared_experts": 2,
            "expert_intermediate_size": 128,
            "shared_intermediate_size": 384,
            "histogram_bins": 256,
            "beta_gate": 4.0,
            "beta_up": 25.0,
        },
        "mtp": {"enabled": True, "depth": 1, "loss_weight": 0.1},
        "train": {
            "max_steps": 10000,
            "seq_len": 8192,
            "validation_seq_len": 32768,
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "learning_rate": 2.0e-4,
            "min_lr": 2.0e-5,
            "weight_decay": 0.1,
            "warmup_steps": 100,
            "gradient_clipping": 1.0,
            "bf16": True,
            "activation_checkpointing": True,
            "log_steps": 10,
            "save_steps": 500,
            "valid_steps": 500,
            "output_dir": "torchforge/model/k3_assembly/outputs/k3_small_1b",
            "optimizer": {
                "momentum": 0.95,
                "nesterov": True,
                "newton_schulz": "hybrid",
                "newton_schulz_iterations": 10,
                "update_rms_target": 0.18,
                "betas": [0.9, 0.95],
                "eps": 1.0e-20,
            },
        },
        "data": {
            "type": "memmap",
            "data_dir": "data/openbmb_UltraFineWeb_5b_random_tokens",
            "train_file": "train.bin",
            "valid_file": "valid.bin",
            "manifest_file": "manifest.json",
            "dtype": "uint32",
            "vocab_size": 49152,
            "num_workers": 2,
            "pin_memory": True,
            "persistent_workers": True,
            "prefetch_factor": 2,
        },
        "seed": 2026,
    }


def tiny_k3_config() -> dict[str, Any]:
    config = copy.deepcopy(small_k3_config())
    config["model"].update(
        name="torchforge_k3_tiny",
        vocab_size=64,
        hidden_size=32,
        num_attention_heads=4,
        head_dim=8,
        num_hybrid_blocks=1,
        attnres_block_size=2,
        dense_intermediate_size=48,
    )
    config["kda"].update(
        backend="reference",
        short_conv_kernel_size=2,
        decay_rank=8,
        chunk_size=4,
        tile_size=2,
    )
    config["mla"].update(q_lora_rank=16, kv_lora_rank=8, attention_backend="reference")
    config["moe"].update(
        latent_size=16,
        num_routed_experts=4,
        num_experts_per_token=2,
        expert_intermediate_size=16,
        shared_intermediate_size=16,
        histogram_bins=16,
    )
    config["train"].update(
        max_steps=2,
        seq_len=8,
        validation_seq_len=16,
        micro_batch_size=2,
        activation_checkpointing=False,
        warmup_steps=1,
    )
    config["data"].update(vocab_size=64, num_workers=0, pin_memory=False, persistent_workers=False)
    validate_config(config)
    return config


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    model, kda, mla, moe, mtp, train, data = (
        config["model"],
        config["kda"],
        config["mla"],
        config["moe"],
        config["mtp"],
        config["train"],
        config["data"],
    )
    positive = {
        "model.vocab_size": model["vocab_size"],
        "model.hidden_size": model["hidden_size"],
        "model.num_attention_heads": model["num_attention_heads"],
        "model.head_dim": model["head_dim"],
        "model.num_hybrid_blocks": model["num_hybrid_blocks"],
        "model.attnres_block_size": model["attnres_block_size"],
        "model.dense_intermediate_size": model["dense_intermediate_size"],
        "kda.short_conv_kernel_size": kda["short_conv_kernel_size"],
        "kda.decay_rank": kda["decay_rank"],
        "kda.chunk_size": kda["chunk_size"],
        "kda.tile_size": kda["tile_size"],
        "mla.q_lora_rank": mla["q_lora_rank"],
        "mla.kv_lora_rank": mla["kv_lora_rank"],
        "moe.latent_size": moe["latent_size"],
        "moe.num_routed_experts": moe["num_routed_experts"],
        "moe.num_experts_per_token": moe["num_experts_per_token"],
        "moe.num_shared_experts": moe["num_shared_experts"],
        "moe.expert_intermediate_size": moe["expert_intermediate_size"],
        "moe.shared_intermediate_size": moe["shared_intermediate_size"],
        "moe.histogram_bins": moe["histogram_bins"],
        "train.max_steps": train["max_steps"],
        "train.seq_len": train["seq_len"],
        "train.validation_seq_len": train["validation_seq_len"],
    }
    for name, value in positive.items():
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive int, got {value!r}.")
    if model["hidden_size"] != model["num_attention_heads"] * model["head_dim"]:
        raise ValueError("hidden_size must equal num_attention_heads * head_dim.")
    if moe["latent_size"] * 2 != model["hidden_size"]:
        raise ValueError("The small-K3 config keeps latent_size at half hidden_size.")
    if moe["num_experts_per_token"] >= moe["num_routed_experts"]:
        raise ValueError("num_experts_per_token must be less than num_routed_experts.")
    if kda["backend"] not in {"reference", "fla"}:
        raise ValueError("kda.backend must be 'reference' or 'fla'.")
    if float(kda["g_min"]) >= 0.0:
        raise ValueError("kda.g_min must be negative.")
    if kda["chunk_size"] % kda["tile_size"] != 0:
        raise ValueError("kda.chunk_size must be divisible by kda.tile_size.")
    if kda["backend"] == "fla" and (
        kda["chunk_size"] not in {32, 64}
        or kda["tile_size"] != 16
        or not -5.0 <= float(kda["g_min"]) < 0.0
    ):
        raise ValueError(
            "The FLA backend requires chunk_size 32/64, tile_size 16, "
            "and g_min in [-5, 0)."
        )
    if mla["attention_backend"] not in {"reference", "sdpa"}:
        raise ValueError("mla.attention_backend must be 'reference' or 'sdpa'.")
    if int(moe["num_shared_experts"]) != 2:
        raise ValueError("Small K3 requires exactly two shared experts.")
    if int(moe["histogram_bins"]) <= 1:
        raise ValueError("moe.histogram_bins must be greater than one.")
    if float(moe["beta_gate"]) <= 0.0 or float(moe["beta_up"]) <= 0.0:
        raise ValueError("moe.beta_gate and moe.beta_up must be positive.")
    if not bool(mtp["enabled"]) or int(mtp["depth"]) != 1:
        raise ValueError("Small K3 requires exactly one enabled MTP layer.")
    if int(data["vocab_size"]) != int(model["vocab_size"]):
        raise ValueError("data.vocab_size must match model.vocab_size.")
    if data["type"] != "memmap" or data["dtype"] != "uint32":
        raise ValueError("Small K3 expects the existing uint32 memmap format.")


__all__ = ["load_config", "small_k3_config", "tiny_k3_config", "validate_config"]
