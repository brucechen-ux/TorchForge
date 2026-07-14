from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


def report_aligned_config() -> dict[str, Any]:
    """Return the approximately 397M report-aligned experiment configuration."""

    return {
        "model": {
            "name": "dsv4_muon_report_aligned_397m",
            "vocab_size": 49152,
            "seq_len": 4096,
            "num_layers": 16,
            "hidden_size": 704,
            "num_attention_heads": 11,
            "dense_intermediate_size": 1888,
            "first_dense_layers": 0,
            "rms_norm_eps": 1.0e-6,
            "tie_word_embeddings": False,
        },
        "v4_attention": {
            "attention_implementation": "sdpa",
            "q_lora_rank": 352,
            "head_dim": 64,
            "qk_rope_head_dim": 32,
            "rope_theta": 10000.0,
            "compress_rope_theta": 160000.0,
            "sliding_window": 128,
            "o_groups": 1,
            "o_lora_rank": 64,
            "index_n_heads": 11,
            "index_head_dim": 32,
            "index_topk": 512,
            "compress_rates": {
                "compressed_sparse_attention": 4,
                "heavily_compressed_attention": 128,
            },
        },
        "moe": {
            "num_routed_experts": 16,
            "num_shared_experts": 1,
            "num_experts_per_token": 1,
            "num_hash_layers": 3,
            "route_scale": 1.5,
            "score_function": "sqrtsoftplus",
            "swiglu_limit": 10.0,
            "aux_loss_weight": 0.0,
            "expert_intermediate_size": 512,
            "use_correction_bias": True,
            "use_packed_experts": True,
        },
        "mtp": {
            "enabled": True,
            "mtp_depth": 1,
            "mtp_loss_weight": 0.1,
            "mtp_use_moe": True,
        },
        "train": {
            "max_steps": 19073,
            "seq_len": 4096,
            "micro_batch_size": 4,
            "gradient_accumulation_steps": 2,
            "learning_rate": 3.0e-4,
            "min_lr": 1.5e-5,
            "weight_decay": 0.1,
            "warmup_steps": 1000,
            "gradient_clipping": 1.0,
            "log_steps": 10,
            "save_steps": 1000,
            "valid_steps": 500,
            "valid_max_batches": 128,
            "bf16": True,
            "ddp_find_unused_parameters": True,
            "target_tokens": 5000000000,
            "output_dir": "experiments/dsv4_muon_report_aligned/outputs/muon_hybrid",
            "optimizer": {
                "name": "muon",
                "momentum": 0.95,
                "nesterov": True,
                "betas": [0.9, 0.95],
                "eps": 1.0e-20,
                "newton_schulz": "hybrid",
                "newton_schulz_iterations": 10,
                "update_rms_target": 0.18,
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


def load_config(path: str | Path) -> dict[str, Any]:
    """Load JSON-compatible YAML without adding a YAML dependency."""

    with Path(path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    model = config["model"]
    moe = config["moe"]
    mtp = config["mtp"]
    train = config["train"]
    optimizer = train["optimizer"]
    required = {
        "model.num_layers": (int(model["num_layers"]), 16),
        "model.hidden_size": (int(model["hidden_size"]), 704),
        "model.num_attention_heads": (int(model["num_attention_heads"]), 11),
        "model.first_dense_layers": (int(model["first_dense_layers"]), 0),
        "moe.num_routed_experts": (int(moe["num_routed_experts"]), 16),
        "moe.num_experts_per_token": (int(moe["num_experts_per_token"]), 1),
        "moe.num_hash_layers": (int(moe["num_hash_layers"]), 3),
        "mtp.mtp_depth": (int(mtp["mtp_depth"]), 1),
    }
    for name, (actual, expected) in required.items():
        if actual != expected:
            raise ValueError(f"{name} must be {expected} for the report-aligned run, got {actual}.")
    if not bool(moe.get("use_packed_experts", False)):
        raise ValueError("moe.use_packed_experts must be true.")
    if str(moe.get("score_function")) != "sqrtsoftplus":
        raise ValueError("moe.score_function must be sqrtsoftplus.")
    if float(moe.get("route_scale", -1.0)) != 1.5:
        raise ValueError("moe.route_scale must be 1.5.")
    if float(moe.get("aux_loss_weight", -1.0)) != 0.0:
        raise ValueError("moe.aux_loss_weight must be 0.0.")
    if float(mtp.get("mtp_loss_weight", -1.0)) != 0.1:
        raise ValueError("mtp.mtp_loss_weight must be 0.1.")
    attention = config["v4_attention"]
    attention_required = {
        "v4_attention.q_lora_rank": (int(attention["q_lora_rank"]), 352),
        "v4_attention.head_dim": (int(attention["head_dim"]), 64),
        "v4_attention.qk_rope_head_dim": (int(attention["qk_rope_head_dim"]), 32),
    }
    for name, (actual, expected) in attention_required.items():
        if actual != expected:
            raise ValueError(f"{name} must be {expected}, got {actual}.")
    if str(attention.get("attention_implementation")) != "sdpa":
        raise ValueError("v4_attention.attention_implementation must be sdpa.")
    data = config["data"]
    if data.get("type") != "memmap" or data.get("dtype") != "uint32":
        raise ValueError("The aligned data path must be uint32 memmap.")
    if optimizer["name"] == "muon":
        if optimizer.get("newton_schulz") not in {"hybrid", "standard"}:
            raise ValueError("Muon newton_schulz must be 'hybrid' or 'standard'.")
        if int(optimizer.get("newton_schulz_iterations", 0)) != 10:
            raise ValueError("Muon requires exactly 10 Newton-Schulz iterations.")
        if float(optimizer.get("momentum", -1.0)) != 0.95:
            raise ValueError("Muon momentum must be 0.95.")
        if float(optimizer.get("update_rms_target", -1.0)) != 0.18:
            raise ValueError("Muon update_rms_target must be 0.18.")
        if float(optimizer.get("eps", -1.0)) != 1.0e-20:
            raise ValueError("Muon auxiliary AdamW eps must be 1e-20.")
    elif optimizer["name"] == "adamw" and float(optimizer.get("eps", -1.0)) != 1.0e-8:
        raise ValueError("AdamW baseline eps must be 1e-8.")


def tiny_parity_config() -> dict[str, Any]:
    """Small shape-compatible config used only by deterministic parity tests."""

    config = copy.deepcopy(report_aligned_config())
    config["model"].update(
        vocab_size=64,
        seq_len=8,
        num_layers=4,
        hidden_size=32,
        num_attention_heads=4,
        dense_intermediate_size=48,
    )
    config["v4_attention"].update(
        q_lora_rank=16,
        head_dim=8,
        qk_rope_head_dim=4,
        sliding_window=4,
        o_groups=2,
        o_lora_rank=4,
        index_n_heads=4,
        index_head_dim=4,
        index_topk=2,
        compress_rates={"compressed_sparse_attention": 2, "heavily_compressed_attention": 4},
    )
    config["moe"].update(num_routed_experts=4, num_hash_layers=3, expert_intermediate_size=16)
    config["train"].update(
        max_steps=2,
        seq_len=8,
        micro_batch_size=2,
        gradient_accumulation_steps=1,
        warmup_steps=1,
        valid_steps=1,
        valid_max_batches=1,
        target_tokens=32,
    )
    config["data"].update(vocab_size=64, num_workers=0, pin_memory=False, persistent_workers=False)
    return config
