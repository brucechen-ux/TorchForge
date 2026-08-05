from __future__ import annotations

import json
from pathlib import Path
import struct

import torch

from torchforge.model.k3_assembly.config import load_config, small_k3_config, tiny_k3_config
from torchforge.model.k3_assembly.data import build_dataloaders
from torchforge.model.k3_assembly.model import SmallK3Model, architecture_summary
from torchforge.model.k3_assembly.optim import WarmupCosineScheduler, build_optimizer
from torchforge.model.k3_assembly.train import load_checkpoint, save_checkpoint
from torchforge.common.moe import StableLatentMoE


def test_tiny_k3_assembly_and_training_smoke() -> None:
    config = tiny_k3_config()
    model = SmallK3Model(config)
    summary = architecture_summary(model)
    input_ids = torch.randint(0, config["model"]["vocab_size"], (2, 8))
    labels = torch.randint(0, config["model"]["vocab_size"], (2, 8))

    outputs = model(input_ids, labels=labels)
    outputs["loss"].backward()
    updated_biases = model.update_router_biases(distributed=False)

    assert summary["decoder_layers"] == 5
    assert summary["kda_layers"] == 3
    assert summary["mla_layers"] == 2
    assert summary["dense_layers"] == 1
    assert summary["moe_layers"] == 4
    assert summary["mtp_layers"] == 1
    assert summary["active_parameters"] < summary["total_parameters"]
    assert outputs["logits"].shape == (2, 8, 64)
    assert outputs["mtp_logits"].shape == (2, 7, 64)
    assert torch.isfinite(outputs["loss"])
    assert len(updated_biases) == 5


def test_full_config_has_25_layer_k3_pattern_without_allocating_weights() -> None:
    config = small_k3_config()
    with torch.device("meta"):
        model = SmallK3Model(config)
    summary = architecture_summary(model)

    assert summary["decoder_layers"] == 25
    assert summary["kda_layers"] == 18
    assert summary["mla_layers"] == 7
    assert summary["dense_layers"] == 1
    assert summary["moe_layers"] == 24
    assert config["moe"]["num_experts_per_token"] == 4
    assert config["moe"]["num_routed_experts"] == 224
    assert model.attention_residual.block_size == 8
    assert 1_000_000_000 < summary["total_parameters"] < 1_100_000_000
    assert 200_000_000 < summary["active_parameters"] < 250_000_000


def test_full_config_matches_checked_in_json() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "torchforge"
        / "model"
        / "k3_assembly"
        / "configs"
        / "k3_1b.json"
    )

    assert load_config(config_path) == small_k3_config()


def test_k3_dataloaders_use_independent_train_and_validation_lengths(tmp_path: Path) -> None:
    config = tiny_k3_config()
    tokens = [index % 64 for index in range(65)]
    payload = struct.pack(f"<{len(tokens)}I", *tokens)
    (tmp_path / "train.bin").write_bytes(payload)
    (tmp_path / "valid.bin").write_bytes(payload)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "dtype": "uint32",
                "vocab_size": 64,
                "train_file": "train.bin",
                "valid_file": "valid.bin",
                "train_tokens_written": len(tokens),
                "valid_tokens_written": len(tokens),
            }
        ),
        encoding="utf-8",
    )
    config["data"]["data_dir"] = str(tmp_path)

    train_loader, valid_loader = build_dataloaders(config, rank=0, world_size=1)

    assert train_loader.dataset.seq_len == config["train"]["seq_len"]
    assert valid_loader.dataset.seq_len == config["train"]["validation_seq_len"]


def test_k3_checkpoint_restores_optimizer_router_and_next_loss(tmp_path: Path) -> None:
    torch.manual_seed(29)
    config = tiny_k3_config()
    model = SmallK3Model(config)
    optimizer = build_optimizer(model, config["train"])
    scheduler = WarmupCosineScheduler(
        optimizer,
        base_lr=float(config["train"]["learning_rate"]),
        min_lr=float(config["train"]["min_lr"]),
        warmup_steps=int(config["train"]["warmup_steps"]),
        total_steps=int(config["train"]["max_steps"]),
    )
    input_ids = torch.randint(0, 64, (2, 8))
    labels = torch.randint(0, 64, (2, 8))
    model(input_ids, labels=labels)["loss"].backward()
    optimizer.step()
    expected_biases = model.update_router_biases(distributed=False)
    scheduler.step()
    optimizer.zero_grad()
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=1,
        cumulative_tokens=16,
        data_state={"epoch": 2, "offset": 3},
        config=config,
    )
    expected_loss = model(input_ids, labels=labels)["loss"].detach()

    restored = SmallK3Model(config)
    restored_optimizer = build_optimizer(restored, config["train"])
    restored_scheduler = WarmupCosineScheduler(
        restored_optimizer,
        base_lr=float(config["train"]["learning_rate"]),
        min_lr=float(config["train"]["min_lr"]),
        warmup_steps=int(config["train"]["warmup_steps"]),
        total_steps=int(config["train"]["max_steps"]),
    )
    resume = load_checkpoint(
        checkpoint_path,
        model=restored,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        config=config,
        device=torch.device("cpu"),
    )
    restored_loss = restored(input_ids, labels=labels)["loss"].detach()
    restored_biases = [
        module.router.expert_bias
        for module in restored.modules()
        if isinstance(module, StableLatentMoE)
    ]

    assert resume["data_state"] == {"epoch": 2, "offset": 3}
    assert restored_scheduler.step_number == scheduler.step_number
    assert len(restored_biases) == len(expected_biases)
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(restored_biases, expected_biases)
    )
    assert torch.equal(restored_loss, expected_loss)
