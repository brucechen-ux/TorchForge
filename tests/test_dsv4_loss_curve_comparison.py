from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from experiments.dsv4_muon_report_aligned.config import report_aligned_config
from experiments.dsv4_muon_report_aligned.compare_curves import (
    compare_logs,
    compare_run_metadata,
    convert_next_step_lr_to_used,
    normalize_log,
    read_run_metadata,
    read_log,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_curve_comparison_joins_exact_cumulative_tokens_not_line_numbers(tmp_path: Path) -> None:
    torchforge_path = tmp_path / "torchforge.jsonl"
    comparison_path = tmp_path / "comparison.jsonl"
    _write_jsonl(
        torchforge_path,
        [
            {"step": 1, "cumulative_tokens": 100, "total_loss": 4.0, "lm_loss": 3.5, "lr": 1.0e-4},
            {"step": 2, "cumulative_tokens": 200, "total_loss": 3.0, "lm_loss": 2.5, "lr": 2.0e-4},
        ],
    )
    _write_jsonl(
        comparison_path,
        [
            {"step": 7, "train/tokens": 200, "train/total_loss": 3.25, "train/lm_loss": 2.75, "train/lr": 2.0e-4},
            {"step": 8, "train/tokens": 300, "train/total_loss": 2.0, "train/lm_loss": 1.8, "train/lr": 3.0e-4},
        ],
    )

    torchforge = normalize_log(read_log(torchforge_path), source="TorchForge")
    comparison = normalize_log(read_log(comparison_path), source="comparison")
    rows, summary = compare_logs(torchforge, comparison)

    assert len(rows) == 1
    assert rows[0]["cumulative_tokens"] == 200
    assert rows[0]["step"] == 2
    assert rows[0]["comparison_step"] == 7
    assert rows[0]["total_loss_absolute_difference"] == pytest.approx(0.25)
    assert rows[0]["lm_loss_relative_difference"] == pytest.approx(0.25 / 2.75)
    assert summary["torchforge_only_points"] == 1
    assert summary["comparison_only_points"] == 1


def test_reference_style_csv_loss_is_lm_loss_not_total_loss(tmp_path: Path) -> None:
    path = tmp_path / "metrics.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "tokens", "loss", "learning_rate"])
        writer.writeheader()
        writer.writerow({"step": 1, "tokens": 128, "loss": 4.5, "learning_rate": 3.0e-4})

    rows = normalize_log(read_log(path), source="comparison")

    assert rows[128]["lm_loss"] == 4.5
    assert rows[128]["total_loss"] is None


def test_curve_comparison_rejects_duplicate_token_positions() -> None:
    records = [
        {"step": 1, "cumulative_tokens": 100, "total_loss": 4.0},
        {"step": 2, "cumulative_tokens": 100, "total_loss": 3.0},
    ]

    with pytest.raises(ValueError, match="duplicate cumulative_tokens=100"):
        normalize_log(records, source="TorchForge")


def test_next_step_lr_is_shifted_to_the_step_where_it_was_used() -> None:
    series = normalize_log(
        [
            {"step": 1, "train/tokens": 100, "train/lr": 2.0e-4},
            {"step": 2, "train/tokens": 200, "train/lr": 3.0e-4},
            {"step": 4, "train/tokens": 400, "train/lr": 4.0e-4},
        ],
        source="comparison",
    )

    mode = convert_next_step_lr_to_used(series)

    assert mode == "adjacent_log_shift"
    assert series[100]["lr"] is None
    assert series[200]["lr"] == 2.0e-4
    assert series[400]["lr"] is None


def test_next_step_lr_is_reconstructed_from_run_metadata() -> None:
    series = normalize_log(
        [
            {"step": 1, "train/tokens": 100, "train/lr": 4.0e-4},
            {"step": 2, "train/tokens": 200, "train/lr": 6.0e-4},
        ],
        source="comparison",
    )
    metadata = {
        "metadata": {"resolved_max_steps": 8},
        "config": {"train": {"learning_rate": 1.0e-3, "min_lr": 1.0e-4, "warmup_steps": 5, "max_steps": 8}},
    }

    mode = convert_next_step_lr_to_used(series, run_metadata=metadata)

    assert mode == "metadata_reconstructed_and_checked"
    assert series[100]["lr"] == pytest.approx(1.0e-3 / 5)
    assert series[200]["lr"] == pytest.approx(2.0e-3 / 5)


def test_strict_curve_comparison_rejects_different_token_grids() -> None:
    torchforge = normalize_log([{"step": 1, "cumulative_tokens": 100}], source="TorchForge")
    comparison = normalize_log([{"step": 1, "cumulative_tokens": 200}], source="comparison")

    with pytest.raises(ValueError, match="Cumulative-token grids differ"):
        compare_logs(torchforge, comparison, require_identical_token_grid=True)


def test_run_metadata_comparison_reads_peer_config_path(tmp_path: Path) -> None:
    config = {
        "seed": 2026,
        "model": {"seq_len": 8, "num_layers": 4, "hidden_size": 32, "num_attention_heads": 4, "vocab_size": 64},
        "moe": {"num_routed_experts": 4, "num_experts_per_token": 1, "num_hash_layers": 3},
        "mtp": {"mtp_depth": 1, "mtp_loss_weight": 0.1},
        "train": {"seq_len": 8, "micro_batch_size": 2, "gradient_accumulation_steps": 1},
    }
    config_path = tmp_path / "peer_config.yaml"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    torchforge_meta = tmp_path / "torchforge_meta.json"
    peer_meta = tmp_path / "peer_meta.json"
    torchforge_meta.write_text(
        json.dumps({"world_size": 2, "tokens_per_step": 32, "seed": 2026, "config": config}),
        encoding="utf-8",
    )
    peer_meta.write_text(
        json.dumps({"world_size": 2, "tokens_per_step": 32, "config_path": str(config_path)}),
        encoding="utf-8",
    )

    result = compare_run_metadata(read_run_metadata(torchforge_meta), read_run_metadata(peer_meta))

    assert result["mismatched"] == {}
    assert result["matched"]["tokens_per_step"] == 32
    assert result["matched"]["hidden_size"] == 32


def test_protocol_signature_detects_moe_behavior_difference() -> None:
    torchforge_config = report_aligned_config()
    peer_config = json.loads(json.dumps(torchforge_config))
    peer_config["moe"]["route_scale"] = 1.0
    torchforge = {"metadata": {"world_size": 8}, "config": torchforge_config}
    peer = {"metadata": {"world_size": 8}, "config": peer_config}

    result = compare_run_metadata(torchforge, peer)

    assert "protocol_signature" in result["mismatched"]


def test_resumed_native_curve_is_rejected_by_metadata_comparison() -> None:
    config = report_aligned_config()
    torchforge = {"metadata": {"world_size": 8, "resume": "step.pt"}, "config": config}
    peer = {"metadata": {"world_size": 8, "resume_from_checkpoint": None}, "config": config}

    result = compare_run_metadata(torchforge, peer)

    assert result["mismatched"]["resumed_curve"] == {"torchforge": True, "comparison": False}
