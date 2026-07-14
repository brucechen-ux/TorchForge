from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

from .fingerprint_data import load_dataset_fingerprint


METRICS = (
    "lr",
    "total_loss",
    "lm_loss",
    "mtp_loss",
    "aux_loss",
    "grad_norm",
    "muon_update_rms",
    "validation_loss",
)

ALIASES = {
    "step": ("step",),
    "cumulative_tokens": ("cumulative_tokens", "tokens", "train/tokens"),
    "lr": ("lr", "learning_rate", "train/lr"),
    "total_loss": ("total_loss", "train/total_loss", "train/loss"),
    "lm_loss": ("lm_loss", "train/lm_loss", "loss"),
    "mtp_loss": ("mtp_loss", "train/mtp_loss"),
    "aux_loss": ("aux_loss", "train/aux_loss"),
    "grad_norm": ("grad_norm", "train/grad_norm", "train/optimizer_grad_norm"),
    "muon_update_rms": ("muon_update_rms", "train/muon_update_rms"),
    "validation_loss": ("validation_loss", "valid/total_loss", "valid/loss"),
}

BASE_CRITICAL_METADATA_FIELDS = {
    "world_size",
    "tokens_per_step",
    "seed",
    "seq_len",
    "micro_batch_size",
    "gradient_accumulation_steps",
    "num_layers",
    "hidden_size",
    "num_attention_heads",
    "vocab_size",
    "num_routed_experts",
    "num_experts_per_token",
    "num_hash_layers",
    "mtp_depth",
    "mtp_loss_weight",
    "learning_rate",
    "min_lr",
    "warmup_steps",
    "max_steps",
    "weight_decay",
    "gradient_clipping",
    "bf16",
    "optimizer_name",
    "optimizer_betas",
    "optimizer_eps",
    "manifest_sha256",
    "train_file_size",
    "valid_file_size",
    "dataset_id",
    "train_sha256",
    "valid_sha256",
    "protocol_signature",
    "resumed",
}
MUON_CRITICAL_METADATA_FIELDS = {
    "optimizer_momentum",
    "optimizer_nesterov",
    "newton_schulz",
    "newton_schulz_iterations",
    "update_rms_target",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two independently produced loss logs at identical cumulative-token positions."
    )
    parser.add_argument("--torchforge-log", required=True)
    parser.add_argument("--comparison-log", "--reference-log", dest="comparison_log", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--torchforge-meta")
    parser.add_argument("--comparison-meta", "--reference-meta", dest="comparison_meta")
    parser.add_argument("--torchforge-data-dir")
    parser.add_argument("--comparison-data-dir", "--reference-data-dir", dest="comparison_data_dir")
    parser.add_argument("--torchforge-dataset-fingerprint")
    parser.add_argument("--comparison-dataset-fingerprint", "--reference-dataset-fingerprint", dest="comparison_dataset_fingerprint")
    parser.add_argument(
        "--require-identical-token-grid",
        action="store_true",
        help="Fail when either log has a cumulative-token position absent from the other log.",
    )
    parser.add_argument(
        "--comparison-lr-is-next-step",
        action="store_true",
        help="Interpret comparison-log LR as the scheduler value for the next step and shift it back.",
    )
    parser.add_argument(
        "--allow-metadata-mismatch",
        action="store_true",
        help="Record rather than reject mismatched run metadata fields.",
    )
    return parser.parse_args()


def read_log(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Loss log not found: {path}")
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if path.suffix.lower() in {".jsonl", ".json"}:
        records = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number} must contain a JSON object.")
                records.append(value)
        return records
    raise ValueError(f"Unsupported loss log format for {path}; expected .csv, .jsonl, or line-delimited .json.")


def read_run_metadata(
    path: str | Path,
    *,
    data_dir_override: str | Path | None = None,
    dataset_fingerprint: str | Path | None = None,
) -> dict[str, Any]:
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise ValueError(f"Run metadata must be a JSON object: {path}")
    config = metadata.get("config")
    config_file: Path | None = None
    config_path = metadata.get("config_path")
    if config is None and config_path:
        candidate = Path(config_path)
        if not candidate.is_absolute():
            candidate = path.parent / candidate
        if candidate.exists():
            config_file = candidate
            with candidate.open(encoding="utf-8") as handle:
                try:
                    config = json.load(handle)
                except json.JSONDecodeError:
                    try:
                        import yaml  # type: ignore[import-not-found]
                    except ImportError:
                        config = None
                    else:
                        handle.seek(0)
                        config = yaml.safe_load(handle)
    if config is not None and not isinstance(config, dict):
        raise ValueError(f"Embedded run config must be an object: {path}")
    if config and (data_dir_override is not None or not isinstance(metadata.get("data"), dict)):
        data = config.get("data", {})
        data_dir = Path(data_dir_override) if data_dir_override is not None else Path(str(data.get("data_dir", "")))
        candidates = [data_dir]
        if not data_dir.is_absolute() and config_file is not None:
            candidates.append(config_file.parent / data_dir)
        resolved_data_dir = next((candidate.resolve() for candidate in candidates if candidate.exists()), None)
        if resolved_data_dir is not None:
            fingerprint: dict[str, Any] = {"data_dir": str(resolved_data_dir)}
            for key in ("train_file", "valid_file", "manifest_file"):
                name = data.get(key)
                if not name:
                    continue
                data_path = resolved_data_dir / str(name)
                fingerprint[f"{key}_size"] = data_path.stat().st_size if data_path.exists() else None
                if key == "manifest_file" and data_path.exists():
                    fingerprint["manifest_sha256"] = hashlib.sha256(data_path.read_bytes()).hexdigest()
            if dataset_fingerprint is not None:
                full_fingerprint = load_dataset_fingerprint(
                    resolved_data_dir,
                    manifest_file=str(data.get("manifest_file", "manifest.json")),
                    fingerprint_path=dataset_fingerprint,
                )
                if full_fingerprint is not None:
                    fingerprint.update(full_fingerprint)
            metadata["data"] = fingerprint
    return {"metadata": metadata, "config": config or {}}


def _nested(mapping: dict[str, Any], *path: str) -> Any:
    value: Any = mapping
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def canonical_protocol_signature(config: dict[str, Any]) -> dict[str, Any] | None:
    fields = (
        ("model.vocab_size", ("model", "vocab_size")),
        ("model.seq_len", ("model", "seq_len")),
        ("model.num_layers", ("model", "num_layers")),
        ("model.hidden_size", ("model", "hidden_size")),
        ("model.num_attention_heads", ("model", "num_attention_heads")),
        ("model.dense_intermediate_size", ("model", "dense_intermediate_size")),
        ("model.first_dense_layers", ("model", "first_dense_layers")),
        ("model.rms_norm_eps", ("model", "rms_norm_eps")),
        ("model.tie_word_embeddings", ("model", "tie_word_embeddings")),
        ("attention.q_lora_rank", ("v4_attention", "q_lora_rank")),
        ("attention.head_dim", ("v4_attention", "head_dim")),
        ("attention.qk_rope_head_dim", ("v4_attention", "qk_rope_head_dim")),
        ("attention.rope_theta", ("v4_attention", "rope_theta")),
        ("attention.compress_rope_theta", ("v4_attention", "compress_rope_theta")),
        ("attention.sliding_window", ("v4_attention", "sliding_window")),
        ("attention.o_groups", ("v4_attention", "o_groups")),
        ("attention.o_lora_rank", ("v4_attention", "o_lora_rank")),
        ("attention.index_n_heads", ("v4_attention", "index_n_heads")),
        ("attention.index_head_dim", ("v4_attention", "index_head_dim")),
        ("attention.index_topk", ("v4_attention", "index_topk")),
        ("attention.compress_rates", ("v4_attention", "compress_rates")),
        ("moe.num_routed_experts", ("moe", "num_routed_experts")),
        ("moe.num_shared_experts", ("moe", "num_shared_experts")),
        ("moe.num_experts_per_token", ("moe", "num_experts_per_token")),
        ("moe.num_hash_layers", ("moe", "num_hash_layers")),
        ("moe.route_scale", ("moe", "route_scale")),
        ("moe.score_function", ("moe", "score_function")),
        ("moe.swiglu_limit", ("moe", "swiglu_limit")),
        ("moe.aux_loss_weight", ("moe", "aux_loss_weight")),
        ("moe.expert_intermediate_size", ("moe", "expert_intermediate_size")),
        ("moe.use_correction_bias", ("moe", "use_correction_bias")),
        ("moe.use_packed_experts", ("moe", "use_packed_experts")),
        ("mtp.enabled", ("mtp", "enabled")),
        ("mtp.mtp_depth", ("mtp", "mtp_depth")),
        ("mtp.mtp_loss_weight", ("mtp", "mtp_loss_weight")),
        ("mtp.mtp_use_moe", ("mtp", "mtp_use_moe")),
    )
    signature = {name: _nested(config, *path) for name, path in fields}
    return None if any(value is None for value in signature.values()) else signature


def normalized_run_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    metadata, config = payload["metadata"], payload["config"]
    train = config.get("train", {}) if isinstance(config, dict) else {}
    optimizer = train.get("optimizer", {})
    data = metadata.get("data", {})
    return {
        "world_size": metadata.get("world_size"),
        "tokens_per_step": metadata.get("tokens_per_step"),
        "seed": metadata.get("seed", config.get("seed", train.get("seed"))),
        "seq_len": train.get("seq_len", _nested(config, "model", "seq_len")),
        "micro_batch_size": train.get("micro_batch_size"),
        "gradient_accumulation_steps": train.get("gradient_accumulation_steps"),
        "num_layers": _nested(config, "model", "num_layers"),
        "hidden_size": _nested(config, "model", "hidden_size"),
        "num_attention_heads": _nested(config, "model", "num_attention_heads"),
        "vocab_size": _nested(config, "model", "vocab_size"),
        "num_routed_experts": _nested(config, "moe", "num_routed_experts"),
        "num_experts_per_token": _nested(config, "moe", "num_experts_per_token"),
        "num_hash_layers": _nested(config, "moe", "num_hash_layers"),
        "mtp_depth": _nested(config, "mtp", "mtp_depth"),
        "mtp_loss_weight": _nested(config, "mtp", "mtp_loss_weight"),
        "learning_rate": train.get("learning_rate"),
        "min_lr": train.get("min_lr"),
        "warmup_steps": train.get("warmup_steps"),
        "max_steps": metadata.get("resolved_max_steps", train.get("max_steps")),
        "weight_decay": train.get("weight_decay"),
        "bf16": train.get("bf16"),
        "gradient_clipping": train.get("gradient_clipping", 1.0),
        "target_tokens": metadata.get("target_tokens", train.get("target_tokens")),
        "optimizer_name": optimizer.get("name"),
        "optimizer_momentum": optimizer.get("momentum"),
        "optimizer_nesterov": optimizer.get("nesterov"),
        "optimizer_betas": optimizer.get("betas"),
        "optimizer_eps": optimizer.get("eps"),
        "newton_schulz": optimizer.get("newton_schulz"),
        "newton_schulz_iterations": optimizer.get("newton_schulz_iterations"),
        "update_rms_target": optimizer.get("update_rms_target"),
        "manifest_sha256": data.get("manifest_sha256"),
        "train_file_size": data.get("train_file_size"),
        "valid_file_size": data.get("valid_file_size"),
        "validation_autocast_bf16": metadata.get("validation_autocast_bf16"),
        "initialization_id": _nested(metadata, "initialization", "initialization_id"),
        "dataset_fingerprint_sha256": data.get("dataset_fingerprint_sha256"),
        "dataset_id": data.get("dataset_id"),
        "train_sha256": data.get("train_sha256"),
        "valid_sha256": data.get("valid_sha256"),
        "protocol_signature": canonical_protocol_signature(config),
        "resumed": bool(metadata.get("resume") or metadata.get("resume_from_checkpoint")),
    }


def compare_run_metadata(
    torchforge: dict[str, Any], comparison: dict[str, Any]
) -> dict[str, Any]:
    torchforge_values = normalized_run_metadata(torchforge)
    comparison_values = normalized_run_metadata(comparison)
    matched = {}
    mismatched = {}
    unavailable = []
    for field in torchforge_values:
        left, right = torchforge_values[field], comparison_values[field]
        if left is None or right is None:
            unavailable.append(field)
        elif left == right:
            matched[field] = left
        else:
            mismatched[field] = {"torchforge": left, "comparison": right}
    if torchforge_values["world_size"] == comparison_values["world_size"] == 1:
        mismatched["single_rank_sampler"] = {
            "torchforge": "DistributedSampler",
            "comparison": "RandomSampler",
        }
    if torchforge_values["resumed"] or comparison_values["resumed"]:
        mismatched["resumed_curve"] = {
            "torchforge": torchforge_values["resumed"],
            "comparison": comparison_values["resumed"],
        }
    critical_fields = set(BASE_CRITICAL_METADATA_FIELDS)
    if torchforge_values["optimizer_name"] == comparison_values["optimizer_name"] == "muon":
        critical_fields.update(MUON_CRITICAL_METADATA_FIELDS)
    critical_unavailable = sorted(field for field in unavailable if field in critical_fields)
    return {
        "matched": matched,
        "mismatched": mismatched,
        "unavailable": unavailable,
        "critical_unavailable": critical_unavailable,
    }


def _first(record: dict[str, Any], aliases: Iterable[str]) -> Any:
    for name in aliases:
        value = record.get(name)
        if value is not None and value != "":
            return value
    return None


def _finite_float(value: Any, *, field: str) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric, got {value!r}.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite, got {value!r}.")
    return result


def normalize_log(records: Iterable[dict[str, Any]], *, source: str) -> dict[int, dict[str, Any]]:
    normalized: dict[int, dict[str, Any]] = {}
    for record_index, record in enumerate(records, start=1):
        token_value = _first(record, ALIASES["cumulative_tokens"])
        if token_value is None:
            continue
        tokens_float = _finite_float(token_value, field=f"{source} cumulative_tokens at record {record_index}")
        assert tokens_float is not None
        tokens = int(tokens_float)
        if tokens_float != tokens or tokens < 0:
            raise ValueError(f"{source} cumulative_tokens must be a non-negative integer, got {token_value!r}.")
        if tokens in normalized:
            raise ValueError(f"{source} contains duplicate cumulative_tokens={tokens}.")

        step_value = _first(record, ALIASES["step"])
        step_float = _finite_float(step_value, field=f"{source} step at cumulative_tokens={tokens}")
        step = None if step_float is None else int(step_float)
        if step_float is not None and (step_float != step or step < 0):
            raise ValueError(f"{source} step must be a non-negative integer, got {step_value!r}.")

        row: dict[str, Any] = {"step": step, "cumulative_tokens": tokens}
        for metric in METRICS:
            row[metric] = _finite_float(
                _first(record, ALIASES[metric]),
                field=f"{source} {metric} at cumulative_tokens={tokens}",
            )
        normalized[tokens] = row
    if not normalized:
        raise ValueError(f"{source} has no records containing cumulative token counts.")
    return normalized


def _difference(actual: float | None, comparison: float | None) -> tuple[float | None, float | None]:
    if actual is None or comparison is None:
        return None, None
    absolute = abs(actual - comparison)
    relative = absolute / max(abs(comparison), 1.0e-30)
    return absolute, relative


def _scheduler_parameters(payload: dict[str, Any]) -> dict[str, float | int] | None:
    metadata, config = payload["metadata"], payload["config"]
    train = config.get("train", {})
    base_lr = train.get("learning_rate")
    warmup_steps = train.get("warmup_steps")
    total_steps = metadata.get("resolved_max_steps", train.get("max_steps"))
    min_lr = train.get("min_lr")
    if min_lr is None and train.get("min_lr_ratio") is not None and base_lr is not None:
        min_lr = float(base_lr) * float(train["min_lr_ratio"])
    if None in {base_lr, min_lr, warmup_steps, total_steps}:
        return None
    return {
        "base_lr": float(base_lr),
        "min_lr": float(min_lr),
        "warmup_steps": int(warmup_steps),
        "total_steps": int(total_steps),
    }


def _scheduled_lr(parameters: dict[str, float | int], step_index: int) -> float:
    base_lr = float(parameters["base_lr"])
    min_lr = float(parameters["min_lr"])
    warmup_steps = max(int(parameters["warmup_steps"]), 0)
    total_steps = max(int(parameters["total_steps"]), 1)
    if warmup_steps and step_index < warmup_steps:
        return base_lr * float(step_index + 1) / float(warmup_steps)
    if total_steps <= warmup_steps:
        return base_lr
    progress = (step_index - warmup_steps) / float(total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + (base_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))


def convert_next_step_lr_to_used(
    series: dict[int, dict[str, Any]],
    *,
    run_metadata: dict[str, Any] | None = None,
) -> str:
    """Convert post-scheduler LR logs into the LR used by each completed step."""

    scheduler = _scheduler_parameters(run_metadata) if run_metadata is not None else None
    previous_step: int | None = None
    previous_next_lr: float | None = None
    for tokens in sorted(series):
        row = series[tokens]
        current_step = row["step"]
        next_lr = row["lr"]
        if scheduler is not None and current_step is not None:
            expected_next_lr = _scheduled_lr(scheduler, current_step)
            if next_lr is not None and not math.isclose(next_lr, expected_next_lr, rel_tol=1.0e-12, abs_tol=1.0e-15):
                raise ValueError(
                    f"Logged next-step LR {next_lr} at step {current_step} does not match "
                    f"metadata-derived LR {expected_next_lr}."
                )
            row["lr"] = _scheduled_lr(scheduler, current_step - 1)
        else:
            row["lr"] = (
                previous_next_lr
                if previous_step is not None and current_step is not None and current_step == previous_step + 1
                else None
            )
        previous_step = current_step
        previous_next_lr = next_lr
    return "metadata_reconstructed_and_checked" if scheduler is not None else "adjacent_log_shift"


def output_fields() -> list[str]:
    fields = ["step", "comparison_step", "cumulative_tokens"]
    for metric in METRICS:
        fields.extend(
            [
                metric,
                f"comparison_{metric}",
                f"{metric}_absolute_difference",
                f"{metric}_relative_difference",
            ]
        )
    return fields


def compare_logs(
    torchforge: dict[int, dict[str, Any]],
    comparison: dict[int, dict[str, Any]],
    *,
    require_identical_token_grid: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    torchforge_tokens = set(torchforge)
    comparison_tokens = set(comparison)
    if require_identical_token_grid and torchforge_tokens != comparison_tokens:
        raise ValueError(
            "Cumulative-token grids differ: "
            f"TorchForge-only={len(torchforge_tokens - comparison_tokens)}, "
            f"comparison-only={len(comparison_tokens - torchforge_tokens)}."
        )
    common_tokens = sorted(torchforge_tokens & comparison_tokens)
    if not common_tokens:
        raise ValueError("The logs have no identical cumulative-token positions to compare.")

    rows = []
    for tokens in common_tokens:
        actual = torchforge[tokens]
        other = comparison[tokens]
        row: dict[str, Any] = {
            "step": actual["step"],
            "comparison_step": other["step"],
            "cumulative_tokens": tokens,
        }
        for metric in METRICS:
            absolute, relative = _difference(actual[metric], other[metric])
            row[metric] = actual[metric]
            row[f"comparison_{metric}"] = other[metric]
            row[f"{metric}_absolute_difference"] = absolute
            row[f"{metric}_relative_difference"] = relative
        rows.append(row)

    summary = {
        "torchforge_points": len(torchforge),
        "comparison_points": len(comparison),
        "aligned_points": len(rows),
        "torchforge_only_points": len(torchforge_tokens - comparison_tokens),
        "comparison_only_points": len(comparison_tokens - torchforge_tokens),
        "first_aligned_cumulative_tokens": common_tokens[0],
        "last_aligned_cumulative_tokens": common_tokens[-1],
        "metrics": {},
    }
    for metric in METRICS:
        absolute_values = [row[f"{metric}_absolute_difference"] for row in rows]
        relative_values = [row[f"{metric}_relative_difference"] for row in rows]
        absolute_values = [value for value in absolute_values if value is not None]
        relative_values = [value for value in relative_values if value is not None]
        summary["metrics"][metric] = {
            "compared_points": len(absolute_values),
            "mean_absolute_difference": (
                sum(absolute_values) / len(absolute_values) if absolute_values else None
            ),
            "max_absolute_difference": max(absolute_values) if absolute_values else None,
            "max_relative_difference": max(relative_values) if relative_values else None,
        }
    return rows, summary


def _format(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.9g}"
    return str(value)


def write_comparison(
    output_dir: str | Path,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    *,
    torchforge_log: str | Path,
    comparison_log: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "loss_curve_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields())
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        **summary,
        "torchforge_log": str(Path(torchforge_log).resolve()),
        "comparison_log": str(Path(comparison_log).resolve()),
        "alignment_key": "cumulative_tokens",
        "interpolation": False,
    }
    with (output_dir / "loss_curve_comparison.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)

    lines = [
        "# Cross-project loss curve comparison",
        "",
        "The comparison project is a peer implementation, not a numerical oracle. "
        "Rows are joined only at exact cumulative-token positions; no interpolation is used.",
        "",
        f"- TorchForge log: `{payload['torchforge_log']}`",
        f"- Comparison log: `{payload['comparison_log']}`",
        f"- Aligned points: `{summary['aligned_points']}`",
        f"- TorchForge-only points: `{summary['torchforge_only_points']}`",
        f"- Comparison-only points: `{summary['comparison_only_points']}`",
    ]
    metadata = summary.get("run_metadata_comparison")
    if isinstance(metadata, dict):
        lines.extend(
            [
                f"- Matched run-metadata fields: `{len(metadata['matched'])}`",
                f"- Mismatched run-metadata fields: `{len(metadata['mismatched'])}`",
                f"- Unavailable run-metadata fields: `{len(metadata['unavailable'])}`",
                f"- Critical unavailable fields: `{len(metadata['critical_unavailable'])}`",
            ]
        )
    else:
        lines.append("- Run metadata: `not checked`")
    lines.extend(
        [
            "",
            "| metric | compared points | mean abs diff | max abs diff | max relative diff |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for metric in METRICS:
        values = summary["metrics"][metric]
        lines.append(
            f"| {metric} | {values['compared_points']} | "
            f"{_format(values['mean_absolute_difference'])} | "
            f"{_format(values['max_absolute_difference'])} | "
            f"{_format(values['max_relative_difference'])} |"
        )
    lines.extend(
        [
            "",
            "The complete per-token-position values and differences are in `loss_curve_comparison.csv`. "
            "Blank cells mean that the source log did not expose that metric.",
            "",
        ]
    )
    (output_dir / "loss_curve_comparison.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if bool(args.torchforge_meta) != bool(args.comparison_meta):
        raise ValueError("Pass both --torchforge-meta and --comparison-meta, or pass neither.")
    torchforge_metadata = comparison_metadata = None
    if args.torchforge_meta:
        torchforge_metadata = read_run_metadata(
            args.torchforge_meta,
            data_dir_override=args.torchforge_data_dir,
            dataset_fingerprint=args.torchforge_dataset_fingerprint,
        )
        comparison_metadata = read_run_metadata(
            args.comparison_meta,
            data_dir_override=args.comparison_data_dir,
            dataset_fingerprint=args.comparison_dataset_fingerprint,
        )

    torchforge = normalize_log(read_log(args.torchforge_log), source="TorchForge log")
    comparison = normalize_log(read_log(args.comparison_log), source="comparison log")
    lr_conversion = "used_as_logged"
    if args.comparison_lr_is_next_step:
        lr_conversion = convert_next_step_lr_to_used(comparison, run_metadata=comparison_metadata)
    rows, summary = compare_logs(
        torchforge,
        comparison,
        require_identical_token_grid=args.require_identical_token_grid,
    )
    summary["comparison_lr_semantics"] = lr_conversion
    if torchforge_metadata is not None and comparison_metadata is not None:
        metadata_comparison = compare_run_metadata(
            torchforge_metadata,
            comparison_metadata,
        )
        summary["run_metadata_comparison"] = metadata_comparison
        if (
            metadata_comparison["mismatched"] or metadata_comparison["critical_unavailable"]
        ) and not args.allow_metadata_mismatch:
            raise ValueError(
                "Run metadata is not comparable: "
                + json.dumps(
                    {
                        "mismatched": metadata_comparison["mismatched"],
                        "critical_unavailable": metadata_comparison["critical_unavailable"],
                    },
                    sort_keys=True,
                )
            )
    else:
        summary["run_metadata_comparison"] = "not_checked"
    write_comparison(
        args.output_dir,
        rows,
        summary,
        torchforge_log=args.torchforge_log,
        comparison_log=args.comparison_log,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
