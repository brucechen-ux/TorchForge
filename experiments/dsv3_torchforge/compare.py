from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _load_result(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _summarize(path: str) -> dict[str, Any]:
    result = _load_result(path)
    losses = result["losses"]
    if not losses:
        raise ValueError(f"{path} has no loss records.")
    loss_values = [float(item["loss"]) for item in losses]
    return {
        "path": path,
        "variant": result.get("variant", Path(path).stem),
        "components": result.get("components", {}),
        "num_steps": len(losses),
        "initial_loss": loss_values[0],
        "final_loss": loss_values[-1],
        "mean_loss": _mean(loss_values),
        "mean_forward_time_ms": _mean([float(item.get("forward_time_ms", 0.0)) for item in losses]),
        "mean_backward_time_ms": _mean([float(item.get("backward_time_ms", 0.0)) for item in losses]),
        "mean_step_time_ms": _mean([float(item.get("step_time_ms", 0.0)) for item in losses]),
        "max_peak_memory_mb": max(float(item.get("peak_memory_mb", 0.0)) for item in losses),
        "diagnostics": _summarize_diagnostics(result.get("diagnostics", {})),
        "loss_values": loss_values,
    }


def _summarize_diagnostics(diagnostics: Any) -> dict[str, Any]:
    if not isinstance(diagnostics, dict) or not diagnostics:
        return {}
    return {
        "loss_abs_diff": (diagnostics.get("loss") or {}).get("abs_diff"),
        "activation_diff": _summary_or_empty(diagnostics.get("activation_diff")),
        "gradient_diff": _summary_or_empty(diagnostics.get("gradient_diff")),
        "parameter_diff": _summary_or_empty(diagnostics.get("parameter_diff")),
        "weight_copy": _summarize_weight_copy(diagnostics.get("weight_copy")),
    }


def _summary_or_empty(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    summary = value.get("summary")
    return summary if isinstance(summary, dict) else {}


def _summarize_weight_copy(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    copied = value.get("copied", [])
    unmapped = value.get("unmapped", [])
    return {
        "copied_count": len(copied) if isinstance(copied, list) else 0,
        "unmapped": unmapped if isinstance(unmapped, list) else [],
        "strict": value.get("strict"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+")
    parser.add_argument("--baseline", default="")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    summaries = [_summarize(path) for path in args.results]
    baseline = next((item for item in summaries if item["path"] == args.baseline), summaries[0])
    baseline_losses = baseline["loss_values"]
    comparisons = []
    for item in summaries:
        losses = item["loss_values"]
        if len(losses) != len(baseline_losses):
            raise ValueError(
                f"loss curves must have the same length for comparison, got "
                f"{len(baseline_losses)} and {len(losses)}."
            )
        diffs = [abs(a - b) for a, b in zip(baseline_losses, losses)]
        comparisons.append(
            {
                "variant": item["variant"],
                "baseline_variant": baseline["variant"],
                "initial_abs_diff": diffs[0],
                "final_abs_diff": diffs[-1],
                "max_abs_diff": max(diffs),
                "mean_abs_diff": _mean(diffs),
            }
        )

    report = {
        "baseline": baseline["variant"],
        "runs": [{key: value for key, value in item.items() if key != "loss_values"} for item in summaries],
        "comparisons": comparisons,
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n")


if __name__ == "__main__":
    main()
