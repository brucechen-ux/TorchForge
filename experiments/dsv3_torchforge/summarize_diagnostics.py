from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("diagnostics", nargs="+")
    parser.add_argument("--output", default="")
    parser.add_argument("--strict-tolerance", type=float, default=0.0)
    args = parser.parse_args()

    summaries = [_summarize_path(path, strict_tolerance=args.strict_tolerance) for path in args.diagnostics]
    report = {
        "strict_tolerance": args.strict_tolerance,
        "runs": summaries,
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n")


def _summarize_path(path: str, *, strict_tolerance: float) -> dict[str, Any]:
    data = json.loads(Path(path).read_text())
    weight_copy = data.get("weight_copy") or {}
    parameter_max = _max_diff(data, "parameter_diff")
    activation_max = _max_diff(data, "activation_diff")
    gradient_max = _max_diff(data, "gradient_diff")
    loss_diff = float((data.get("loss") or {}).get("abs_diff", 0.0))
    strict = (
        loss_diff <= strict_tolerance
        and parameter_max <= strict_tolerance
        and activation_max <= strict_tolerance
        and gradient_max <= strict_tolerance
        and not weight_copy.get("unmapped")
    )
    return {
        "path": path,
        "case": data.get("diagnostic_case", {}).get("name", Path(path).stem),
        "variant": data.get("diagnostic_case", {}).get("variant", ""),
        "components": data.get("diagnostic_case", {}).get("components", {}),
        "weight_copy": {
            "success": bool(weight_copy) and not weight_copy.get("unmapped"),
            "copied_count": len(weight_copy.get("copied", [])) if isinstance(weight_copy.get("copied"), list) else 0,
            "unmapped": weight_copy.get("unmapped", []),
        },
        "loss_abs_diff": loss_diff,
        "parameter_diff": _summary(data, "parameter_diff"),
        "activation_diff": _summary(data, "activation_diff"),
        "gradient_diff": _summary(data, "gradient_diff"),
        "strict_replacement": strict,
        "training_compatible": not strict,
    }


def _summary(data: dict[str, Any], key: str) -> dict[str, float]:
    summary = ((data.get(key) or {}).get("summary") or {})
    return {
        "max_abs_diff": float(summary.get("max_abs_diff", 0.0)),
        "mean_abs_diff": float(summary.get("mean_abs_diff", 0.0)),
    }


def _max_diff(data: dict[str, Any], key: str) -> float:
    return _summary(data, key)["max_abs_diff"]


if __name__ == "__main__":
    main()

