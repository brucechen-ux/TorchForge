from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from experiments.dsv3_reference.config import DSV3ReferenceConfig
from experiments.dsv3_reference.model import DSV3ReferenceModel
from experiments.dsv3_torchforge.diagnostics import run_forward_backward_diagnostics
from experiments.dsv3_torchforge.train import TrainConfig, make_batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--num-attention-heads", type=int, default=4)
    parser.add_argument("--intermediate-size", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--seq-length", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", default="experiments/dsv3_torchforge/diagnostics_self_test.json")
    args = parser.parse_args()

    model_config = _model_config_from_args(args)
    train_config = TrainConfig(
        seed=args.seed,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        device=args.device,
    )
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    reference_model = DSV3ReferenceModel(model_config).to(device)
    target_model = DSV3ReferenceModel(model_config).to(device)
    target_model.load_state_dict(reference_model.state_dict())
    reference_model.train()
    target_model.train()
    input_ids, labels = make_batch(
        step=0,
        model_config=model_config,
        train_config=train_config,
        device=device,
    )
    report = run_forward_backward_diagnostics(
        reference_model,
        target_model,
        input_ids=input_ids,
        labels=labels,
        copy_weights=False,
    )
    report["self_test"] = _self_test_summary(report)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.output}")
    if not report["self_test"]["passed"]:
        raise SystemExit(_format_self_test_failure(report))


def _model_config_from_args(args: argparse.Namespace) -> DSV3ReferenceConfig:
    return DSV3ReferenceConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_attention_heads,
        q_lora_rank=max(1, args.hidden_size // 4),
        kv_lora_rank=max(1, args.hidden_size // 4),
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=args.hidden_size // args.num_attention_heads,
        intermediate_size=args.intermediate_size,
    )


def _self_test_summary(report: dict[str, Any]) -> dict[str, Any]:
    parameter_max = _max_diff(report, "parameter_diff")
    activation_max = _max_diff(report, "activation_diff")
    gradient_max = _max_diff(report, "gradient_diff")
    loss_diff = float((report.get("loss") or {}).get("abs_diff", 0.0))
    passed = parameter_max == 0.0 and activation_max == 0.0 and gradient_max == 0.0 and loss_diff == 0.0
    return {
        "passed": passed,
        "loss_abs_diff": loss_diff,
        "parameter_max_abs_diff": parameter_max,
        "activation_max_abs_diff": activation_max,
        "gradient_max_abs_diff": gradient_max,
    }


def _max_diff(report: dict[str, Any], key: str) -> float:
    return float(((report.get(key) or {}).get("summary") or {}).get("max_abs_diff", 0.0))


def _format_self_test_failure(report: dict[str, Any]) -> str:
    summary = report["self_test"]
    return (
        "Diagnostics self-test failed: "
        f"loss={summary['loss_abs_diff']}, "
        f"parameter={summary['parameter_max_abs_diff']}, "
        f"activation={summary['activation_max_abs_diff']}, "
        f"gradient={summary['gradient_max_abs_diff']}."
    )


if __name__ == "__main__":
    main()

