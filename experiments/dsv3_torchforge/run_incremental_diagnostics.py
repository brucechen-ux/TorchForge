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
from experiments.dsv3_torchforge.model import ComponentConfig, build_model, variant_name
from experiments.dsv3_torchforge.train import TrainConfig, make_batch


STEPS = {
    "mla": ComponentConfig(attention="torchforge", norm="pytorch", ffn="pytorch", kv="pytorch"),
    "rmsnorm": ComponentConfig(attention="pytorch", norm="torchforge", ffn="pytorch", kv="pytorch"),
    "feedforward": ComponentConfig(attention="pytorch", norm="pytorch", ffn="torchforge", kv="pytorch"),
    "moe": ComponentConfig(attention="torchforge", norm="torchforge", ffn="moe", kv="pytorch"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--components", nargs="*", choices=sorted(STEPS), default=sorted(STEPS))
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--num-attention-heads", type=int, default=4)
    parser.add_argument("--intermediate-size", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--seq-length", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", default="experiments/dsv3_torchforge/diagnostics")
    args = parser.parse_args()

    model_config = _model_config_from_args(args)
    train_config = TrainConfig(
        seed=args.seed,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        device=args.device,
    )
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []

    for name in args.components:
        components = STEPS[name]
        case_config = model_config
        if name == "moe":
            case_config = DSV3ReferenceConfig(
                **{**model_config.to_dict(), "ffn_type": "moe"}
            )
        torch.manual_seed(args.seed)
        reference_model = DSV3ReferenceModel(case_config).to(device)
        torch.manual_seed(args.seed)
        target_model = build_model(case_config, components).to(device)
        reference_model.train()
        target_model.train()
        input_ids, labels = make_batch(
            step=0,
            model_config=case_config,
            train_config=train_config,
            device=device,
        )
        report = run_forward_backward_diagnostics(
            reference_model,
            target_model,
            input_ids=input_ids,
            labels=labels,
            copy_weights=True,
            strict_copy=True,
        )
        report["diagnostic_case"] = {
            "name": name,
            "variant": variant_name(components),
            "components": components.to_dict(),
        }
        path = output_dir / f"{name}_diagnostics.json"
        path.write_text(json.dumps(report, indent=2) + "\n")
        written.append(str(path))
        print(f"wrote {path}")

    index = {"files": written}
    (output_dir / "index.json").write_text(json.dumps(index, indent=2) + "\n")


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


if __name__ == "__main__":
    main()
