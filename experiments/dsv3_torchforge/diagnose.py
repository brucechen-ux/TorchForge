from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from experiments.dsv3_reference.config import DSV3ReferenceConfig
from experiments.dsv3_reference.model import DSV3ReferenceModel
from experiments.dsv3_torchforge.diagnostics import run_forward_backward_diagnostics
from experiments.dsv3_torchforge.model import ComponentConfig, build_model
from experiments.dsv3_torchforge.train import TrainConfig, make_batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attention", choices=["pytorch", "torchforge"], default="torchforge")
    parser.add_argument("--norm", choices=["pytorch", "torchforge"], default="pytorch")
    parser.add_argument("--ffn", choices=["pytorch", "torchforge", "moe"], default="pytorch")
    parser.add_argument("--kv", choices=["pytorch", "torchforge"], default="pytorch")
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--num-attention-heads", type=int, default=4)
    parser.add_argument("--intermediate-size", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--seq-length", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--copy-reference-weights", action="store_true")
    parser.add_argument("--output", default="experiments/dsv3_torchforge/diagnostics.json")
    args = parser.parse_args()

    model_config = DSV3ReferenceConfig(
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
    train_config = TrainConfig(
        seed=args.seed,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        device=args.device,
    )
    components = ComponentConfig(attention=args.attention, norm=args.norm, ffn=args.ffn, kv=args.kv)
    device = torch.device(args.device)

    torch.manual_seed(args.seed)
    reference_model = DSV3ReferenceModel(model_config).to(device)
    torch.manual_seed(args.seed)
    target_model = build_model(model_config, components).to(device)
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
        copy_weights=args.copy_reference_weights,
        strict_copy=True,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

