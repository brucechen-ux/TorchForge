from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from experiments.dsv3_reference.config import DSV3ReferenceConfig
from experiments.dsv3_reference.model import DSV3ReferenceModel
from experiments.dsv3_torchforge.diagnostics import copy_reference_weights, run_forward_backward_diagnostics
from experiments.dsv3_torchforge.model import ComponentConfig, build_model, variant_name


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 2026
    steps: int = 100
    batch_size: int = 4
    seq_length: int = 16
    learning_rate: float = 1e-3
    device: str = "cpu"
    output: str = ""
    copy_reference_weights: bool = False
    diagnostics: bool = False
    diagnostics_output: str = ""


def parse_args() -> tuple[DSV3ReferenceConfig, TrainConfig, ComponentConfig, str]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attention", choices=["pytorch", "torchforge"], default="pytorch")
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
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--copy-reference-weights", action="store_true")
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--diagnostics-output", type=str, default="")
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
        steps=args.steps,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        learning_rate=args.learning_rate,
        device=args.device,
        output=args.output,
        copy_reference_weights=args.copy_reference_weights,
        diagnostics=args.diagnostics,
        diagnostics_output=args.diagnostics_output,
    )
    components = ComponentConfig(attention=args.attention, norm=args.norm, ffn=args.ffn, kv=args.kv)
    variant = variant_name(components)
    if not train_config.output:
        train_config = TrainConfig(
            seed=train_config.seed,
            steps=train_config.steps,
            batch_size=train_config.batch_size,
            seq_length=train_config.seq_length,
            learning_rate=train_config.learning_rate,
            device=train_config.device,
            output=f"experiments/dsv3_torchforge/{variant}_losses.json",
            copy_reference_weights=train_config.copy_reference_weights,
            diagnostics=train_config.diagnostics,
            diagnostics_output=train_config.diagnostics_output,
        )
    return model_config, train_config, components, variant


def make_batch(
    *,
    step: int,
    model_config: DSV3ReferenceConfig,
    train_config: TrainConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(train_config.seed + step)
    input_ids = torch.randint(
        low=0,
        high=model_config.vocab_size,
        size=(train_config.batch_size, train_config.seq_length),
        generator=generator,
        device=device,
    )
    labels = input_ids.roll(shifts=-1, dims=1)
    return input_ids, labels


def train_model(
    *,
    model_config: DSV3ReferenceConfig,
    train_config: TrainConfig,
    components: ComponentConfig,
    variant: str,
) -> dict[str, Any]:
    torch.manual_seed(train_config.seed)
    device = torch.device(train_config.device)
    model = build_model(model_config, components).to(device)
    weight_copy_report = None
    if train_config.copy_reference_weights:
        torch.manual_seed(train_config.seed)
        reference_for_copy = DSV3ReferenceModel(model_config).to(device)
        weight_copy_report = copy_reference_weights(reference_for_copy, model, strict=True)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate)
    losses: list[dict[str, float]] = []
    use_cuda_memory = device.type == "cuda" and torch.cuda.is_available()

    for step in range(train_config.steps):
        input_ids, labels = make_batch(
            step=step,
            model_config=model_config,
            train_config=train_config,
            device=device,
        )
        if use_cuda_memory:
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        optimizer.zero_grad(set_to_none=True)
        step_start = time.perf_counter()
        forward_start = time.perf_counter()
        outputs = model(input_ids, labels=labels)
        loss = outputs["loss"]
        if loss is None:
            raise RuntimeError("model did not return loss.")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        forward_time_ms = (time.perf_counter() - forward_start) * 1000.0
        backward_start = time.perf_counter()
        loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        backward_time_ms = (time.perf_counter() - backward_start) * 1000.0
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_time_ms = (time.perf_counter() - step_start) * 1000.0
        peak_memory_mb = (
            float(torch.cuda.max_memory_allocated(device) / (1024 * 1024)) if use_cuda_memory else 0.0
        )
        losses.append(
            {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "forward_time_ms": forward_time_ms,
                "backward_time_ms": backward_time_ms,
                "step_time_ms": step_time_ms,
                "peak_memory_mb": peak_memory_mb,
            }
        )

    return {
        "variant": variant,
        "components": components.to_dict(),
        "model_config": model_config.to_dict(),
        "train_config": asdict(train_config),
        "weight_copy": weight_copy_report,
        "losses": losses,
    }


def write_json(result: dict[str, Any], path: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")


def main() -> None:
    model_config, train_config, components, variant = parse_args()
    diagnostics = None
    if train_config.diagnostics:
        diagnostics = run_diagnostics(
            model_config=model_config,
            train_config=train_config,
            components=components,
        )
        if train_config.diagnostics_output:
            write_json(diagnostics, train_config.diagnostics_output)

    result = train_model(
        model_config=model_config,
        train_config=train_config,
        components=components,
        variant=variant,
    )
    if diagnostics is not None:
        result["diagnostics"] = diagnostics
    write_json(result, train_config.output)
    print(f"wrote {train_config.output}")


def run_diagnostics(
    *,
    model_config: DSV3ReferenceConfig,
    train_config: TrainConfig,
    components: ComponentConfig,
) -> dict[str, Any]:
    torch.manual_seed(train_config.seed)
    device = torch.device(train_config.device)
    reference_model = DSV3ReferenceModel(model_config).to(device)
    torch.manual_seed(train_config.seed)
    target_model = build_model(model_config, components).to(device)
    reference_model.train()
    target_model.train()
    input_ids, labels = make_batch(
        step=0,
        model_config=model_config,
        train_config=train_config,
        device=device,
    )
    return run_forward_backward_diagnostics(
        reference_model,
        target_model,
        input_ids=input_ids,
        labels=labels,
        copy_weights=train_config.copy_reference_weights,
        strict_copy=True,
    )


if __name__ == "__main__":
    main()
