from __future__ import annotations

import torch

from experiments.dsv4_assembly.deepseek_v4_assembly import (
    build_deepseek_v4_components,
    forward_deepseek_v4_components,
    tiny_deepseek_v4_config,
    train_deepseek_v4_components,
)
from torchforge.common.loss import CausalLMLoss
from torchforge.common.optim import AdamW, build_param_groups
from torchforge.common.train import TrainStep, random_token_batches


def test_forward_shape_flash() -> None:
    config = tiny_deepseek_v4_config(variant="flash")
    components = build_deepseek_v4_components(config)
    components.eval()
    input_ids = torch.randint(0, config["vocab_size"], (2, 8))
    logits = forward_deepseek_v4_components(components, input_ids)
    assert logits.shape == (2, 8, config["vocab_size"])


def test_forward_shape_pro() -> None:
    config = tiny_deepseek_v4_config(variant="pro")
    components = build_deepseek_v4_components(config)
    components.eval()
    input_ids = torch.randint(0, config["vocab_size"], (2, 8))
    logits = forward_deepseek_v4_components(components, input_ids)
    assert logits.shape == (2, 8, config["vocab_size"])


def test_train_step_runs_without_error() -> None:
    torch.manual_seed(0)
    config = tiny_deepseek_v4_config(variant="flash")
    components = build_deepseek_v4_components(config)
    optimizer = AdamW(build_param_groups(components, weight_decay=0.1), lr=1e-3)
    step = TrainStep(
        forward_fn=lambda ids: forward_deepseek_v4_components(components, ids),
        loss_module=CausalLMLoss(),
        optimizer=optimizer,
    )
    input_ids = torch.randint(0, config["vocab_size"], (2, 8))
    metrics = step.run(input_ids, input_ids.clone())
    assert set(metrics) == {"loss", "grad_norm"}
    assert metrics["loss"] > 0.0
    assert metrics["grad_norm"] >= 0.0


def test_train_loop_runs_n_steps() -> None:
    config = tiny_deepseek_v4_config(variant="flash")
    components = build_deepseek_v4_components(config)
    # Collect losses across a few steps to confirm the loop drives the pipeline end-to-end.
    optimizer = AdamW(build_param_groups(components, weight_decay=0.1), lr=1e-3)
    step = TrainStep(
        forward_fn=lambda ids: forward_deepseek_v4_components(components, ids),
        loss_module=CausalLMLoss(),
        optimizer=optimizer,
    )
    generator = torch.Generator().manual_seed(42)
    losses = [
        step.run(input_ids, labels)["loss"]
        for input_ids, labels in random_token_batches(
            vocab_size=config["vocab_size"],
            batch_size=2,
            seq_length=8,
            num_steps=5,
            generator=generator,
        )
    ]
    assert len(losses) == 5
    assert all(l > 0.0 for l in losses)
