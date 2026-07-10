from __future__ import annotations

import importlib.util
import pathlib

import torch

_ASSEMBLY_PATH = pathlib.Path(__file__).resolve().parents[1] / "experiments" / "dsv4_assembly" / "deepseek_v4_assembly.py"
_SPEC = importlib.util.spec_from_file_location("deepseek_v4_assembly", _ASSEMBLY_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Could not load DeepSeek-V4 assembly from {_ASSEMBLY_PATH}.")
_ASSEMBLY = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_ASSEMBLY)

build_deepseek_v4_components = _ASSEMBLY.build_deepseek_v4_components
forward_deepseek_v4_components = _ASSEMBLY.forward_deepseek_v4_components
tiny_deepseek_v4_config = _ASSEMBLY.tiny_deepseek_v4_config
train_deepseek_v4_components = _ASSEMBLY.train_deepseek_v4_components
router_has_correction_bias = _ASSEMBLY.router_has_correction_bias
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


def test_tiny_config_preserves_dsv4_training_mechanisms() -> None:
    config = tiny_deepseek_v4_config(variant="flash")
    assert config["mtp_depth"] == 1
    assert config["router_score_correction_bias"] is True
    assert config["moe_aux_loss_alpha"] > 0.0
    assert config["expert_clamp_limit"] == 10.0


def test_forward_return_dict_includes_mtp_and_loss_terms() -> None:
    config = tiny_deepseek_v4_config(variant="flash")
    components = build_deepseek_v4_components(config)
    input_ids = torch.randint(0, config["vocab_size"], (2, 8))
    outputs = forward_deepseek_v4_components(
        components,
        input_ids,
        labels=input_ids.clone(),
        return_dict=True,
    )
    assert outputs["logits"].shape == (2, 8, config["vocab_size"])
    assert outputs["mtp_logits"].shape == (2, 7, config["vocab_size"])
    assert outputs["loss"].dim() == 0
    assert outputs["lm_loss"].dim() == 0
    assert outputs["mtp_loss"].dim() == 0
    assert outputs["moe_aux_loss"].dim() == 0


def test_dsv4_moe_enables_clamp_and_correction_bias_only_for_learned_router() -> None:
    config = tiny_deepseek_v4_config(variant="flash")
    components = build_deepseek_v4_components(config)
    hash_moe = components["layers"][0]["ffn"]
    learned_moe = components["layers"][config["hash_routing_layers"]]["ffn"]

    assert hash_moe.experts[0].clamp_limit == 10.0
    assert hash_moe.shared_expert.expert.clamp_limit == 10.0
    assert not router_has_correction_bias(hash_moe)
    assert router_has_correction_bias(learned_moe)


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
