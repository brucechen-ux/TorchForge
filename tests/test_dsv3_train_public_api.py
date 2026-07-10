from __future__ import annotations

import importlib.util
import pathlib

import torch


_ASSEMBLY_PATH = pathlib.Path(__file__).resolve().parents[1] / "experiments" / "dsv3_assembly" / "deepseek_v3_assembly.py"
_SPEC = importlib.util.spec_from_file_location("deepseek_v3_assembly", _ASSEMBLY_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Could not load DeepSeek-V3 assembly from {_ASSEMBLY_PATH}.")
_ASSEMBLY = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_ASSEMBLY)

build_deepseek_v3_components = _ASSEMBLY.build_deepseek_v3_components
forward_deepseek_v3_components = _ASSEMBLY.forward_deepseek_v3_components
tiny_deepseek_v3_config = _ASSEMBLY.tiny_deepseek_v3_config


def test_dsv3_forward_shape() -> None:
    config = tiny_deepseek_v3_config()
    components = build_deepseek_v3_components(config)
    input_ids = torch.randint(0, config["vocab_size"], (2, 8))

    logits = forward_deepseek_v3_components(components, input_ids)

    assert logits.shape == (2, 8, config["vocab_size"])


def test_dsv3_forward_includes_mtp_and_training_losses() -> None:
    config = tiny_deepseek_v3_config()
    components = build_deepseek_v3_components(config)
    input_ids = torch.randint(0, config["vocab_size"], (2, 8))

    outputs = forward_deepseek_v3_components(
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


def test_dsv3_learned_moe_enables_aux_loss_free_balancing() -> None:
    config = tiny_deepseek_v3_config()
    components = build_deepseek_v3_components(config)
    first_moe = components["layers"][config["first_k_dense_replace"]]["ffn"]
    mtp_moe = components["mtp"].transformer_block.layer["ffn"]

    assert first_moe.router.e_score_correction_bias is not None
    assert first_moe.return_aux_loss is True
    assert mtp_moe.router.e_score_correction_bias is not None
    assert mtp_moe.return_aux_loss is True


def test_dsv3_forward_updates_router_bias_for_the_next_step() -> None:
    config = tiny_deepseek_v3_config()
    components = build_deepseek_v3_components(config)
    first_moe = components["layers"][config["first_k_dense_replace"]]["ffn"]
    mtp_moe = components["mtp"].transformer_block.layer["ffn"]
    with torch.no_grad():
        first_moe.router.proj.weight.zero_()
        mtp_moe.router.proj.weight.zero_()
    before = first_moe.router.e_score_correction_bias.detach().clone()
    mtp_before = mtp_moe.router.e_score_correction_bias.detach().clone()
    input_ids = torch.randint(0, config["vocab_size"], (2, 8))

    forward_deepseek_v3_components(
        components,
        input_ids,
        return_dict=True,
        update_router_bias=True,
    )

    assert not torch.equal(first_moe.router.e_score_correction_bias, before)
    assert not torch.equal(mtp_moe.router.e_score_correction_bias, mtp_before)
