from __future__ import annotations

import copy
import json
import math
import socket
import struct
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from experiments.dsv4_muon_report_aligned.config import load_config, tiny_parity_config
from experiments.dsv4_muon_report_aligned.data import MemmapTokenDataset
from experiments.dsv4_muon_report_aligned.model import ReportAlignedDeepSeekV4, load_reference_weights
from experiments.dsv4_muon_report_aligned.optim import (
    HybridOptimizer,
    WarmupCosineScheduler,
    build_optimizer,
    parameter_group_rows,
)
from experiments.dsv4_muon_report_aligned.parity import (
    forward_parity,
    import_reference,
    training_parity,
)
from experiments.dsv4_muon_report_aligned.train import load_checkpoint, save_checkpoint
from torchforge.common.optim import Muon
from torchforge.common.optim.muon import _newton_schulz_orthogonalize, _scale_muon_update


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "dsv4_muon_report_aligned"
REFERENCE = ROOT.parent / "deepseek_v4_muon_report_aligned_package_20260713"


def _manual_ns(matrix: torch.Tensor, coefficients: list[tuple[float, float, float]]) -> torch.Tensor:
    work = matrix.float()
    max_abs = work.abs().max()
    norm = (work / max_abs).norm() * max_abs if max_abs > 0 else max_abs
    value = matrix.float()
    transposed = value.shape[0] > value.shape[1]
    if transposed:
        value = value.mT
    value = value / norm.clamp_min(1.0e-7)
    for a, b, c in coefficients:
        gram = value @ value.mT
        value = a * value + (b * gram + c * (gram @ gram)) @ value
    return value.mT if transposed else value


def test_newton_schulz_uses_exact_hybrid_8_plus_2_and_standard_10() -> None:
    torch.manual_seed(2026)
    matrix = torch.randn(13, 7)
    hybrid_coefficients = [(3.4445, -4.7750, 2.0315)] * 8 + [(2.0, -1.5, 0.5)] * 2
    standard_coefficients = [(2.0, -1.5, 0.5)] * 10
    torch.testing.assert_close(
        _newton_schulz_orthogonalize(matrix, steps=10, method="hybrid"),
        _manual_ns(matrix, hybrid_coefficients),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        _newton_schulz_orthogonalize(matrix, steps=10, method="standard"),
        _manual_ns(matrix, standard_coefficients),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize("shape", [(32, 32), (16, 64), (64, 16)])
def test_muon_scaled_logical_matrix_rms_is_0_18(shape: tuple[int, int]) -> None:
    torch.manual_seed(sum(shape))
    matrix = torch.randn(*shape)
    update = _newton_schulz_orthogonalize(matrix, steps=10, method="hybrid")
    scaled = _scale_muon_update(update, scale=0.18)
    observed = float(scaled.float().square().mean().sqrt().item())
    assert abs(observed - 0.18) <= 5.0e-4


def test_muon_nesterov_formula_and_single_decay_match_reference() -> None:
    torch.manual_seed(11)
    parameter = nn.Parameter(torch.randn(8, 5))
    old = parameter.detach().clone()
    prior = torch.randn_like(parameter)
    gradient = torch.randn_like(parameter)
    optimizer = Muon(
        [parameter],
        lr=3.0e-4,
        momentum=0.95,
        ns_steps=10,
        ns_method="hybrid",
        nesterov=True,
        weight_decay=0.1,
        update_scale=0.18,
    )
    optimizer.state[parameter]["momentum_buffer"] = prior.clone()
    parameter.grad = gradient.clone()
    expected_m = 0.95 * prior + gradient
    expected_n = 0.95 * expected_m + gradient
    expected_update = _scale_muon_update(
        _newton_schulz_orthogonalize(expected_n, steps=10, method="hybrid"),
        scale=0.18,
    )
    expected_parameter = old * (1.0 - 3.0e-4 * 0.1) - 3.0e-4 * expected_update
    optimizer.step()
    torch.testing.assert_close(optimizer.state[parameter]["momentum_buffer"], expected_m, rtol=0.0, atol=0.0)
    torch.testing.assert_close(parameter, expected_parameter, rtol=1.0e-6, atol=2.0e-7)

    decay_only = nn.Parameter(old.clone())
    decay_optimizer = Muon(
        [decay_only],
        lr=3.0e-4,
        momentum=0.95,
        weight_decay=0.1,
        update_scale=0.18,
    )
    decay_only.grad = torch.zeros_like(decay_only)
    decay_optimizer.step()
    torch.testing.assert_close(decay_only, old * (1.0 - 3.0e-4 * 0.1), rtol=0.0, atol=0.0)


def test_packed_3d_parameter_runs_ns_per_expert() -> None:
    torch.manual_seed(7)
    parameter = nn.Parameter(torch.randn(3, 6, 4))
    gradient = torch.randn_like(parameter)
    before = parameter.detach().clone()
    optimizer = Muon(
        [parameter],
        lr=1.0e-3,
        momentum=0.0,
        ns_steps=10,
        ns_method="hybrid",
        weight_decay=0.0,
        update_scale=0.18,
    )
    parameter.grad = gradient.clone()
    expected_updates = torch.stack(
        [
            _scale_muon_update(
                _newton_schulz_orthogonalize(expert_gradient, steps=10, method="hybrid"),
                scale=0.18,
            )
            for expert_gradient in gradient.unbind(0)
        ]
    )
    optimizer.step()
    torch.testing.assert_close(parameter, before - 1.0e-3 * expected_updates, rtol=1.0e-6, atol=2.0e-7)
    assert optimizer.last_step_metrics["logical_matrix_count"] == 3
    flattened = _scale_muon_update(
        _newton_schulz_orthogonalize(gradient.view(gradient.shape[0], -1), steps=10, method="hybrid"),
        scale=0.18,
    ).view_as(gradient)
    assert not torch.allclose(expected_updates, flattened)


def test_report_parameter_groups_have_no_missing_duplicates_or_role_mismatches() -> None:
    model = ReportAlignedDeepSeekV4(tiny_parity_config())
    rows = parameter_group_rows(model, weight_decay=0.1)
    ids = [id(parameter) for parameter in model.parameters() if parameter.requires_grad]
    assert len(rows) == len(set(ids))
    assignments = {row["parameter_name"]: row["optimizer"] for row in rows}
    for name, optimizer_name in assignments.items():
        lowered = name.lower()
        forced_adamw = (
            "embed_tokens" in lowered
            or "lm_head" in lowered
            or "norm" in lowered
            or ".router." in lowered
            or len(next(parameter for parameter_name, parameter in model.named_parameters() if parameter_name == name).shape) < 2
        )
        assert optimizer_name == ("adamw" if forced_adamw else "muon"), name
    assert assignments["layers.0.ffn.experts.gate_up_proj"] == "muon"
    assert assignments["layers.0.ffn.experts.down_proj"] == "muon"


def test_reference_mapping_ignores_derived_hash_router_table() -> None:
    model = ReportAlignedDeepSeekV4(tiny_parity_config())
    table = torch.arange(64).remainder(4).unsqueeze(1)
    report = load_reference_weights(model, {"layers.0.ffn.gate.tid2eid": table})
    assert report.copied == []
    assert report.ignored_reference == ["layers.0.ffn.gate.tid2eid"]


def test_muon_aux_adamw_lr_eps_and_checkpoint_resume(tmp_path: Path) -> None:
    torch.manual_seed(19)
    model = nn.Sequential(nn.Linear(4, 4, bias=False), nn.LayerNorm(4))
    train_config = {
        "learning_rate": 3.0e-4,
        "min_lr": 1.5e-5,
        "weight_decay": 0.1,
        "warmup_steps": 2,
        "max_steps": 8,
        "optimizer": {
            "name": "muon",
            "momentum": 0.95,
            "nesterov": True,
            "betas": [0.9, 0.95],
            "eps": 1.0e-20,
            "newton_schulz": "hybrid",
            "newton_schulz_iterations": 10,
            "update_rms_target": 0.18,
        },
    }
    optimizer = build_optimizer(model, train_config)
    assert isinstance(optimizer, HybridOptimizer)
    assert {group["lr"] for group in optimizer.param_groups} == {3.0e-4}
    assert optimizer.adamw is not None
    assert {group["eps"] for group in optimizer.adamw.param_groups} == {1.0e-20}
    scheduler = WarmupCosineScheduler(optimizer, base_lr=3.0e-4, min_lr=1.5e-5, warmup_steps=2, total_steps=8)
    for parameter in model.parameters():
        parameter.grad = torch.randn_like(parameter)
    optimizer.step()
    scheduler.step()
    checkpoint_path = tmp_path / "resume.pt"
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=1,
        cumulative_tokens=64,
        data_state={"epoch": 0, "offset": 1},
        config={"test": True},
    )

    clone = copy.deepcopy(model)
    clone_optimizer = build_optimizer(clone, train_config)
    clone_scheduler = WarmupCosineScheduler(
        clone_optimizer, base_lr=3.0e-4, min_lr=1.5e-5, warmup_steps=2, total_steps=8
    )
    resumed = load_checkpoint(
        checkpoint_path,
        model=clone,
        optimizer=clone_optimizer,
        scheduler=clone_scheduler,
        device=torch.device("cpu"),
    )
    assert resumed == {"step": 1, "cumulative_tokens": 64, "data_state": {"epoch": 0, "offset": 1}}
    assert clone_scheduler.state_dict() == scheduler.state_dict()
    torch.manual_seed(23)
    gradients = [torch.randn_like(parameter) for parameter in model.parameters()]
    for parameter, clone_parameter, gradient in zip(model.parameters(), clone.parameters(), gradients):
        parameter.grad = gradient.clone()
        clone_parameter.grad = gradient.clone()
    optimizer.step()
    clone_optimizer.step()
    scheduler.step()
    clone_scheduler.step()
    for parameter, clone_parameter in zip(model.parameters(), clone.parameters()):
        torch.testing.assert_close(parameter, clone_parameter, rtol=0.0, atol=0.0)


def _recursive_differences(left: Any, right: Any, prefix: str = "") -> set[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        differences: set[str] = set()
        for key in left.keys() | right.keys():
            path = f"{prefix}.{key}" if prefix else key
            if key not in left or key not in right:
                differences.add(path)
            else:
                differences.update(_recursive_differences(left[key], right[key], path))
        return differences
    return set() if left == right else {prefix}


def test_b_and_c_differ_only_by_ns_method_and_output_path() -> None:
    adamw = load_config(EXPERIMENT / "configs" / "A_adamw.yaml")
    hybrid = load_config(EXPERIMENT / "configs" / "B_muon_hybrid.yaml")
    standard = load_config(EXPERIMENT / "configs" / "C_muon_standard.yaml")
    assert adamw["train"]["optimizer"]["eps"] == 1.0e-8
    assert hybrid["train"]["optimizer"]["eps"] == 1.0e-20
    assert _recursive_differences(hybrid, standard) == {
        "train.optimizer.newton_schulz",
        "train.output_dir",
    }


def test_memmap_dataset_produces_shifted_tokens(tmp_path: Path) -> None:
    tokens = list(range(17))
    (tmp_path / "train.bin").write_bytes(struct.pack(f"<{len(tokens)}I", *tokens))
    (tmp_path / "valid.bin").write_bytes(struct.pack(f"<{len(tokens)}I", *tokens))
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "dtype": "uint32",
                "vocab_size": 64,
                "train_file": "train.bin",
                "valid_file": "valid.bin",
                "train_tokens_written": len(tokens),
                "valid_tokens_written": len(tokens),
            }
        ),
        encoding="utf-8",
    )
    config = {
        "data_dir": str(tmp_path),
        "manifest_file": "manifest.json",
        "train_file": "train.bin",
        "valid_file": "valid.bin",
        "dtype": "uint32",
        "vocab_size": 64,
    }
    dataset = MemmapTokenDataset(config, "train", seq_len=4)
    sample = dataset[1]
    torch.testing.assert_close(sample["input_ids"], torch.tensor([4, 5, 6, 7]))
    torch.testing.assert_close(sample["labels"], torch.tensor([5, 6, 7, 8]))


@pytest.mark.skipif(not (REFERENCE / "src" / "modeling_v3.py").exists(), reason="read-only reference package unavailable")
def test_fixed_batch_forward_bf16_and_single_step_reference_parity() -> None:
    config = tiny_parity_config()
    reference_model_type, reference_optimizer_builder = import_reference(REFERENCE)
    fp32 = forward_parity(config, reference_model_type, torch.device("cpu"), bf16=False)
    assert fp32["weight_mapping"]["missing_local_parameters"] == []
    assert fp32["output_errors"]["logits"]["max_abs_error"] < 2.0e-5
    assert fp32["output_errors"]["loss"]["max_abs_error"] < 2.0e-6
    bf16 = forward_parity(config, reference_model_type, torch.device("cpu"), bf16=True)
    assert math.isfinite(bf16["output_errors"]["logits"]["max_abs_error"])
    assert bf16["output_errors"]["logits"]["max_abs_error"] < 2.0e-2
    rows, gradient_error = training_parity(
        config,
        reference_model_type,
        reference_optimizer_builder,
        torch.device("cpu"),
        steps=1,
    )
    assert gradient_error["max_abs"] < 2.0e-5
    assert rows[0]["absolute_difference"] < 2.0e-6
    assert rows[0]["parameter_delta_max_abs_error"] < 2.0e-5
    assert abs(rows[0]["muon_update_rms"] - 0.18) < 5.0e-3
    assert abs(rows[0]["reference_muon_update_rms"] - 0.18) < 5.0e-3


def _ddp_worker(rank: int, world_size: int, port: int, queue: Any) -> None:
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world_size)
    torch.manual_seed(123)
    model = DDP(nn.Linear(7, 12, bias=False))
    optimizer = Muon(model.parameters(), lr=3.0e-4, momentum=0.95, update_scale=0.18)
    inputs = torch.randn(4, 7) + rank
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        loss = model(inputs).square().mean()
    loss.backward()
    optimizer.step()
    parameter = next(model.parameters()).detach()
    gathered = [torch.empty_like(parameter) for _ in range(world_size)]
    dist.all_gather(gathered, parameter)
    if rank == 0:
        queue.put(max(float((gathered[0] - item).abs().max().item()) for item in gathered[1:]))
    dist.destroy_process_group()


def test_two_rank_bf16_ddp_muon_update_is_rank_consistent() -> None:
    context = mp.get_context("spawn")
    queue = context.SimpleQueue()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])
    processes = [context.Process(target=_ddp_worker, args=(rank, 2, port, queue)) for rank in range(2)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=60)
    for process in processes:
        if process.is_alive():
            process.terminate()
            process.join()
    assert all(process.exitcode == 0 for process in processes)
    assert queue.get() == 0.0
