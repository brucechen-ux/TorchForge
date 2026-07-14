from __future__ import annotations

import argparse
import copy
import csv
import importlib
import json
from pathlib import Path
import sys
from typing import Any, Callable

import torch

from .config import tiny_parity_config
from .model import ReportAlignedDeepSeekV4, WeightMappingReport, load_reference_weights
from .optim import WarmupCosineScheduler, build_optimizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic reference-to-TorchForge parity checks.")
    parser.add_argument(
        "--reference-root",
        default=r"D:\infra-project\deepseek_v4_muon_report_aligned_package_20260713",
    )
    parser.add_argument("--output-dir", default="experiments/dsv4_muon_report_aligned")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=3)
    return parser.parse_args()


def reference_config(config: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(config)
    result["v4_attention"]["_attn_implementation"] = result["v4_attention"].pop("attention_implementation")
    result["moe"].update(
        enabled=True,
        normalize_topk_prob=True,
        implementation="torch",
        moe_ep_size=1,
        moe_capacity_factor=1.25,
        moe_eval_capacity_factor=2.0,
        moe_min_capacity=4,
        moe_drop_tokens=False,
        moe_use_tutel=False,
    )
    return result


def import_reference(reference_root: str | Path) -> tuple[type[torch.nn.Module], Callable[..., Any]]:
    root = Path(reference_root).resolve()
    if not (root / "src" / "modeling_v3.py").exists():
        raise FileNotFoundError(f"Reference package not found at {root}.")
    sys.path.insert(0, str(root))
    try:
        modeling = importlib.import_module("src.modeling_v3")
        reference_muon = importlib.import_module("src.muon")
    finally:
        sys.path.pop(0)

    def build_reference_optimizer(model: torch.nn.Module, train_config: dict[str, Any], _deepspeed: bool) -> Any:
        optimizer_config = train_config["optimizer"]
        named_parameters = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
        muon_parameters = []
        adamw_parameters = []
        for name, parameter in named_parameters:
            lowered = name.lower()
            use_adamw = (
                "embed_tokens" in lowered
                or "lm_head" in lowered
                or "norm" in lowered
                or ".router." in lowered
                or ".ffn.gate." in lowered
                or parameter.ndim not in {2, 3}
            )
            (adamw_parameters if use_adamw else muon_parameters).append(parameter)
        groups = [
            {"params": muon_parameters, "name": "muon-matrices", "use_muon": True},
            {"params": adamw_parameters, "name": "adamw-aux", "use_muon": False},
        ]
        groups = [group for group in groups if group["params"]]
        optimizer = reference_muon.MuonWithAuxAdamW(
            groups,
            lr=float(train_config["learning_rate"]),
            weight_decay=float(train_config["weight_decay"]),
            momentum=float(optimizer_config["momentum"]),
            nesterov=bool(optimizer_config["nesterov"]),
            ns_method=str(optimizer_config["newton_schulz"]),
            ns_iterations=int(optimizer_config["newton_schulz_iterations"]),
            ns_first_stage_steps=8,
            ns_second_stage_steps=2,
            ns_standard_steps=10,
            update_rms_target=float(optimizer_config["update_rms_target"]),
            adamw_betas=tuple(optimizer_config["betas"]),
            adamw_eps=float(optimizer_config["eps"]),
            diagnostics_max_matrices=0,
        )
        optimizer.parameter_names = {id(parameter): name for name, parameter in named_parameters}
        return optimizer

    return modeling.DeepSeekV3LikeLM, build_reference_optimizer


def fixed_batches(config: dict[str, Any], *, count: int, seed: int = 2026) -> list[dict[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    batch_size = int(config["train"]["micro_batch_size"])
    seq_len = int(config["train"]["seq_len"])
    vocab_size = int(config["model"]["vocab_size"])
    result = []
    for _ in range(count):
        tokens = torch.randint(0, vocab_size, (batch_size, seq_len + 1), generator=generator)
        result.append({"input_ids": tokens[:, :-1], "labels": tokens[:, 1:]})
    return result


def error_metrics(actual: torch.Tensor, reference: torch.Tensor) -> tuple[float, float]:
    actual_float, reference_float = actual.detach().float(), reference.detach().float()
    maximum = float((actual_float - reference_float).abs().max().item())
    denominator = float(reference_float.abs().max().clamp_min(1.0e-30).item())
    return maximum, maximum / denominator


def _extract_tensor(value: Any) -> torch.Tensor | None:
    if torch.is_tensor(value):
        return value
    if isinstance(value, dict):
        for key in ("hidden_states", "logits", "loss"):
            if torch.is_tensor(value.get(key)):
                return value[key]
    if isinstance(value, (tuple, list)):
        for item in value:
            candidate = _extract_tensor(item)
            if candidate is not None:
                return candidate
    return None


def capture_modules(model: torch.nn.Module, names: list[str]) -> tuple[dict[str, torch.Tensor], list[Any]]:
    captures: dict[str, torch.Tensor] = {}
    modules = dict(model.named_modules())
    handles = []
    for name in names:
        if name not in modules:
            continue

        def hook(_module: torch.nn.Module, _inputs: Any, output: Any, *, capture_name: str = name) -> None:
            tensor = _extract_tensor(output)
            if tensor is not None:
                captures[capture_name] = tensor.detach().float().cpu()

        handles.append(modules[name].register_forward_hook(hook))
    return captures, handles


def module_pairs(num_layers: int) -> list[tuple[str, str]]:
    pairs = [("embed_tokens", "embed_tokens")]
    for index in range(num_layers):
        pairs.extend(
            [
                (f"layers.{index}.attn", f"layers.{index}.self_attn"),
                (f"layers.{index}.ffn", f"layers.{index}.ffn"),
                (f"layers.{index}", f"layers.{index}"),
            ]
        )
    pairs.extend(
        [
            ("final_norm", "final_norm"),
            ("lm_head", "lm_head"),
            ("mtp_modules.0.block.attn", "mtp.transformer_block.layer.self_attn"),
            ("mtp_modules.0.block.ffn", "mtp.transformer_block.layer.ffn"),
            ("mtp_modules.0.block", "mtp.transformer_block.layer"),
        ]
    )
    return pairs


def mapped_gradient_errors(
    reference_model: torch.nn.Module,
    local_model: torch.nn.Module,
    mapping: WeightMappingReport,
) -> dict[str, float | str | None]:
    reference_parameters = dict(reference_model.named_parameters())
    local_parameters = dict(local_model.named_parameters())
    max_abs = 0.0
    max_rel = 0.0
    first_difference: str | None = None
    for reference_name, local_name in mapping.copied:
        if reference_name not in reference_parameters or local_name not in local_parameters:
            continue
        reference_grad = reference_parameters[reference_name].grad
        local_grad = local_parameters[local_name].grad
        if reference_grad is None and local_grad is None:
            continue
        if reference_grad is None or local_grad is None:
            return {"max_abs": float("inf"), "max_rel": float("inf"), "first_difference": reference_name}
        if reference_name == "mtp_modules.0.eh_proj.weight":
            half = reference_grad.shape[1] // 2
            reference_grad = torch.cat([reference_grad[:, half:], reference_grad[:, :half]], dim=1)
        absolute, relative = error_metrics(local_grad, reference_grad)
        if absolute > max_abs:
            max_abs, max_rel = absolute, relative
        if first_difference is None and absolute > 2.0e-6:
            first_difference = reference_name
    return {"max_abs": max_abs, "max_rel": max_rel, "first_difference": first_difference}


def _global_grad_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.detach().float().square().sum().item())
    return total**0.5


def mapped_delta_errors(
    reference_model: torch.nn.Module,
    local_model: torch.nn.Module,
    mapping: WeightMappingReport,
    reference_before: dict[str, torch.Tensor],
    local_before: dict[str, torch.Tensor],
) -> dict[str, float | str | None]:
    reference_parameters = dict(reference_model.named_parameters())
    local_parameters = dict(local_model.named_parameters())
    max_abs = 0.0
    max_rel = 0.0
    first_difference: str | None = None
    for reference_name, local_name in mapping.copied:
        if reference_name not in reference_parameters or local_name not in local_parameters:
            continue
        reference_delta = reference_parameters[reference_name].detach() - reference_before[reference_name]
        local_delta = local_parameters[local_name].detach() - local_before[local_name]
        if reference_name == "mtp_modules.0.eh_proj.weight":
            half = reference_delta.shape[1] // 2
            reference_delta = torch.cat([reference_delta[:, half:], reference_delta[:, :half]], dim=1)
        absolute, relative = error_metrics(local_delta, reference_delta)
        if absolute > max_abs:
            max_abs, max_rel = absolute, relative
        if first_difference is None and absolute > 2.0e-6:
            first_difference = reference_name
    return {"max_abs": max_abs, "max_rel": max_rel, "first_difference": first_difference}


def muon_update_rms_from_delta(
    optimizer: Any,
    before: dict[int, torch.Tensor],
    *,
    lr: float,
    weight_decay: float,
) -> float:
    total = 0.0
    count = 0
    groups = optimizer.param_groups
    for group in groups:
        use_muon = bool(group.get("use_muon", False)) or "ns_method" in group
        if not use_muon:
            continue
        for parameter in group["params"]:
            old = before[id(parameter)]
            update = (old * (1.0 - lr * weight_decay) - parameter.detach()) / lr
            total += float(update.float().square().sum().item())
            count += update.numel()
    return (total / count) ** 0.5 if count else 0.0


def build_models(
    config: dict[str, Any],
    reference_model_type: type[torch.nn.Module],
    device: torch.device,
) -> tuple[torch.nn.Module, ReportAlignedDeepSeekV4, WeightMappingReport]:
    torch.manual_seed(int(config["seed"]))
    reference_model = reference_model_type(reference_config(config)).to(device)
    torch.manual_seed(int(config["seed"]))
    local_model = ReportAlignedDeepSeekV4(config).to(device)
    mapping = load_reference_weights(local_model, reference_model.state_dict())
    return reference_model, local_model, mapping


def forward_parity(
    config: dict[str, Any],
    reference_model_type: type[torch.nn.Module],
    device: torch.device,
    *,
    bf16: bool,
) -> dict[str, Any]:
    reference_model, local_model, mapping = build_models(config, reference_model_type, device)
    reference_model.eval()
    local_model.eval()
    batch = {key: value.to(device) for key, value in fixed_batches(config, count=1)[0].items()}
    pairs = module_pairs(int(config["model"]["num_layers"]))
    reference_captures, reference_handles = capture_modules(reference_model, [pair[0] for pair in pairs])
    local_captures, local_handles = capture_modules(local_model, [pair[1] for pair in pairs])
    enabled = bf16 and (device.type == "cpu" or torch.cuda.is_bf16_supported())
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled):
        reference_output = reference_model(**batch)
        local_output = local_model(**batch)
    for handle in [*reference_handles, *local_handles]:
        handle.remove()
    module_errors = []
    first_difference = None
    for reference_name, local_name in pairs:
        if reference_name not in reference_captures or local_name not in local_captures:
            continue
        absolute, relative = error_metrics(local_captures[local_name], reference_captures[reference_name])
        module_errors.append(
            {
                "reference_module": reference_name,
                "torchforge_module": local_name,
                "max_abs_error": absolute,
                "max_relative_error": relative,
            }
        )
        if first_difference is None and absolute > (2.0e-2 if bf16 else 2.0e-6):
            first_difference = reference_name
    output_errors = {}
    for key in ("logits", "loss", "lm_loss", "mtp_loss", "aux_loss"):
        absolute, relative = error_metrics(local_output[key], reference_output[key])
        output_errors[key] = {"max_abs_error": absolute, "max_relative_error": relative}
    return {
        "dtype": "bf16" if bf16 else "fp32",
        "weight_mapping": {
            "copied": len(mapping.copied),
            "ignored_reference": mapping.ignored_reference,
            "missing_local_parameters": mapping.missing_local_parameters,
        },
        "module_errors": module_errors,
        "first_residual_difference": first_difference,
        "output_errors": output_errors,
    }


def training_parity(
    config: dict[str, Any],
    reference_model_type: type[torch.nn.Module],
    reference_optimizer_builder: Callable[..., Any],
    device: torch.device,
    steps: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reference_model, local_model, mapping = build_models(config, reference_model_type, device)
    reference_model.train()
    local_model.train()
    reference_config_value = reference_config(config)
    reference_optimizer = reference_optimizer_builder(reference_model, reference_config_value["train"], False)
    local_optimizer = build_optimizer(local_model, config["train"])
    reference_scheduler = WarmupCosineScheduler(
        reference_optimizer,
        base_lr=float(config["train"]["learning_rate"]),
        min_lr=float(config["train"]["min_lr"]),
        warmup_steps=int(config["train"]["warmup_steps"]),
        total_steps=steps,
    )
    local_scheduler = WarmupCosineScheduler(
        local_optimizer,
        base_lr=float(config["train"]["learning_rate"]),
        min_lr=float(config["train"]["min_lr"]),
        warmup_steps=int(config["train"]["warmup_steps"]),
        total_steps=steps,
    )
    batches = fixed_batches(config, count=steps + 1)
    rows = []
    final_gradient_error: dict[str, Any] = {}
    for index in range(steps):
        batch = {key: value.to(device) for key, value in batches[index].items()}
        reference_optimizer.zero_grad(set_to_none=True)
        local_optimizer.zero_grad(set_to_none=True)
        reference_output = reference_model(**batch)
        local_output = local_model(**batch)
        reference_output["loss"].backward()
        local_output["loss"].backward()
        final_gradient_error = mapped_gradient_errors(reference_model, local_model, mapping)
        reference_grad_norm = float(torch.nn.utils.clip_grad_norm_(reference_model.parameters(), 1.0).item())
        local_grad_norm = float(torch.nn.utils.clip_grad_norm_(local_model.parameters(), 1.0).item())
        reference_grad_norm_after = _global_grad_norm(reference_model)
        local_grad_norm_after = _global_grad_norm(local_model)
        reference_lr = float(reference_optimizer.param_groups[0]["lr"])
        local_lr = float(local_optimizer.param_groups[0]["lr"])
        reference_parameter_before = {
            name: parameter.detach().clone() for name, parameter in reference_model.named_parameters()
        }
        local_parameter_before = {name: parameter.detach().clone() for name, parameter in local_model.named_parameters()}
        reference_by_id = {id(parameter): parameter.detach().clone() for parameter in reference_model.parameters()}
        local_by_id = {id(parameter): parameter.detach().clone() for parameter in local_model.parameters()}
        reference_optimizer.step()
        local_optimizer.step()
        delta_error = mapped_delta_errors(
            reference_model,
            local_model,
            mapping,
            reference_parameter_before,
            local_parameter_before,
        )
        reference_muon_rms = muon_update_rms_from_delta(
            reference_optimizer,
            reference_by_id,
            lr=reference_lr,
            weight_decay=float(config["train"]["weight_decay"]),
        )
        local_muon_rms = muon_update_rms_from_delta(
            local_optimizer,
            local_by_id,
            lr=local_lr,
            weight_decay=float(config["train"]["weight_decay"]),
        )
        reference_scheduler.step()
        local_scheduler.step()
        validation_batch = {key: value.to(device) for key, value in batches[index + 1].items()}
        reference_model.eval()
        local_model.eval()
        with torch.no_grad():
            reference_validation = reference_model(**validation_batch)["loss"]
            local_validation = local_model(**validation_batch)["loss"]
        reference_model.train()
        local_model.train()
        absolute = abs(float(local_output["loss"].item()) - float(reference_output["loss"].item()))
        denominator = max(abs(float(reference_output["loss"].item())), 1.0e-30)
        rows.append(
            {
                "status": "MEASURED",
                "step": index + 1,
                "cumulative_tokens": (index + 1)
                * int(config["train"]["micro_batch_size"])
                * int(config["train"]["seq_len"]),
                "lr": local_lr,
                "total_loss": float(local_output["loss"].item()),
                "lm_loss": float(local_output["lm_loss"].item()),
                "mtp_loss": float(local_output["mtp_loss"].item()),
                "aux_loss": float(local_output["aux_loss"].item()),
                "grad_norm": local_grad_norm,
                "grad_norm_after_clip": local_grad_norm_after,
                "muon_update_rms": local_muon_rms,
                "validation_loss": float(local_validation.item()),
                "reference_total_loss": float(reference_output["loss"].item()),
                "reference_lm_loss": float(reference_output["lm_loss"].item()),
                "reference_mtp_loss": float(reference_output["mtp_loss"].item()),
                "reference_aux_loss": float(reference_output["aux_loss"].item()),
                "reference_grad_norm": reference_grad_norm,
                "reference_grad_norm_after_clip": reference_grad_norm_after,
                "reference_muon_update_rms": reference_muon_rms,
                "absolute_difference": absolute,
                "relative_difference": absolute / denominator,
                "lr_absolute_difference": abs(local_lr - reference_lr),
                "parameter_delta_max_abs_error": delta_error["max_abs"],
                "parameter_delta_max_relative_error": delta_error["max_rel"],
                "first_parameter_delta_difference": delta_error["first_difference"],
            }
        )
    return rows, final_gradient_error


def write_results(output_dir: str | Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "loss_alignment_report.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "parity_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    report_path = output_dir / "loss_alignment_report.md"
    if report_path.exists():
        report = report_path.read_text(encoding="utf-8")
        start_marker = "<!-- PARITY_RESULTS_START -->"
        end_marker = "<!-- PARITY_RESULTS_END -->"
        measured_lines = [
            start_marker,
            "Status: **DETERMINISTIC SHORT PARITY MEASURED**.",
            "",
            "| dtype | logits max abs | logits max relative | total-loss max abs | first residual difference |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
        for dtype in ("fp32", "bf16"):
            result = summary[dtype]
            measured_lines.append(
                "| {dtype} | {logits_abs:.9g} | {logits_rel:.9g} | {loss_abs:.9g} | {first} |".format(
                    dtype=dtype.upper(),
                    logits_abs=result["output_errors"]["logits"]["max_abs_error"],
                    logits_rel=result["output_errors"]["logits"]["max_relative_error"],
                    loss_abs=result["output_errors"]["loss"]["max_abs_error"],
                    first=result["first_residual_difference"] or "none above fixed threshold",
                )
            )
        gradient = summary["last_step_gradient_error"]
        measured_lines.extend(
            [
                "",
                f"Last-step gradient max absolute error: `{gradient['max_abs']:.9g}`; "
                f"max relative error: `{gradient['max_rel']:.9g}`; first difference: "
                f"`{gradient['first_difference'] or 'none above fixed threshold'}`.",
                "",
                "| step | cumulative tokens | LR | total loss | reference total loss | abs diff | relative diff | Muon RMS | reference Muon RMS | delta max abs |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in rows:
            measured_lines.append(
                "| {step} | {tokens} | {lr:.9g} | {loss:.9g} | {reference:.9g} | {absolute:.9g} | "
                "{relative:.9g} | {rms:.9g} | {reference_rms:.9g} | {delta:.9g} |".format(
                    step=row["step"],
                    tokens=row["cumulative_tokens"],
                    lr=row["lr"],
                    loss=row["total_loss"],
                    reference=row["reference_total_loss"],
                    absolute=row["absolute_difference"],
                    relative=row["relative_difference"],
                    rms=row["muon_update_rms"],
                    reference_rms=row["reference_muon_update_rms"],
                    delta=row["parameter_delta_max_abs_error"],
                )
            )
        measured_lines.extend(
            [
                "",
                "These are deterministic tiny-shape parity measurements, not a 5B-token curve acceptance.",
                end_marker,
            ]
        )
        start = report.index(start_marker)
        end = report.index(end_marker) + len(end_marker)
        report_path.write_text(report[:start] + "\n".join(measured_lines) + report[end:], encoding="utf-8")


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    config = tiny_parity_config()
    reference_model_type, reference_optimizer_builder = import_reference(args.reference_root)
    fp32 = forward_parity(config, reference_model_type, device, bf16=False)
    bf16 = forward_parity(config, reference_model_type, device, bf16=True)
    rows, gradients = training_parity(
        config,
        reference_model_type,
        reference_optimizer_builder,
        device,
        args.steps,
    )
    summary = {"fp32": fp32, "bf16": bf16, "last_step_gradient_error": gradients}
    write_results(args.output_dir, rows, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
