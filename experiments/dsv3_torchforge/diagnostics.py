from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

import torch
from torch import nn

from experiments.dsv3_reference.model import (
    DSV3ReferenceModel,
    ReferenceDenseFFN,
    ReferenceMLA,
    ReferenceMoE,
    ReferenceRMSNorm,
)
from experiments.dsv3_torchforge.model import DSV3TorchForgeModel, TorchForgeMLAWrapper, TorchForgeMoEWrapper
from torchforge.common.attention import MLA
from torchforge.common.moe import MoE
from torchforge.common.nn import FeedForward, RMSNorm


@dataclass
class _Pair:
    name: str
    reference: torch.Tensor
    target: torch.Tensor


def copy_reference_weights(
    reference_model: nn.Module,
    torchforge_model: nn.Module,
    *,
    strict: bool = True,
) -> dict[str, Any]:
    """Copy mappable reference weights into a TorchForge replacement model.

    The function is intentionally experiment-local. It maps the shared DSV3
    reference structure to the replacement structure and reports any component
    that cannot be represented as a direct parameter copy.
    """

    copied: list[str] = []
    unmapped: list[str] = []

    def copy_state(name: str, source: nn.Module, target: nn.Module) -> None:
        try:
            target.load_state_dict(source.state_dict())
        except Exception as exc:  # noqa: BLE001 - the report needs the original failure text.
            unmapped.append(f"{name}: failed to load state_dict: {exc}")
            return
        copied.append(name)

    def copy_tensor(name: str, source: torch.Tensor, target: torch.Tensor) -> None:
        if source.shape != target.shape:
            unmapped.append(f"{name}: shape mismatch {tuple(source.shape)} vs {tuple(target.shape)}")
            return
        target.data.copy_(source.data)
        copied.append(name)

    copy_state("embed_tokens", reference_model.embed_tokens, torchforge_model.embed_tokens)
    copy_state("norm", reference_model.norm, torchforge_model.norm)
    copy_state("lm_head", reference_model.lm_head, torchforge_model.lm_head)

    if len(reference_model.layers) != len(torchforge_model.layers):
        unmapped.append(
            f"layers: count mismatch {len(reference_model.layers)} vs {len(torchforge_model.layers)}"
        )
    else:
        for layer_idx, (reference_layer, target_layer) in enumerate(zip(reference_model.layers, torchforge_model.layers)):
            prefix = f"layers.{layer_idx}"
            _copy_norm(f"{prefix}.input_norm", reference_layer.input_norm, target_layer.input_norm, copy_state, unmapped)
            _copy_attention(
                f"{prefix}.self_attn",
                reference_layer.self_attn,
                target_layer.self_attn,
                copy_state,
                copy_tensor,
                unmapped,
            )
            _copy_norm(
                f"{prefix}.post_attention_norm",
                reference_layer.post_attention_norm,
                target_layer.post_attention_norm,
                copy_state,
                unmapped,
            )
            _copy_ffn(f"{prefix}.mlp", reference_layer.mlp, target_layer.mlp, copy_state, unmapped)

    if strict and unmapped:
        raise ValueError("Cannot copy all reference weights:\n" + "\n".join(f"- {item}" for item in unmapped))
    return {"copied": copied, "unmapped": unmapped, "strict": strict}


def collect_activations(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[dict[str, dict[str, Any]], Any]:
    records: dict[str, dict[str, Any]] = {}
    handles = []

    for name, module in _iter_diagnostic_modules(model):
        handles.append(module.register_forward_hook(_make_activation_hook(name, records)))
    try:
        outputs = model(input_ids, labels=labels)
    finally:
        for handle in handles:
            handle.remove()
    return records, outputs


def compare_activations(
    reference_activations: dict[str, dict[str, Any]],
    target_activations: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    records = []
    missing = []
    for name, reference in reference_activations.items():
        target = target_activations.get(name)
        if target is None:
            missing.append(name)
            continue
        records.append(
            {
                "module": name,
                "reference_shape": reference["shape"],
                "target_shape": target["shape"],
                "mean_abs_diff": abs(reference["mean"] - target["mean"]),
                "std_abs_diff": abs(reference["std"] - target["std"]),
                "min_abs_diff": abs(reference["min"] - target["min"]),
                "max_abs_diff": abs(reference["max"] - target["max"]),
            }
        )
    extra = [name for name in target_activations if name not in reference_activations]
    return {
        "records": records,
        "missing": missing,
        "extra": extra,
        "summary": _summarize_records(records),
    }


def compare_parameters(reference_model: nn.Module, target_model: nn.Module) -> dict[str, Any]:
    return _compare_pairs(_iter_parameter_pairs(reference_model, target_model), value_name="parameter")


def compare_gradients(reference_model: nn.Module, target_model: nn.Module) -> dict[str, Any]:
    pairs = []
    missing = []
    for name, reference_param, target_param in _iter_named_parameter_pairs(reference_model, target_model):
        if reference_param.grad is None or target_param.grad is None:
            missing.append(name)
            continue
        pairs.append(_Pair(name=name, reference=reference_param.grad, target=target_param.grad))
    report = _compare_pairs(pairs, value_name="gradient")
    report["missing_gradients"] = missing
    return report


def run_forward_backward_diagnostics(
    reference_model: nn.Module,
    target_model: nn.Module,
    *,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    copy_weights: bool = False,
    strict_copy: bool = True,
) -> dict[str, Any]:
    copy_report: Optional[dict[str, Any]] = None
    if copy_weights:
        copy_report = copy_reference_weights(reference_model, target_model, strict=strict_copy)

    reference_model.zero_grad(set_to_none=True)
    target_model.zero_grad(set_to_none=True)
    reference_activations, reference_outputs = collect_activations(reference_model, input_ids, labels)
    target_activations, target_outputs = collect_activations(target_model, input_ids, labels)

    reference_loss = reference_outputs["loss"]
    target_loss = target_outputs["loss"]
    if reference_loss is None or target_loss is None:
        raise RuntimeError("diagnostics require both models to return loss.")
    reference_loss.backward()
    target_loss.backward()

    return {
        "weight_copy": copy_report,
        "loss": {
            "reference": float(reference_loss.detach().cpu()),
            "target": float(target_loss.detach().cpu()),
            "abs_diff": float(abs(reference_loss.detach().cpu() - target_loss.detach().cpu())),
        },
        "activations": {
            "reference": reference_activations,
            "target": target_activations,
        },
        "activation_diff": compare_activations(reference_activations, target_activations),
        "parameter_diff": compare_parameters(reference_model, target_model),
        "gradient_diff": compare_gradients(reference_model, target_model),
    }


def _copy_norm(
    name: str,
    source: nn.Module,
    target: nn.Module,
    copy_state: Callable[[str, nn.Module, nn.Module], None],
    unmapped: list[str],
) -> None:
    if isinstance(source, ReferenceRMSNorm) and isinstance(target, (ReferenceRMSNorm, RMSNorm)):
        copy_state(name, source, target)
        return
    unmapped.append(f"{name}: unsupported norm mapping {type(source).__name__} -> {type(target).__name__}")


def _copy_attention(
    name: str,
    source: nn.Module,
    target: nn.Module,
    copy_state: Callable[[str, nn.Module, nn.Module], None],
    copy_tensor: Callable[[str, torch.Tensor, torch.Tensor], None],
    unmapped: list[str],
) -> None:
    if isinstance(source, ReferenceMLA) and isinstance(target, ReferenceMLA):
        copy_state(name, source, target)
        return
    if isinstance(source, ReferenceMLA) and isinstance(target, TorchForgeMLAWrapper):
        attention = target.attention
        if not isinstance(attention, MLA):
            unmapped.append(f"{name}: expected MLA inside TorchForgeMLAWrapper, got {type(attention).__name__}")
            return
        copy_state(f"{name}.q_a_proj", source.q_a_proj, attention.query_projection.q_a_proj)
        copy_state(f"{name}.q_b_proj", source.q_b_proj, attention.query_projection.q_b_proj)
        copy_tensor(f"{name}.q_a_norm_weight", source.q_a_norm_weight, attention.query_projection.q_a_norm_weight)
        copy_state(f"{name}.kv_a_proj_with_mqa", source.kv_a_proj_with_mqa, attention.kv_projection.kv_a_proj_with_mqa)
        copy_tensor(f"{name}.kv_a_norm_weight", source.kv_a_norm_weight, attention.kv_projection.kv_a_norm_weight)
        copy_state(f"{name}.kv_b_proj", source.kv_b_proj, attention.kv_projection.kv_b_proj)
        copy_state(f"{name}.o_proj", source.o_proj, attention.output_projection.o_proj)
        return
    unmapped.append(f"{name}: unsupported attention mapping {type(source).__name__} -> {type(target).__name__}")


def _copy_ffn(
    name: str,
    source: nn.Module,
    target: nn.Module,
    copy_state: Callable[[str, nn.Module, nn.Module], None],
    unmapped: list[str],
) -> None:
    if isinstance(source, ReferenceDenseFFN) and isinstance(target, ReferenceDenseFFN):
        copy_state(name, source, target)
        return
    if isinstance(source, ReferenceDenseFFN) and isinstance(target, FeedForward):
        copy_state(f"{name}.up_proj", source.up_proj, target.up_proj)
        copy_state(f"{name}.down_proj", source.down_proj, target.down_proj)
        return
    if isinstance(source, ReferenceMoE) and isinstance(target, TorchForgeMoEWrapper):
        _copy_moe(name, source, target.moe, copy_state, unmapped)
        return
    unmapped.append(f"{name}: unsupported FFN mapping {type(source).__name__} -> {type(target).__name__}")


def _copy_moe(
    name: str,
    source: ReferenceMoE,
    target: MoE,
    copy_state: Callable[[str, nn.Module, nn.Module], None],
    unmapped: list[str],
) -> None:
    if source.num_experts != target.num_experts:
        unmapped.append(f"{name}: num_experts mismatch {source.num_experts} vs {target.num_experts}")
        return
    if source.top_k != target.top_k:
        unmapped.append(f"{name}: top_k mismatch {source.top_k} vs {target.top_k}")
        return
    copy_state(f"{name}.router.proj", source.router.proj, target.router.proj)
    for expert_idx, (source_expert, target_expert) in enumerate(zip(source.experts, target.experts)):
        copy_state(f"{name}.experts.{expert_idx}.up_proj", source_expert.up_proj, target_expert.up_proj)
        copy_state(f"{name}.experts.{expert_idx}.gate_proj", source_expert.gate_proj, target_expert.gate_proj)
        copy_state(f"{name}.experts.{expert_idx}.down_proj", source_expert.down_proj, target_expert.down_proj)


def _make_activation_hook(name: str, records: dict[str, dict[str, Any]]) -> Callable[..., None]:
    def hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
        tensor = _first_tensor(output)
        if tensor is None:
            return
        detached = tensor.detach().float()
        records[name] = {
            "module": name,
            "shape": list(tensor.shape),
            "mean": float(detached.mean().cpu()),
            "std": float(detached.std(unbiased=False).cpu()),
            "min": float(detached.min().cpu()),
            "max": float(detached.max().cpu()),
        }

    return hook


def _first_tensor(value: Any) -> Optional[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, dict):
        for item in value.values():
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _iter_diagnostic_modules(model: nn.Module) -> Iterable[tuple[str, nn.Module]]:
    for name, module in model.named_modules():
        if not name:
            continue
        if _is_diagnostic_module_name(name):
            yield name, module


def _is_diagnostic_module_name(name: str) -> bool:
    if name in {"embed_tokens", "norm", "lm_head"}:
        return True
    parts = name.split(".")
    return len(parts) == 3 and parts[0] == "layers" and parts[2] in {
        "input_norm",
        "self_attn",
        "post_attention_norm",
        "mlp",
    }


def _iter_parameter_pairs(reference_model: nn.Module, target_model: nn.Module) -> Iterable[_Pair]:
    for name, reference_param, target_param in _iter_named_parameter_pairs(reference_model, target_model):
        yield _Pair(name=name, reference=reference_param, target=target_param)


def _iter_named_parameter_pairs(reference_model: nn.Module, target_model: nn.Module) -> Iterable[tuple[str, nn.Parameter, nn.Parameter]]:
    yield "embed_tokens.weight", reference_model.embed_tokens.weight, target_model.embed_tokens.weight
    yield "norm.weight", reference_model.norm.weight, target_model.norm.weight
    yield "lm_head.weight", reference_model.lm_head.weight, target_model.lm_head.weight

    for layer_idx, (reference_layer, target_layer) in enumerate(zip(reference_model.layers, target_model.layers)):
        prefix = f"layers.{layer_idx}"
        yield f"{prefix}.input_norm.weight", reference_layer.input_norm.weight, target_layer.input_norm.weight
        yield (
            f"{prefix}.post_attention_norm.weight",
            reference_layer.post_attention_norm.weight,
            target_layer.post_attention_norm.weight,
        )
        yield from _iter_attention_parameter_pairs(f"{prefix}.self_attn", reference_layer.self_attn, target_layer.self_attn)
        yield from _iter_ffn_parameter_pairs(f"{prefix}.mlp", reference_layer.mlp, target_layer.mlp)


def _iter_attention_parameter_pairs(prefix: str, source: nn.Module, target: nn.Module) -> Iterable[tuple[str, nn.Parameter, nn.Parameter]]:
    if isinstance(source, ReferenceMLA) and isinstance(target, ReferenceMLA):
        source_params = dict(source.named_parameters())
        target_params = dict(target.named_parameters())
        for name, source_param in source_params.items():
            if name in target_params:
                yield f"{prefix}.{name}", source_param, target_params[name]
        return
    if isinstance(source, ReferenceMLA) and isinstance(target, TorchForgeMLAWrapper):
        attention = target.attention
        yield f"{prefix}.q_a_proj.weight", source.q_a_proj.weight, attention.query_projection.q_a_proj.weight
        if source.q_a_proj.bias is not None:
            yield f"{prefix}.q_a_proj.bias", source.q_a_proj.bias, attention.query_projection.q_a_proj.bias
        yield f"{prefix}.q_b_proj.weight", source.q_b_proj.weight, attention.query_projection.q_b_proj.weight
        yield f"{prefix}.q_a_norm_weight", source.q_a_norm_weight, attention.query_projection.q_a_norm_weight
        yield f"{prefix}.kv_a_proj_with_mqa.weight", source.kv_a_proj_with_mqa.weight, attention.kv_projection.kv_a_proj_with_mqa.weight
        if source.kv_a_proj_with_mqa.bias is not None:
            yield f"{prefix}.kv_a_proj_with_mqa.bias", source.kv_a_proj_with_mqa.bias, attention.kv_projection.kv_a_proj_with_mqa.bias
        yield f"{prefix}.kv_a_norm_weight", source.kv_a_norm_weight, attention.kv_projection.kv_a_norm_weight
        yield f"{prefix}.kv_b_proj.weight", source.kv_b_proj.weight, attention.kv_projection.kv_b_proj.weight
        yield f"{prefix}.o_proj.weight", source.o_proj.weight, attention.output_projection.o_proj.weight
        if source.o_proj.bias is not None:
            yield f"{prefix}.o_proj.bias", source.o_proj.bias, attention.output_projection.o_proj.bias


def _iter_ffn_parameter_pairs(prefix: str, source: nn.Module, target: nn.Module) -> Iterable[tuple[str, nn.Parameter, nn.Parameter]]:
    if isinstance(source, ReferenceDenseFFN) and isinstance(target, ReferenceDenseFFN):
        yield f"{prefix}.up_proj.weight", source.up_proj.weight, target.up_proj.weight
        yield f"{prefix}.down_proj.weight", source.down_proj.weight, target.down_proj.weight
        return
    if isinstance(source, ReferenceDenseFFN) and isinstance(target, FeedForward):
        yield f"{prefix}.up_proj.weight", source.up_proj.weight, target.up_proj.weight
        yield f"{prefix}.down_proj.weight", source.down_proj.weight, target.down_proj.weight
        return
    if isinstance(source, ReferenceMoE) and isinstance(target, TorchForgeMoEWrapper):
        yield f"{prefix}.router.proj.weight", source.router.proj.weight, target.moe.router.proj.weight
        for expert_idx, (source_expert, target_expert) in enumerate(zip(source.experts, target.moe.experts)):
            yield f"{prefix}.experts.{expert_idx}.up_proj.weight", source_expert.up_proj.weight, target_expert.up_proj.weight
            yield (
                f"{prefix}.experts.{expert_idx}.gate_proj.weight",
                source_expert.gate_proj.weight,
                target_expert.gate_proj.weight,
            )
            yield (
                f"{prefix}.experts.{expert_idx}.down_proj.weight",
                source_expert.down_proj.weight,
                target_expert.down_proj.weight,
            )


def _compare_pairs(pairs: Iterable[_Pair], *, value_name: str) -> dict[str, Any]:
    records = []
    for pair in pairs:
        if pair.reference.shape != pair.target.shape:
            records.append(
                {
                    "name": pair.name,
                    "shape_mismatch": [list(pair.reference.shape), list(pair.target.shape)],
                }
            )
            continue
        diff = (pair.reference.detach().float() - pair.target.detach().float()).abs()
        reference_norm = pair.reference.detach().float().norm()
        target_norm = pair.target.detach().float().norm()
        records.append(
            {
                "name": pair.name,
                f"{value_name}_max_abs_diff": float(diff.max().cpu()),
                f"{value_name}_mean_abs_diff": float(diff.mean().cpu()),
                "reference_norm": float(reference_norm.cpu()),
                "target_norm": float(target_norm.cpu()),
            }
        )
    return {"records": records, "summary": _summarize_records(records)}


def _summarize_records(records: list[dict[str, Any]]) -> dict[str, float]:
    max_values = []
    mean_values = []
    for record in records:
        for key, value in record.items():
            if key.endswith("max_abs_diff") and isinstance(value, (int, float)):
                max_values.append(float(value))
            if key.endswith("mean_abs_diff") and isinstance(value, (int, float)):
                mean_values.append(float(value))
    return {
        "max_abs_diff": max(max_values) if max_values else 0.0,
        "mean_abs_diff": sum(mean_values) / len(mean_values) if mean_values else 0.0,
    }


__all__ = [
    "collect_activations",
    "compare_activations",
    "compare_gradients",
    "compare_parameters",
    "copy_reference_weights",
    "run_forward_backward_diagnostics",
]

