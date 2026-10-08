from __future__ import annotations

import math
from typing import Any, Iterable

import torch
from torch import nn

from .adamw import AdamW
from .muon import Muon


def sinkhorn_balance_update(
    update: torch.Tensor, *, steps: int = 11, tau: float = 1e-3, eps: float = 1e-20,
) -> torch.Tensor:
    """按照报告 Algorithm 1 交替归一化矩阵的行与列。"""
    if update.ndim != 2 or not update.is_floating_point() or min(update.shape) == 0:
        raise ValueError("Sinkhorn update must be a nonempty floating-point matrix.")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0 or steps % 2 != 1:
        raise ValueError("Sinkhorn steps must be a positive odd integer.")
    if not math.isfinite(tau) or tau < 0 or not math.isfinite(eps) or eps <= 0:
        raise ValueError("Sinkhorn requires finite tau >= 0 and eps > 0.")
    if not torch.isfinite(update).all():
        raise FloatingPointError("Sinkhorn update contains NaN or Inf.")
    work = update.float()
    row_norms = torch.linalg.vector_norm(work, dim=1, keepdim=True)
    work = work.masked_fill(row_norms <= tau * row_norms.mean(), 0)
    for step in range(steps):
        dimension = 1 if step % 2 == 0 else 0
        work = work / (torch.linalg.vector_norm(work, dim=dimension, keepdim=True) + eps)
    return work * math.sqrt(update.shape[1])


class SinkhornMomentum(torch.optim.Optimizer):
    """V4.1 embedding 与 prediction head 的 Nesterov、Sinkhorn 更新。"""

    def __init__(
        self, params: Iterable[Any], *, lr: float = 2.6e-4, momentum: float = 0.95,
        update_scale: float = 0.18, steps: int = 11, tau: float = 1e-3, eps: float = 1e-20,
    ) -> None:
        if not math.isfinite(lr) or lr <= 0 or not 0 <= momentum < 1:
            raise ValueError("lr must be positive and momentum must be in [0, 1).")
        if not math.isfinite(update_scale) or update_scale <= 0:
            raise ValueError("update_scale must be finite and positive.")
        if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0 or steps % 2 != 1:
            raise ValueError("steps must be a positive odd integer.")
        if not math.isfinite(tau) or tau < 0 or not math.isfinite(eps) or eps <= 0:
            raise ValueError("Sinkhorn requires finite tau >= 0 and eps > 0.")
        super().__init__(params, dict(lr=lr, momentum=momentum, update_scale=update_scale,
                                     steps=steps, tau=tau, eps=eps, weight_decay=0.0))
        for group in self.param_groups:
            if group["weight_decay"] != 0:
                raise ValueError("V4.1 SinkhornMomentum requires weight_decay=0.")
            if any(param.ndim != 2 for param in group["params"]):
                raise ValueError("SinkhornMomentum parameters must be matrices.")

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.grad.is_sparse:
                    raise ValueError("SinkhornMomentum requires dense gradients.")
                gradient = param.grad.float()
                if not torch.isfinite(gradient).all():
                    raise FloatingPointError("SinkhornMomentum gradient contains NaN or Inf.")
                state = self.state[param]
                if not state:
                    state["momentum_buffer"] = torch.zeros_like(param, dtype=torch.float32)
                beta = group["momentum"]
                buffer = state["momentum_buffer"]
                buffer.mul_(beta).add_(gradient, alpha=1 - beta)
                nesterov = buffer.mul(beta).add(gradient, alpha=1 - beta)
                update = sinkhorn_balance_update(nesterov, steps=group["steps"], tau=group["tau"], eps=group["eps"])
                param.add_(update.to(param.dtype), alpha=-group["lr"] * group["update_scale"])
        return loss


class HeadwiseMuon(Muon):
    """head_shape 参数组将 Q/K 权重按 head 分别计算 Muon。"""


class DeepSeekV41Optimizer:
    """管理 Muon、SinkhornMomentum、AdamW 的更新与 checkpoint。"""

    def __init__(self, *, muon: HeadwiseMuon | None, sinkhorn: SinkhornMomentum | None,
                 adamw: AdamW | None) -> None:
        self.muon = muon
        self.sinkhorn = sinkhorn
        self.adamw = adamw
        self.optimizers = {name: optimizer for name, optimizer in
                           (("muon", muon), ("sinkhorn", sinkhorn), ("adamw", adamw)) if optimizer is not None}
        if not self.optimizers:
            raise ValueError("At least one optimizer is required.")

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return [group for optimizer in self.optimizers.values() for group in optimizer.param_groups]

    def zero_grad(self, *, set_to_none: bool = True) -> None:
        for optimizer in self.optimizers.values():
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        for optimizer in self.optimizers.values():
            optimizer.step()

    def state_dict(self) -> dict[str, Any]:
        return {name: optimizer.state_dict() for name, optimizer in self.optimizers.items()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if set(state) != set(self.optimizers):
            raise ValueError("Checkpoint optimizer families do not match the model.")
        for name, optimizer in self.optimizers.items():
            optimizer.load_state_dict(state[name])


def _head_shapes(module: nn.Module) -> dict[int, tuple[int, int, int]]:
    shapes = {}
    for child in module.modules():
        explicit = getattr(child, "muon_head_shape", None)
        if explicit is not None:
            shapes[id(child.weight)] = tuple(explicit)
        if type(child).__name__ in {"CSA2Attention", "SlidingWindowAttention"}:
            shapes[id(child.q_proj.weight)] = (child.num_attention_heads, child.head_dim, child.q_lora_rank)
        if type(child).__name__ == "HierarchicalSparseIndexer":
            shapes[id(child.indexer_q_proj.weight)] = (child.index_num_heads, child.index_head_dim, child.q_lora_rank)
    return shapes


def build_deepseek_v41_param_groups(
    module: nn.Module, *, lr: float = 2.6e-4, weight_decay: float = 0.1,
    engram_lr_scale: float = 5.0,
) -> dict[str, list[dict[str, Any]]]:
    """根据参数用途分配优化器；所有可训练参数恰好分配一次。"""
    if not isinstance(module, nn.Module):
        raise TypeError("module must be an nn.Module.")
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and nonnegative.")
    if not math.isfinite(lr) or lr <= 0 or not math.isfinite(engram_lr_scale) or engram_lr_scale <= 0:
        raise ValueError("lr and engram_lr_scale must be finite and positive.")
    sinkhorn_roles: dict[int, str] = {}
    norm_parameters: set[int] = set()
    for name, child in module.named_modules():
        role = getattr(child, "optimizer_role", None)
        if role in {"engram_embedding", "token_embedding", "prediction_head"}:
            for param in child.parameters(recurse=False):
                if param.ndim == 2:
                    sinkhorn_roles[id(param)] = role
        if isinstance(child, nn.Embedding):
            sinkhorn_roles[id(child.weight)] = role or "token_embedding"
        if type(child).__name__ == "LMHead" or name.rsplit(".", 1)[-1] in {"lm_head", "prediction_head"}:
            for param in child.parameters():
                if param.ndim == 2:
                    sinkhorn_roles[id(param)] = "prediction_head"
        if isinstance(child, nn.LayerNorm) or type(child).__name__ in {"RMSNorm", "UnweightedRMSNorm"}:
            for param_name, param in child.named_parameters(recurse=False):
                if param_name == "weight":
                    norm_parameters.add(id(param))
        for param_name, param in child.named_parameters(recurse=False):
            if param_name.endswith("norm_weight") or param_name in getattr(child, "optimizer_norm_parameter_names", ()):
                norm_parameters.add(id(param))
    shapes = _head_shapes(module)
    groups: dict[str, list[dict[str, Any]]] = {"muon": [], "sinkhorn": [], "adamw": []}
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        entry = {"params": [param], "lr": lr, "parameter_names": [name]}
        if id(param) in sinkhorn_roles:
            entry["weight_decay"] = 0.0
            if sinkhorn_roles[id(param)] == "engram_embedding":
                entry["lr"] = lr * engram_lr_scale
            groups["sinkhorn"].append(entry)
        elif id(param) in norm_parameters or param.ndim < 2 or name.rsplit(".", 1)[-1].endswith("bias"):
            entry["weight_decay"] = weight_decay if id(param) in norm_parameters else 0.0
            groups["adamw"].append(entry)
        else:
            if param.ndim not in {2, 3}:
                raise ValueError(f"Matrix parameter {name} requires an explicit 2D/3D representation.")
            entry["weight_decay"] = weight_decay
            if id(param) in shapes:
                entry["head_shape"] = shapes[id(param)]
            groups["muon"].append(entry)
    if not any(groups.values()):
        raise ValueError("The model has no trainable parameters.")
    return groups


def build_deepseek_v41_optimizer(
    module: nn.Module, *, lr: float = 2.6e-4, weight_decay: float = 0.1,
    engram_lr_scale: float = 5.0,
) -> DeepSeekV41Optimizer:
    groups = build_deepseek_v41_param_groups(module, lr=lr, weight_decay=weight_decay,
                                            engram_lr_scale=engram_lr_scale)
    return DeepSeekV41Optimizer(
        muon=HeadwiseMuon(groups["muon"], lr=lr, momentum=0.95, weight_decay=weight_decay,
                          ns_steps=10, ns_method="hybrid", update_scale=0.18) if groups["muon"] else None,
        sinkhorn=SinkhornMomentum(groups["sinkhorn"], lr=lr, momentum=0.95,
                                  steps=11, tau=1e-3, eps=1e-20, update_scale=0.18) if groups["sinkhorn"] else None,
        adamw=AdamW(groups["adamw"], lr=lr, betas=(0.9, 0.95), eps=1e-20,
                    weight_decay=weight_decay) if groups["adamw"] else None,
    )


__all__ = ["sinkhorn_balance_update", "SinkhornMomentum", "HeadwiseMuon", "DeepSeekV41Optimizer",
           "build_deepseek_v41_param_groups", "build_deepseek_v41_optimizer"]
