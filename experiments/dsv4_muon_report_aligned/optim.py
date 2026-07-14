from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn

from torchforge.common.optim import AdamW, Muon, build_hybrid_optimizer_param_groups


class HybridOptimizer:
    """One checkpointable optimizer facade over Muon and auxiliary AdamW."""

    def __init__(self, muon: Muon | None, adamw: AdamW | None) -> None:
        if muon is None and adamw is None:
            raise ValueError("HybridOptimizer requires at least one optimizer.")
        self.muon = muon
        self.adamw = adamw

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        groups: list[dict[str, Any]] = []
        if self.muon is not None:
            groups.extend(self.muon.param_groups)
        if self.adamw is not None:
            groups.extend(self.adamw.param_groups)
        return groups

    @property
    def muon_update_rms(self) -> float:
        if self.muon is None:
            return 0.0
        return float(self.muon.last_step_metrics.get("muon_update_rms", 0.0))

    def step(self) -> None:
        if self.muon is not None:
            self.muon.step()
        if self.adamw is not None:
            self.adamw.step()

    def zero_grad(self, *, set_to_none: bool = True) -> None:
        if self.muon is not None:
            self.muon.zero_grad(set_to_none=set_to_none)
        if self.adamw is not None:
            self.adamw.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict[str, Any]:
        return {
            "kind": "muon_with_aux_adamw",
            "muon": self.muon.state_dict() if self.muon is not None else None,
            "adamw": self.adamw.state_dict() if self.adamw is not None else None,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if self.muon is not None:
            if state.get("muon") is None:
                raise ValueError("Checkpoint is missing Muon optimizer state.")
            self.muon.load_state_dict(state["muon"])
        if self.adamw is not None:
            if state.get("adamw") is None:
                raise ValueError("Checkpoint is missing auxiliary AdamW state.")
            self.adamw.load_state_dict(state["adamw"])


Optimizer = AdamW | HybridOptimizer


class WarmupCosineScheduler:
    """Single scheduler that writes the same numeric LR to every optimizer group."""

    def __init__(self, optimizer: Optimizer, *, base_lr: float, min_lr: float, warmup_steps: int, total_steps: int) -> None:
        if not (0.0 < min_lr <= base_lr):
            raise ValueError("Expected 0 < min_lr <= base_lr.")
        self.optimizer = optimizer
        self.base_lr = float(base_lr)
        self.min_lr = float(min_lr)
        self.warmup_steps = max(int(warmup_steps), 0)
        self.total_steps = max(int(total_steps), 1)
        self.step_number = 0
        self._apply_lr(self.lr_for_step(0))

    def lr_for_step(self, step: int) -> float:
        if self.warmup_steps and step < self.warmup_steps:
            return self.base_lr * float(step + 1) / float(self.warmup_steps)
        if self.total_steps <= self.warmup_steps:
            return self.base_lr
        progress = (step - self.warmup_steps) / float(self.total_steps - self.warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))

    def _apply_lr(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = float(lr)

    def step(self) -> None:
        self.step_number += 1
        self._apply_lr(self.lr_for_step(self.step_number))

    def get_last_lr(self) -> list[float]:
        return [float(group["lr"]) for group in self.optimizer.param_groups]

    def state_dict(self) -> dict[str, Any]:
        return {
            "base_lr": self.base_lr,
            "min_lr": self.min_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "step_number": self.step_number,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = (self.base_lr, self.min_lr, self.warmup_steps, self.total_steps)
        observed = (
            float(state["base_lr"]),
            float(state["min_lr"]),
            int(state["warmup_steps"]),
            int(state["total_steps"]),
        )
        if observed != expected:
            raise ValueError(f"Scheduler checkpoint/config mismatch: checkpoint={observed}, current={expected}.")
        self.step_number = int(state["step_number"])
        self._apply_lr(self.lr_for_step(self.step_number))


def _parameter_partition(model: nn.Module, weight_decay: float) -> tuple[dict[str, list[dict[str, Any]]], dict[int, str]]:
    groups = build_hybrid_optimizer_param_groups(model, weight_decay=weight_decay)
    assignment: dict[int, str] = {}
    for optimizer_name, optimizer_groups in groups.items():
        for group in optimizer_groups:
            group["weight_decay"] = float(weight_decay)
            for parameter in group["params"]:
                parameter_id = id(parameter)
                if parameter_id in assignment:
                    raise ValueError(f"Parameter is assigned more than once: {parameter_id}.")
                assignment[parameter_id] = optimizer_name
    trainable = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if set(assignment) != trainable:
        missing = trainable - set(assignment)
        unexpected = set(assignment) - trainable
        raise ValueError(f"Invalid optimizer partition: missing={len(missing)}, unexpected={len(unexpected)}.")
    return groups, assignment


def build_optimizer(model: nn.Module, train_config: dict[str, Any]) -> Optimizer:
    optimizer_config = train_config["optimizer"]
    name = str(optimizer_config["name"]).lower()
    lr = float(train_config["learning_rate"])
    weight_decay = float(train_config["weight_decay"])
    betas = tuple(float(value) for value in optimizer_config.get("betas", (0.9, 0.95)))
    if name == "adamw":
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        return AdamW(
            [{"params": parameters, "weight_decay": weight_decay}],
            lr=lr,
            betas=betas,
            eps=float(optimizer_config.get("eps", 1.0e-8)),
            weight_decay=weight_decay,
            foreach=False,
        )
    if name != "muon":
        raise ValueError(f"Unsupported optimizer: {name!r}.")
    groups, _ = _parameter_partition(model, weight_decay)
    muon = None
    if groups["muon"]:
        muon = Muon(
            groups["muon"],
            lr=lr,
            momentum=float(optimizer_config["momentum"]),
            ns_steps=int(optimizer_config["newton_schulz_iterations"]),
            ns_method=str(optimizer_config["newton_schulz"]),
            nesterov=bool(optimizer_config.get("nesterov", True)),
            weight_decay=weight_decay,
            update_scale=float(optimizer_config["update_rms_target"]),
        )
    adamw = None
    if groups["adamw"]:
        adamw = AdamW(
            groups["adamw"],
            lr=lr,
            betas=betas,
            eps=float(optimizer_config["eps"]),
            weight_decay=weight_decay,
            foreach=False,
        )
    optimizer = HybridOptimizer(muon=muon, adamw=adamw)
    lrs = {float(group["lr"]) for group in optimizer.param_groups}
    if lrs != {lr}:
        raise ValueError(f"Muon and auxiliary AdamW must use the same LR, got {sorted(lrs)}.")
    return optimizer


def parameter_group_rows(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    _, assignment = _parameter_partition(model, weight_decay)
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        rows.append(
            {
                "parameter_name": name,
                "shape": "x".join(str(value) for value in parameter.shape),
                "ndim": parameter.ndim,
                "numel": parameter.numel(),
                "optimizer": assignment[id(parameter)],
                "weight_decay": weight_decay,
            }
        )
    return rows


def write_parameter_group_csv(model: nn.Module, path: str | Path, weight_decay: float) -> None:
    rows = parameter_group_rows(model, weight_decay)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
