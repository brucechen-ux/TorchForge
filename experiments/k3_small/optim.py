from __future__ import annotations

import math
from typing import Any

from torch import nn

from torchforge.common.optim import AdamW, Muon, build_k3_optimizer_param_groups


class K3Optimizer:
    """Checkpointable facade over Per-Head Muon and auxiliary AdamW."""

    def __init__(self, muon: Muon | None, adamw: AdamW | None) -> None:
        if muon is None and adamw is None:
            raise ValueError("K3Optimizer requires at least one optimizer.")
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
            "muon": None if self.muon is None else self.muon.state_dict(),
            "adamw": None if self.adamw is None else self.adamw.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if self.muon is not None:
            if state.get("muon") is None:
                raise ValueError("Checkpoint is missing Muon state.")
            self.muon.load_state_dict(state["muon"])
        if self.adamw is not None:
            if state.get("adamw") is None:
                raise ValueError("Checkpoint is missing AdamW state.")
            self.adamw.load_state_dict(state["adamw"])


class WarmupCosineScheduler:
    def __init__(
        self,
        optimizer: K3Optimizer,
        *,
        base_lr: float,
        min_lr: float,
        warmup_steps: int,
        total_steps: int,
    ) -> None:
        if not 0.0 < min_lr <= base_lr:
            raise ValueError("Expected 0 < min_lr <= base_lr.")
        self.optimizer = optimizer
        self.base_lr = float(base_lr)
        self.min_lr = float(min_lr)
        self.warmup_steps = max(int(warmup_steps), 0)
        self.total_steps = max(int(total_steps), 1)
        self.step_number = 0
        self._apply(self.lr_for_step(0))

    def lr_for_step(self, step: int) -> float:
        if self.warmup_steps and step < self.warmup_steps:
            return self.base_lr * (step + 1) / self.warmup_steps
        denominator = max(self.total_steps - self.warmup_steps, 1)
        progress = min(max((step - self.warmup_steps) / denominator, 0.0), 1.0)
        return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
            1.0 + math.cos(math.pi * progress)
        )

    def _apply(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = float(lr)

    def step(self) -> None:
        self.step_number += 1
        self._apply(self.lr_for_step(self.step_number))

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
        actual = (
            float(state["base_lr"]),
            float(state["min_lr"]),
            int(state["warmup_steps"]),
            int(state["total_steps"]),
        )
        if actual != expected:
            raise ValueError(f"Scheduler config mismatch: checkpoint={actual}, current={expected}.")
        self.step_number = int(state["step_number"])
        self._apply(self.lr_for_step(self.step_number))


def build_optimizer(model: nn.Module, train_config: dict[str, Any]) -> K3Optimizer:
    optimizer_cfg = train_config["optimizer"]
    lr = float(train_config["learning_rate"])
    weight_decay = float(train_config["weight_decay"])
    groups = build_k3_optimizer_param_groups(model, weight_decay=weight_decay)
    muon = (
        Muon(
            groups["muon"],
            lr=lr,
            momentum=float(optimizer_cfg["momentum"]),
            ns_steps=int(optimizer_cfg["newton_schulz_iterations"]),
            ns_method=str(optimizer_cfg["newton_schulz"]),
            nesterov=bool(optimizer_cfg["nesterov"]),
            weight_decay=weight_decay,
            update_scale=float(optimizer_cfg["update_rms_target"]),
        )
        if groups["muon"]
        else None
    )
    betas = tuple(float(value) for value in optimizer_cfg["betas"])
    adamw = (
        AdamW(
            groups["adamw"],
            lr=lr,
            betas=betas,
            eps=float(optimizer_cfg["eps"]),
            weight_decay=weight_decay,
            foreach=False,
        )
        if groups["adamw"]
        else None
    )
    return K3Optimizer(muon=muon, adamw=adamw)


__all__ = ["K3Optimizer", "WarmupCosineScheduler", "build_optimizer"]
