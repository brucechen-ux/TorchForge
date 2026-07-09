from __future__ import annotations

import torch
from torch import nn

from torchforge.common.optim import AdamW, Muon, build_hybrid_optimizer_param_groups, build_param_groups


def test_adamw_step_updates_parameters() -> None:
    torch.manual_seed(0)
    model = nn.Linear(4, 4)
    optimizer = AdamW(model.parameters(), lr=1e-1)
    before = model.weight.detach().clone()
    loss = model(torch.randn(3, 4)).pow(2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert not torch.allclose(before, model.weight)


def test_build_param_groups_splits_decay_and_no_decay() -> None:
    model = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
    groups = build_param_groups(model, weight_decay=0.1)
    assert len(groups) == 2
    decay = next(g for g in groups if g["weight_decay"] == 0.1)
    no_decay = next(g for g in groups if g["weight_decay"] == 0.0)
    # 2-D weights decay; biases and 1-D norm params do not.
    assert all(p.dim() >= 2 for p in decay["params"])
    assert all(p.dim() < 2 for p in no_decay["params"])


def test_adamw_rejects_invalid_lr() -> None:
    model = nn.Linear(2, 2)
    try:
        AdamW(model.parameters(), lr=0.0)
    except ValueError:
        return
    raise AssertionError("expected ValueError for non-positive lr.")


def test_muon_rejects_1d_parameters() -> None:
    param = nn.Parameter(torch.ones(4))
    try:
        Muon([param])
    except ValueError:
        return
    raise AssertionError("expected Muon to reject non-matrix parameters.")


def test_muon_step_updates_2d_parameters() -> None:
    torch.manual_seed(0)
    param = nn.Parameter(torch.randn(4, 4))
    optimizer = Muon([param], lr=1e-2, momentum=0.0, ns_steps=3)
    before = param.detach().clone()
    param.grad = torch.randn_like(param)

    optimizer.step()

    assert not torch.allclose(before, param)


def test_build_hybrid_optimizer_param_groups_splits_muon_and_adamw_params() -> None:
    model = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
    groups = build_hybrid_optimizer_param_groups(model, weight_decay=0.1)

    assert set(groups) == {"muon", "adamw"}
    assert all(p.dim() >= 2 for group in groups["muon"] for p in group["params"])
    assert all(p.dim() < 2 for group in groups["adamw"] for p in group["params"])
    assert groups["muon"][0]["weight_decay"] == 0.1
    assert groups["adamw"][0]["weight_decay"] == 0.0
