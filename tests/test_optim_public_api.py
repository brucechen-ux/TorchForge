from __future__ import annotations

import torch
from torch import nn

from torchforge.common.optim import AdamW, build_param_groups


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
