from __future__ import annotations

import torch

from experiments.k3_small.config import tiny_k3_config
from experiments.k3_small.model import SmallK3Model
from torchforge.common.optim import Muon, build_k3_optimizer_param_groups


def test_k3_optimizer_groups_packed_heads_with_muon_and_router_with_adamw() -> None:
    model = SmallK3Model(tiny_k3_config())
    groups = build_k3_optimizer_param_groups(model)
    muon_ids = {id(parameter) for group in groups["muon"] for parameter in group["params"]}
    adamw_ids = {id(parameter) for group in groups["adamw"] for parameter in group["params"]}

    first_kda = model.layers[0].attention
    first_moe = model.layers[1].ffn
    assert first_kda.q_weight.ndim == 3
    assert id(first_kda.q_weight) in muon_ids
    assert id(first_kda.output_norm_weight) in adamw_ids
    assert id(first_moe.router.proj.weight) in adamw_ids
    assert muon_ids.isdisjoint(adamw_ids)


def test_muon_updates_each_packed_head_as_an_independent_matrix() -> None:
    torch.manual_seed(11)
    initial = torch.randn(3, 4, 6)
    gradient = torch.randn_like(initial)
    packed = torch.nn.Parameter(initial.clone())
    separate = [torch.nn.Parameter(matrix.clone()) for matrix in initial]
    packed.grad = gradient.clone()
    for parameter, matrix_gradient in zip(separate, gradient):
        parameter.grad = matrix_gradient.clone()

    Muon([packed], lr=0.01, momentum=0.0, ns_steps=3).step()
    for parameter in separate:
        Muon([parameter], lr=0.01, momentum=0.0, ns_steps=3).step()

    assert torch.allclose(packed, torch.stack(separate), atol=1.0e-6, rtol=1.0e-6)
