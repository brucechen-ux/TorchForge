from __future__ import annotations

import torch

from torchforge.common.moe import HashRouter


def test_hash_router_public_api_forward_shape() -> None:
    router = HashRouter(num_experts=4, top_k=2, seed=1)
    output = router(torch.tensor([[1, 2, 3]]))
    assert output["routing_weights"].shape == (1, 3, 2)
    assert output["selected_experts"].shape == (1, 3, 2)


def test_hash_router_return_tuple() -> None:
    router = HashRouter(num_experts=4, top_k=1)
    weights, experts = router(torch.tensor([[1, 2]]), return_dict=False)
    assert weights.shape == experts.shape == (1, 2, 1)
