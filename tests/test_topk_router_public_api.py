from __future__ import annotations

import torch

from torchforge.common.moe import TopKRouter


def test_public_topk_router_can_be_instantiated_directly() -> None:
    router = TopKRouter(
        hidden_size=16,
        num_experts=4,
        top_k=2,
        score_function="softmax",
        normalize_topk=True,
    )
    router.eval()

    hidden_states = torch.randn(2, 3, 16)
    outputs = router(hidden_states)

    assert outputs["routing_weights"].shape == (2, 3, 2)
    assert outputs["selected_experts"].shape == (2, 3, 2)
    assert outputs["router_logits"].shape == (2, 3, 4)
    assert outputs["router_scores"].shape == (2, 3, 4)

    routing_weights, selected_experts = router(hidden_states, return_dict=False)
    assert routing_weights.shape == (2, 3, 2)
    assert selected_experts.shape == (2, 3, 2)
