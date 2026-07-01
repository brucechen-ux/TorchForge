from __future__ import annotations

import torch

from torchforge.common.moe import MoE


def test_public_moe_can_be_instantiated_directly() -> None:
    moe = MoE(
        hidden_size=16,
        num_experts=4,
        top_k=2,
        expert_intermediate_size=32,
        router_score_function="softmax",
        normalize_topk=True,
    )
    moe.eval()

    hidden_states = torch.randn(2, 3, 16)
    outputs = moe(hidden_states, output_router_logits=True)

    assert outputs["hidden_states"].shape == hidden_states.shape
    assert outputs["routing_weights"].shape == (2, 3, 2)
    assert outputs["selected_experts"].shape == (2, 3, 2)
    assert outputs["router_logits"].shape == (2, 3, 4)
    assert outputs["router_scores"].shape == (2, 3, 4)
    assert outputs["expert_load"].shape == (4,)

    hidden_out, routing_weights, selected_experts = moe(hidden_states, return_dict=False)
    assert hidden_out.shape == hidden_states.shape
    assert routing_weights.shape == (2, 3, 2)
    assert selected_experts.shape == (2, 3, 2)
