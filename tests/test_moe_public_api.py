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


def test_moe_can_return_aux_loss() -> None:
    moe = MoE(
        hidden_size=8,
        num_experts=4,
        top_k=2,
        expert_intermediate_size=16,
        return_aux_loss=True,
        aux_loss_alpha=0.01,
    )

    outputs = moe(torch.randn(2, 3, 8))

    assert "aux_loss" in outputs
    assert outputs["aux_loss"].dim() == 0
    assert outputs["aux_loss"] >= 0.0


def test_moe_updates_router_score_correction_bias() -> None:
    moe = MoE(
        hidden_size=8,
        num_experts=4,
        top_k=1,
        expert_intermediate_size=16,
        router_score_correction_bias=True,
        router_bias_update_rate=0.1,
        bias=True,
    )
    with torch.no_grad():
        moe.router.proj.weight.zero_()
        moe.router.proj.bias.copy_(torch.tensor([10.0, 0.0, 0.0, 0.0]))
    assert moe.router.e_score_correction_bias is not None
    before = moe.router.e_score_correction_bias.detach().clone()

    outputs = moe(torch.randn(2, 3, 8), update_router_bias=True)

    after = moe.router.e_score_correction_bias.detach()
    assert outputs["router_bias"].shape == (4,)
    assert not torch.allclose(before, after)
