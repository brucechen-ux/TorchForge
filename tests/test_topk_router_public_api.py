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


def test_score_correction_bias_changes_selection_but_not_weights() -> None:
    router = TopKRouter(
        hidden_size=2,
        num_experts=4,
        top_k=2,
        score_function="sigmoid",
        normalize_topk=False,
        score_correction_bias=True,
    )
    with torch.no_grad():
        router.proj.weight.zero_()
        router.e_score_correction_bias.copy_(torch.tensor([0.0, 0.0, 10.0, 9.0]))

    hidden_states = torch.zeros(1, 2)
    outputs = router(hidden_states)

    assert outputs["selected_experts"].tolist() == [[2, 3]]
    assert torch.allclose(outputs["router_scores"], torch.full((1, 4), 0.5))
    assert torch.allclose(outputs["routing_weights"], torch.tensor([[0.5, 0.5]]))
    assert torch.allclose(outputs["selection_scores"], torch.tensor([[0.5, 0.5, 10.5, 9.5]]))


def test_group_limited_routing_selects_from_top_groups_only() -> None:
    router = TopKRouter(
        hidden_size=2,
        num_experts=6,
        top_k=2,
        score_function="sigmoid",
        normalize_topk=False,
        n_group=3,
        topk_group=1,
        bias=True,
    )
    with torch.no_grad():
        router.proj.weight.zero_()
        router.proj.bias.copy_(torch.tensor([0.1, 0.2, 5.0, 4.0, 3.0, 2.0]))

    outputs = router(torch.zeros(1, 2))

    assert outputs["selected_experts"].tolist() == [[2, 3]]
    assert set(outputs["selected_groups"].flatten().tolist()) == {1}


def test_routed_scaling_factor_alias_scales_selected_weights() -> None:
    router = TopKRouter(
        hidden_size=2,
        num_experts=4,
        top_k=2,
        score_function="sigmoid",
        normalize_topk=True,
        routed_scaling_factor=2.5,
    )
    with torch.no_grad():
        router.proj.weight.zero_()
        router.proj.bias.zero_()

    outputs = router(torch.zeros(1, 2))

    assert torch.allclose(outputs["routing_weights"].sum(dim=-1), torch.tensor([2.5]))
