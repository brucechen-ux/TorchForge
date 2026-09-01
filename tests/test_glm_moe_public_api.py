from __future__ import annotations

import pytest
import torch

from torchforge.common.moe import GLM53FlashMoE, GLMMoE, SharedExpertMLP


def _moe(**kwargs: object) -> GLM53FlashMoE:
    arguments = {
        "hidden_size": 8,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "expert_intermediate_size": 16,
        "shared_expert_intermediate_size": 12,
        "router_score_function": "sigmoid",
        "normalize_topk": True,
        "routed_scaling_factor": 1.5,
    }
    arguments.update(kwargs)
    return GLM53FlashMoE(**arguments)


def test_glm_moe_public_forward_and_alias() -> None:
    assert GLMMoE is GLM53FlashMoE
    moe = _moe(return_aux_loss=True, aux_loss_alpha=0.01)
    hidden_states = torch.randn(2, 3, 8, requires_grad=True)

    outputs = moe(hidden_states, output_router_logits=True)
    outputs["hidden_states"].square().mean().backward()

    assert isinstance(moe.shared_expert, SharedExpertMLP)
    assert outputs["hidden_states"].shape == hidden_states.shape
    assert outputs["routing_weights"].shape == (2, 3, 2)
    assert outputs["selected_experts"].shape == (2, 3, 2)
    assert outputs["expert_load"].shape == (4,)
    assert outputs["router_logits"].shape == (2, 3, 4)
    assert outputs["router_scores"].shape == (2, 3, 4)
    assert outputs["aux_loss"].shape == ()
    assert torch.isfinite(outputs["hidden_states"]).all()
    assert hidden_states.grad is not None
    assert torch.isfinite(hidden_states.grad).all()


def test_glm_moe_adds_shared_and_routed_paths() -> None:
    moe = _moe(num_experts_per_tok=1, routed_scaling_factor=1.0).eval()
    hidden_states = torch.randn(2, 3, 8)
    flat = hidden_states.reshape(-1, 8)
    router_outputs = moe.router(flat)
    expected_routed = torch.zeros_like(flat)
    for expert_id, expert in enumerate(moe.experts):
        token_mask = router_outputs["selected_experts"] == expert_id
        if token_mask.any():
            token_pos, route_pos = token_mask.nonzero(as_tuple=True)
            weights = router_outputs["routing_weights"][token_pos, route_pos].unsqueeze(-1)
            expected_routed.index_add_(0, token_pos, expert(flat[token_pos]) * weights)
    expected = expected_routed + moe.shared_expert(flat)

    actual = moe(hidden_states)["hidden_states"]

    assert torch.allclose(actual, expected.reshape_as(hidden_states))


def test_glm_moe_tuple_output_and_score_correction_bias() -> None:
    moe = _moe(router_score_correction_bias=True)
    hidden_states = torch.randn(1, 2, 8)

    output, weights, selected = moe(hidden_states, return_dict=False)

    assert output.shape == hidden_states.shape
    assert weights.shape == (1, 2, 2)
    assert selected.shape == (1, 2, 2)
    assert moe.router.e_score_correction_bias is not None


def test_glm_moe_validates_top_k_aliases() -> None:
    with pytest.raises(TypeError, match="top_k or num_experts_per_tok"):
        GLM53FlashMoE(
            hidden_size=8,
            num_experts=4,
            expert_intermediate_size=16,
            shared_expert_intermediate_size=12,
        )
    with pytest.raises(ValueError, match="must match"):
        _moe(top_k=1, num_experts_per_tok=2)
