from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from torchforge.common.moe import QuantileBalancingRouter, StableLatentMoE
from torchforge.common.nn import SiTUGLU, SwiGLU


def test_quantile_router_bias_changes_selection_not_mixture_weights() -> None:
    router = QuantileBalancingRouter(hidden_size=4, num_experts=4, top_k=2, histogram_bins=16)
    router.eval()
    with torch.no_grad():
        router.proj.weight.zero_()
        router.expert_bias.copy_(torch.tensor([0.0, 0.0, 2.0, 1.0]))

    outputs = router(torch.zeros(3, 4))

    assert outputs["selected_experts"].unique().sort().values.tolist() == [2, 3]
    assert torch.allclose(outputs["routing_weights"], torch.full((3, 2), 0.5))


def test_quantile_router_eval_freezes_bias() -> None:
    router = QuantileBalancingRouter(hidden_size=4, num_experts=4, top_k=1, histogram_bins=16)
    router.eval()
    before = router.expert_bias.clone()
    updated = router.update_bias(distributed=False)
    assert torch.equal(updated, before)
    assert torch.equal(router.expert_bias, before)


def test_quantile_router_records_histogram_and_updates_next_bias() -> None:
    router = QuantileBalancingRouter(hidden_size=4, num_experts=4, top_k=1, histogram_bins=16)
    router.train()

    router(torch.randn(8, 4))
    updated = router.update_bias(distributed=False)

    assert updated.shape == (4,)
    assert torch.allclose(updated.mean(), torch.zeros(()), atol=1.0e-6)
    assert router.margin_histogram.sum() == 0


def test_quantile_router_toy_case_reaches_target_load() -> None:
    scores = torch.tensor(
        [
            [0.3768987, 0.7187048, 0.5793358, 0.1525653],
            [0.2476536, 0.3393543, 0.3242126, 0.7521984],
            [0.1764412, 0.1946016, 0.6142234, 0.6641689],
            [0.2648422, 0.2611809, 0.1011741, 0.8895128],
            [0.6914095, 0.6065961, 0.5251021, 0.5760433],
            [0.4467588, 0.7812403, 0.1654405, 0.7284009],
            [0.1265813, 0.1465102, 0.1301152, 0.2394405],
            [0.4644331, 0.2970643, 0.1048283, 0.7143591],
        ]
    )
    router = QuantileBalancingRouter(
        hidden_size=8,
        num_experts=4,
        top_k=1,
        histogram_bins=256,
    )
    with torch.no_grad():
        router.proj.weight.copy_(torch.logit(scores).transpose(0, 1))

    tokens = torch.eye(8)
    before = router(tokens)["selected_experts"]
    router.update_bias(distributed=False)
    after = router(tokens, record_statistics=False)["selected_experts"]

    assert torch.bincount(before.flatten(), minlength=4).tolist() == [1, 2, 0, 5]
    assert torch.bincount(after.flatten(), minlength=4).tolist() == [2, 2, 2, 2]


def test_stable_latent_moe_public_forward() -> None:
    moe = StableLatentMoE(
        hidden_size=8,
        latent_size=4,
        num_experts=4,
        top_k=2,
        expert_intermediate_size=6,
        num_shared_experts=2,
        shared_intermediate_size=6,
        histogram_bins=16,
    )
    hidden_states = torch.randn(2, 3, 8, requires_grad=True)

    outputs = moe(hidden_states)
    outputs["hidden_states"].sum().backward()

    assert outputs["hidden_states"].shape == hidden_states.shape
    assert outputs["routing_weights"].shape == (2, 3, 2)
    assert outputs["selected_experts"].shape == (2, 3, 2)
    assert outputs["expert_load"].sum() == 12
    assert hidden_states.grad is not None


@pytest.mark.parametrize("expert_activation", ["swiglu", "situglu"])
def test_stable_latent_moe_packed_experts_match_per_route_reference(
    expert_activation: str,
) -> None:
    torch.manual_seed(23)
    moe = StableLatentMoE(
        hidden_size=8,
        latent_size=4,
        num_experts=4,
        top_k=2,
        expert_intermediate_size=6,
        num_shared_experts=2,
        shared_intermediate_size=6,
        expert_activation=expert_activation,
        histogram_bins=16,
    ).eval()
    hidden_states = torch.randn(1, 3, 8)
    outputs = moe(hidden_states)
    flat = hidden_states.flatten(0, 1)
    routing = moe.router(flat, record_statistics=False)
    latent = moe.latent_down(flat)
    routed = torch.zeros_like(latent)
    for token_index in range(flat.shape[0]):
        for route_index in range(moe.top_k):
            expert_index = int(routing["selected_experts"][token_index, route_index])
            gate = F.linear(latent[token_index], moe.experts.gate_weight[expert_index])
            up = F.linear(latent[token_index], moe.experts.up_weight[expert_index])
            if expert_activation == "swiglu":
                activated = F.silu(gate) * up
            else:
                activated = (
                    moe.experts.beta_gate
                    * torch.tanh(gate / moe.experts.beta_gate)
                    * torch.sigmoid(gate)
                    * moe.experts.beta_up
                    * torch.tanh(up / moe.experts.beta_up)
                )
            expert_output = F.linear(activated, moe.experts.down_weight[expert_index])
            routed[token_index] += routing["routing_weights"][token_index, route_index] * expert_output
    expected = moe.latent_up(moe.routed_norm(routed))
    for shared_expert in moe.shared_experts:
        expected += shared_expert(flat)

    assert torch.allclose(outputs["hidden_states"].flatten(0, 1), expected, atol=1.0e-6, rtol=1.0e-6)
    expected_type = SwiGLU if expert_activation == "swiglu" else SiTUGLU
    assert isinstance(moe.experts.activation, expected_type)
    for shared_expert in moe.shared_experts:
        assert shared_expert.activation == expert_activation
        if expert_activation == "situglu":
            assert isinstance(shared_expert.gated_activation, SiTUGLU)


def test_stable_latent_moe_rejects_unknown_expert_activation() -> None:
    with pytest.raises(ValueError, match="expert_activation"):
        StableLatentMoE(
            hidden_size=8,
            latent_size=4,
            num_experts=4,
            top_k=2,
            expert_intermediate_size=6,
            expert_activation="unknown",
        )
