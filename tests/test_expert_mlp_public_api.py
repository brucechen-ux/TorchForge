from __future__ import annotations

import torch

from torchforge.common.moe import ExpertMLP


def test_public_expert_mlp_can_be_instantiated_directly() -> None:
    expert = ExpertMLP(
        hidden_size=16,
        intermediate_size=32,
        activation="silu",
        gated=True,
    )
    expert.eval()

    hidden_states = torch.randn(2, 3, 16)
    outputs = expert(hidden_states)

    assert outputs.shape == hidden_states.shape
