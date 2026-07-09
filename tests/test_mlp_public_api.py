from __future__ import annotations

import torch

from torchforge.common.mlp import FeedForward, GatedMLP


def test_public_feed_forward_can_be_instantiated_directly() -> None:
    feed_forward = FeedForward(
        hidden_size=16,
        intermediate_size=32,
        activation="swiglu",
        dropout=0.0,
        bias=False,
    )
    feed_forward.eval()
    hidden_states = torch.randn(2, 3, 16)

    outputs = feed_forward(hidden_states)

    assert outputs.shape == hidden_states.shape


def test_public_gated_mlp_can_be_instantiated_directly() -> None:
    mlp = GatedMLP(
        hidden_size=16,
        intermediate_size=32,
        activation="silu",
        gated=True,
        bias=False,
    )
    mlp.eval()
    hidden_states = torch.randn(2, 3, 16)

    outputs = mlp(hidden_states)

    assert outputs.shape == hidden_states.shape
    assert mlp.gate_proj is not None


def test_public_gated_mlp_ungated_has_no_gate_proj() -> None:
    mlp = GatedMLP(hidden_size=8, intermediate_size=16, gated=False)
    mlp.eval()

    outputs = mlp(torch.randn(2, 3, 8))

    assert outputs.shape == (2, 3, 8)
    assert mlp.gate_proj is None
