from __future__ import annotations

import torch

from torchforge.common.nn import FeedForward, GEGLU, MLP, RMSNorm, SwiGLU, UnweightedRMSNorm


def test_public_rms_norm_can_be_instantiated_directly() -> None:
    norm = RMSNorm(hidden_size=16, eps=1e-6)
    hidden_states = torch.randn(2, 3, 16)

    outputs = norm(hidden_states)

    assert outputs.shape == hidden_states.shape


def test_public_unweighted_rms_norm_can_be_instantiated_directly() -> None:
    norm = UnweightedRMSNorm(eps=1e-6)
    hidden_states = torch.randn(2, 3, 16)

    outputs = norm(hidden_states)

    assert outputs.shape == hidden_states.shape


def test_public_swiglu_can_be_instantiated_directly() -> None:
    activation = SwiGLU()
    hidden_states = torch.randn(2, 3, 32)

    outputs = activation(hidden_states)

    assert outputs.shape == (2, 3, 16)


def test_public_geglu_can_be_instantiated_directly() -> None:
    activation = GEGLU()
    gate = torch.randn(2, 3, 16)
    value = torch.randn(2, 3, 16)

    outputs = activation((gate, value))

    assert outputs.shape == gate.shape


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


def test_public_mlp_can_be_instantiated_directly() -> None:
    mlp = MLP(
        input_size=16,
        hidden_size=32,
        output_size=8,
        num_layers=3,
        activation="gelu",
        dropout=0.0,
        bias=True,
    )
    mlp.eval()
    inputs = torch.randn(2, 3, 16)

    outputs = mlp(inputs)

    assert outputs.shape == (2, 3, 8)
