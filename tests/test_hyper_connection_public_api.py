from __future__ import annotations

import torch

from torchforge.common.residual import ManifoldConstrainedHyperConnection


def test_hyper_connection_public_api_forward_shape() -> None:
    mhc = ManifoldConstrainedHyperConnection(hidden_size=8, expansion_factor=3, sinkhorn_iters=5)
    hidden_states = torch.randn(2, 4, 8)
    residual_state = mhc.init_state(hidden_states)
    output = mhc(residual_state, torch.randn(2, 4, 8))
    assert output["residual_state"].shape == (2, 4, 3, 8)
    assert output["hidden_states"].shape == (2, 4, 8)


def test_dynamic_hyper_connection_public_api_forward_shape() -> None:
    mhc = ManifoldConstrainedHyperConnection(hidden_size=8, expansion_factor=3, sinkhorn_iters=5, dynamic=True)
    hidden_states = torch.randn(2, 4, 8)
    residual_state = mhc.init_state(hidden_states)
    output = mhc(residual_state, torch.randn(2, 4, 8))
    assert output["residual_mapping"].shape == (2, 4, 3, 3)
    assert output["hidden_states"].shape == (2, 4, 8)
