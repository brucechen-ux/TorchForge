from __future__ import annotations

import torch

from torchforge.common.moe import SharedExpertMLP


def test_shared_expert_public_api_forward_shape() -> None:
    expert = SharedExpertMLP(hidden_size=8, intermediate_size=16)
    hidden_states = torch.randn(2, 3, 8)
    output = expert(hidden_states)
    assert output.shape == hidden_states.shape
