from __future__ import annotations

import torch

from torchforge.common.attention import CausalMask


def test_causal_mask_shape_and_values() -> None:
    mask = CausalMask()(batch_size=1, seq_length=3, dtype=torch.float32)
    assert mask.shape == (1, 1, 3, 3)
    assert mask[0, 0, 0, 0].item() == 0.0
    assert mask[0, 0, 0, 1].item() < -1.0e20
    assert mask[0, 0, 2, 0].item() == 0.0


def test_causal_mask_with_past_length() -> None:
    mask = CausalMask()(batch_size=2, seq_length=3, past_length=2, dtype=torch.float32)
    assert mask.shape == (2, 1, 3, 5)
