from __future__ import annotations

import torch

from torchforge.common.attention import SlidingWindowCausalMask


def test_sliding_window_causal_mask_shape_and_values() -> None:
    mask = SlidingWindowCausalMask(window_size=2)(batch_size=1, seq_length=4, dtype=torch.float32)
    assert mask.shape == (1, 1, 4, 4)
    assert mask[0, 0, 0, 0].item() == 0.0
    assert mask[0, 0, 0, 1].item() < -1.0e20
    assert mask[0, 0, 2, 0].item() < -1.0e20
    assert mask[0, 0, 2, 1].item() == 0.0


def test_sliding_window_causal_mask_with_past_length() -> None:
    mask = SlidingWindowCausalMask(window_size=3)(batch_size=2, seq_length=4, past_length=2)
    assert mask.shape == (2, 1, 4, 6)
