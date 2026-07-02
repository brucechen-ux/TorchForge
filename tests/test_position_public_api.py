from __future__ import annotations

import torch

from torchforge.common.position import PositionIds


def test_position_ids_from_input_ids() -> None:
    position_ids = PositionIds()
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    output = position_ids(input_ids)
    assert output.shape == (2, 3)
    assert output.tolist() == [[0, 1, 2], [0, 1, 2]]


def test_position_ids_with_past_length() -> None:
    output = PositionIds()(batch_size=1, seq_length=3, past_length=4)
    assert output.tolist() == [[4, 5, 6]]
