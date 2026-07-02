from __future__ import annotations

import torch

from torchforge.common.residual import ResidualAdd


def test_residual_add_forward() -> None:
    residual = torch.ones(2, 3)
    update = torch.full((2, 3), 2.0)
    output = ResidualAdd(scale=0.5)(residual, update)
    torch.testing.assert_close(output, torch.full((2, 3), 2.0))
