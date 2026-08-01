from __future__ import annotations

import torch

from torchforge.common.nn import SiTUGLU


def test_situ_glu_is_public_and_bounded() -> None:
    activation = SiTUGLU(beta_gate=4.0, beta_up=25.0)
    gate = torch.tensor([-1000.0, -1.0, 0.0, 1.0, 1000.0])
    value = torch.full_like(gate, 1000.0)

    output = activation((gate, value))

    assert output.shape == gate.shape
    assert torch.all(output.abs() <= 100.0 + 1.0e-5)


def test_situ_glu_accepts_concatenated_input() -> None:
    activation = SiTUGLU()
    inputs = torch.randn(2, 3, 8)

    output = activation(inputs)

    assert output.shape == (2, 3, 4)
