from __future__ import annotations

import torch
from torch import nn


class ResidualAdd(nn.Module):
    """Residual addition helper.

    Forward:
        ``residual`` and ``update`` must have identical shapes.

    Returns:
        ``residual + scale * update``.
    """

    def __init__(self, *, scale: float = 1.0) -> None:
        super().__init__()
        self.scale = float(scale)

    def forward(self, residual: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
        if not isinstance(residual, torch.Tensor) or not isinstance(update, torch.Tensor):
            raise TypeError("residual and update must be torch.Tensor instances.")
        if residual.shape != update.shape:
            raise ValueError(
                f"residual and update must have identical shapes, got {tuple(residual.shape)} and {tuple(update.shape)}."
            )
        return residual + update * self.scale


__all__ = ["ResidualAdd"]
