from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Root-mean-square normalization.

    Args:
        hidden_size: Size of the normalized last dimension.
        eps: Numerical stability epsilon.

    Forward:
        ``hidden_states`` has shape ``(..., hidden_size)``.

    Returns:
        Tensor with the same shape as ``hidden_states``.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")
        if eps <= 0.0:
            raise ValueError("eps must be positive.")
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        _validate_hidden_states(hidden_states, self.hidden_size)
        input_dtype = hidden_states.dtype
        hidden_fp32 = hidden_states.to(torch.float32)
        hidden_fp32 = hidden_fp32 * torch.rsqrt(hidden_fp32.square().mean(-1, keepdim=True) + self.eps)
        return self.weight * hidden_fp32.to(input_dtype)


class UnweightedRMSNorm(nn.Module):
    """Root-mean-square normalization without a learned scale parameter.

    Args:
        eps: Numerical stability epsilon.

    Forward:
        ``hidden_states`` has shape ``(..., hidden_size)``.

    Returns:
        Tensor with the same shape as ``hidden_states``.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        if eps <= 0.0:
            raise ValueError("eps must be positive.")
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not isinstance(hidden_states, torch.Tensor):
            raise TypeError(f"hidden_states must be a torch.Tensor, got {type(hidden_states).__name__}.")
        if hidden_states.dim() < 1:
            raise ValueError("hidden_states must have at least 1 dimension.")
        return hidden_states * torch.rsqrt(hidden_states.float().square().mean(-1, keepdim=True) + self.eps).to(
            hidden_states.dtype
        )


def _validate_hidden_states(hidden_states: torch.Tensor, hidden_size: int) -> None:
    if not isinstance(hidden_states, torch.Tensor):
        raise TypeError(f"hidden_states must be a torch.Tensor, got {type(hidden_states).__name__}.")
    if hidden_states.dim() < 1:
        raise ValueError("hidden_states must have at least 1 dimension.")
    if hidden_states.shape[-1] != hidden_size:
        raise ValueError(f"hidden_states last dimension must be {hidden_size}, got {hidden_states.shape[-1]}.")


__all__ = ["RMSNorm", "UnweightedRMSNorm"]
