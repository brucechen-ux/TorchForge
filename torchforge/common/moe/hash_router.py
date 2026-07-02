from __future__ import annotations

from typing import Any, Optional

import torch
from torch import nn


class HashRouter(nn.Module):
    """Route tokens to experts using a deterministic token-id hash.

    Args:
        num_experts: Number of candidate experts.
        top_k: Number of experts selected for each token.
        seed: Integer offset mixed into the hash.

    Forward:
        ``input_ids`` has shape ``(...)`` and integer dtype.

    Returns:
        With ``return_dict=True``, returns ``routing_weights`` and ``selected_experts``.
        With ``return_dict=False``, returns ``(routing_weights, selected_experts)``.
    """

    def __init__(self, *, num_experts: int, top_k: int = 1, seed: int = 0) -> None:
        super().__init__()
        if not isinstance(num_experts, int) or num_experts <= 0:
            raise ValueError(f"num_experts must be a positive int, got {num_experts!r}.")
        if not isinstance(top_k, int) or top_k <= 0 or top_k > num_experts:
            raise ValueError("top_k must be in the range [1, num_experts].")
        if not isinstance(seed, int):
            raise ValueError(f"seed must be an int, got {seed!r}.")
        self.num_experts = num_experts
        self.top_k = top_k
        self.seed = seed

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        dtype: Optional[torch.dtype] = None,
        return_dict: bool = True,
    ) -> Any:
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError(f"input_ids must be a torch.Tensor, got {type(input_ids).__name__}.")
        if input_ids.dim() < 1:
            raise ValueError("input_ids must have at least 1 dimension.")
        if torch.is_floating_point(input_ids) or torch.is_complex(input_ids):
            raise TypeError("input_ids must use an integer dtype.")

        token_ids = input_ids.long()
        offsets = torch.arange(self.top_k, device=input_ids.device, dtype=torch.long)
        start = torch.remainder(token_ids + self.seed, self.num_experts)
        selected_experts = torch.remainder(start.unsqueeze(-1) + offsets, self.num_experts)
        routing_weights = torch.full(
            selected_experts.shape,
            1.0 / self.top_k,
            device=input_ids.device,
            dtype=dtype or torch.float32,
        )
        if return_dict:
            return {
                "routing_weights": routing_weights,
                "selected_experts": selected_experts,
            }
        return routing_weights, selected_experts


__all__ = ["HashRouter"]
