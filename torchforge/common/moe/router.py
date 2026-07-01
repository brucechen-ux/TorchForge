from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class TopKRouter(nn.Module):
    """Route each token to its top-k experts.

    Args:
        hidden_size: Size of the input hidden-state dimension.
        num_experts: Number of candidate experts.
        top_k: Number of experts selected for each token.
        score_function: Router score function, either ``"softmax"`` or ``"sigmoid"``.
        normalize_topk: Whether selected expert weights are normalized to sum to one.
        route_scale: Multiplicative scale applied to selected routing weights.
        bias: Whether the router projection uses bias.

    Forward:
        ``hidden_states`` has shape ``(..., hidden_size)``.

    Returns:
        With ``return_dict=True``, returns ``routing_weights``, ``selected_experts``,
        ``router_logits``, and ``router_scores``.
        With ``return_dict=False``, returns ``(routing_weights, selected_experts)``.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        score_function: str = "softmax",
        normalize_topk: bool = True,
        route_scale: float = 1.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")
        if num_experts <= 0:
            raise ValueError("num_experts must be positive.")
        if top_k <= 0 or top_k > num_experts:
            raise ValueError("top_k must be in the range [1, num_experts].")
        if score_function not in {"softmax", "sigmoid"}:
            raise ValueError(f"Unsupported score_function: {score_function!r}.")

        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.score_function = score_function
        self.normalize_topk = normalize_topk
        self.route_scale = route_scale
        self.proj = nn.Linear(hidden_size, num_experts, bias=bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        return_dict: bool = True,
    ) -> Any:
        if not isinstance(hidden_states, torch.Tensor):
            raise TypeError(f"hidden_states must be a torch.Tensor, got {type(hidden_states).__name__}.")
        if hidden_states.dim() < 2:
            raise ValueError("hidden_states must have at least 2 dimensions.")
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(f"hidden_states last dimension must be {self.hidden_size}, got {hidden_states.shape[-1]}.")
        logits = self.proj(hidden_states.float())
        if self.score_function == "softmax":
            scores = F.softmax(logits, dim=-1)
        elif self.score_function == "sigmoid":
            scores = torch.sigmoid(logits)
        else:
            raise ValueError(f"Unsupported score_function: {self.score_function!r}.")

        routing_weights, selected_experts = torch.topk(scores, k=self.top_k, dim=-1)
        if self.normalize_topk and self.top_k > 1:
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-9)
        routing_weights = routing_weights * self.route_scale
        routing_weights = routing_weights.to(hidden_states.dtype)

        if return_dict:
            return {
                "routing_weights": routing_weights,
                "selected_experts": selected_experts,
                "router_logits": logits,
                "router_scores": scores,
            }
        return routing_weights, selected_experts


__all__ = ["TopKRouter"]
