from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn


class TopKRouter(nn.Module):
    """Route each token to its top-k experts.

    Args:
        hidden_size: Size of the input hidden-state dimension.
        num_experts: Number of candidate experts.
        top_k: Number of experts selected for each token.
        score_function: Router score function, one of ``"softmax"``, ``"sigmoid"``,
            or ``"sqrt_softplus"`` (the DeepSeek-V4 affinity function).
        normalize_topk: Whether selected expert weights are normalized to sum to one.
        route_scale: Multiplicative scale applied to selected routing weights.
        routed_scaling_factor: DeepSeek-style alias for ``route_scale``.
        score_correction_bias: Whether to maintain a per-expert correction bias
            used only for expert selection, not for selected routing weights.
        e_score_correction_bias: Optional initial value for the correction bias.
        n_group: Number of expert groups for group-limited routing.
        topk_group: Number of groups retained before top-k expert selection.
        bias: Whether the router projection uses bias.

    Forward:
        ``hidden_states`` has shape ``(..., hidden_size)``.

    Returns:
        With ``return_dict=True``, returns ``routing_weights``, ``selected_experts``,
        ``router_logits``, ``router_scores``, ``selection_scores``, and optionally
        ``selected_groups`` when group-limited routing is active.
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
        routed_scaling_factor: Optional[float] = None,
        score_correction_bias: bool = False,
        e_score_correction_bias: Optional[torch.Tensor] = None,
        n_group: Optional[int] = None,
        topk_group: Optional[int] = None,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")
        if num_experts <= 0:
            raise ValueError("num_experts must be positive.")
        if top_k <= 0 or top_k > num_experts:
            raise ValueError("top_k must be in the range [1, num_experts].")
        if score_function not in {"softmax", "sigmoid", "sqrt_softplus"}:
            raise ValueError(f"Unsupported score_function: {score_function!r}.")
        if routed_scaling_factor is not None:
            if route_scale != 1.0 and float(route_scale) != float(routed_scaling_factor):
                raise ValueError("Pass only one of route_scale or routed_scaling_factor, or pass matching values.")
            route_scale = float(routed_scaling_factor)
        if route_scale <= 0.0:
            raise ValueError("route_scale must be positive.")
        if (n_group is None) != (topk_group is None):
            raise ValueError("n_group and topk_group must be passed together.")
        if n_group is not None:
            if n_group <= 0 or num_experts % n_group != 0:
                raise ValueError("n_group must be positive and divide num_experts.")
            if topk_group is None or topk_group <= 0 or topk_group > n_group:
                raise ValueError("topk_group must be in the range [1, n_group].")

        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.score_function = score_function
        self.normalize_topk = normalize_topk
        self.route_scale = route_scale
        self.routed_scaling_factor = route_scale
        self.n_group = n_group
        self.topk_group = topk_group
        self.proj = nn.Linear(hidden_size, num_experts, bias=bias)

        if score_correction_bias or e_score_correction_bias is not None:
            initial = torch.zeros(num_experts) if e_score_correction_bias is None else e_score_correction_bias.detach().float()
            if initial.shape != (num_experts,):
                raise ValueError(f"e_score_correction_bias must have shape ({num_experts},).")
            self.e_score_correction_bias = nn.Parameter(initial.clone(), requires_grad=False)
        else:
            self.register_parameter("e_score_correction_bias", None)

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
        scores = _router_scores(logits, self.score_function)
        selection_scores = scores
        if self.e_score_correction_bias is not None:
            selection_scores = selection_scores + self.e_score_correction_bias.to(selection_scores.dtype)

        selected_groups = None
        if self.n_group is not None and self.topk_group is not None:
            selection_scores, selected_groups = _apply_group_limited_routing(
                selection_scores,
                n_group=self.n_group,
                topk_group=self.topk_group,
            )

        _, selected_experts = torch.topk(selection_scores, k=self.top_k, dim=-1)
        routing_weights = torch.gather(scores, dim=-1, index=selected_experts)
        if self.normalize_topk and self.top_k > 1:
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-9)
        routing_weights = (routing_weights * self.route_scale).to(hidden_states.dtype)

        if return_dict:
            result = {
                "routing_weights": routing_weights,
                "selected_experts": selected_experts,
                "router_logits": logits,
                "router_scores": scores,
                "selection_scores": selection_scores,
            }
            if selected_groups is not None:
                result["selected_groups"] = selected_groups
            return result
        return routing_weights, selected_experts

    @torch.no_grad()
    def update_score_correction_bias(
        self,
        expert_load: torch.Tensor,
        *,
        target_load: Optional[torch.Tensor] = None,
        update_rate: float = 1.0e-3,
    ) -> torch.Tensor:
        """Update aux-loss-free load-balancing bias from observed expert load.

        Experts above target receive a lower correction bias and underloaded
        experts receive a higher one. The bias affects future top-k selection
        only; it never changes already selected routing weights.
        """

        if self.e_score_correction_bias is None:
            raise RuntimeError("score correction bias is not enabled for this router.")
        if update_rate <= 0.0:
            raise ValueError("update_rate must be positive.")
        load = _validate_load("expert_load", expert_load, self.num_experts).to(self.e_score_correction_bias.device)
        if target_load is None:
            target = load.new_full(load.shape, load.mean())
        else:
            target = _validate_load("target_load", target_load, self.num_experts).to(load.device)
        delta = torch.sign(target - load) * float(update_rate)
        self.e_score_correction_bias.add_(delta.to(self.e_score_correction_bias.dtype))
        return self.e_score_correction_bias


def _router_scores(logits: torch.Tensor, score_function: str) -> torch.Tensor:
    if score_function == "softmax":
        return F.softmax(logits, dim=-1)
    if score_function == "sigmoid":
        return torch.sigmoid(logits)
    if score_function == "sqrt_softplus":
        # DeepSeek-V4 affinity function: Sqrt(Softplus(.)), a non-negative,
        # per-expert (unnormalized) gate replacing DeepSeek-V3's Sigmoid.
        return torch.sqrt(F.softplus(logits))
    raise ValueError(f"Unsupported score_function: {score_function!r}.")


def _apply_group_limited_routing(
    selection_scores: torch.Tensor,
    *,
    n_group: int,
    topk_group: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    group_size = selection_scores.shape[-1] // n_group
    grouped = selection_scores.view(*selection_scores.shape[:-1], n_group, group_size)
    group_top_k = min(2, group_size)
    group_scores = torch.topk(grouped, k=group_top_k, dim=-1).values.sum(dim=-1)
    _, selected_groups = torch.topk(group_scores, k=topk_group, dim=-1)
    group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
    group_mask.scatter_(-1, selected_groups, True)
    expert_mask = group_mask.unsqueeze(-1).expand_as(grouped).reshape_as(selection_scores)
    masked = selection_scores.masked_fill(~expert_mask, float("-inf"))
    return masked, selected_groups


def _validate_load(name: str, value: torch.Tensor, num_experts: int) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}.")
    if tuple(value.shape) != (num_experts,):
        raise ValueError(f"{name} must have shape ({num_experts},), got {tuple(value.shape)}.")
    if not torch.is_floating_point(value):
        value = value.float()
    return value


__all__ = ["TopKRouter"]
