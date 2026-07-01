from __future__ import annotations

from typing import Any, Iterable, Optional

import torch
from torch import nn

from .expert import ExpertMLP
from .router import TopKRouter


class MoE(nn.Module):
    """Local Mixture-of-Experts layer with top-k routing.

    Args:
        hidden_size: Size of the input and output hidden-state dimension.
        router: Optional prebuilt ``TopKRouter``.
        experts: Optional iterable of expert modules.
        num_experts: Number of experts to create when ``experts`` is omitted.
        top_k: Number of experts selected for each token.
        expert_intermediate_size: Intermediate size for created ``ExpertMLP`` experts.
        shared_expert: Optional expert module added to every token output.
        router_score_function: Score function used by a created router.
        normalize_topk: Whether selected expert weights are normalized to sum to one.
        route_scale: Multiplicative scale applied to selected routing weights.
        expert_activation: Activation used by created ``ExpertMLP`` experts.
        expert_gated: Whether created experts use gated MLPs.
        bias: Whether created router and expert projections use bias.
        return_router_outputs: Whether router logits/scores are returned by default.

    Forward:
        ``hidden_states`` has shape ``(..., hidden_size)``.

    Returns:
        With ``return_dict=True``, returns ``hidden_states``, ``routing_weights``,
        ``selected_experts``, ``expert_load``, and optional router logits/scores.
        With ``return_dict=False``, returns ``(hidden_states, routing_weights, selected_experts)``,
        plus ``router_logits`` as a fourth item when requested.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        router: Optional[TopKRouter] = None,
        experts: Optional[Iterable[nn.Module]] = None,
        num_experts: Optional[int] = None,
        top_k: int = 1,
        expert_intermediate_size: Optional[int] = None,
        shared_expert: Optional[nn.Module] = None,
        router_score_function: str = "softmax",
        normalize_topk: bool = True,
        route_scale: float = 1.0,
        expert_activation: str = "silu",
        expert_gated: bool = True,
        bias: bool = False,
        return_router_outputs: bool = False,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")

        if experts is None:
            if num_experts is None:
                raise TypeError("MoE requires either experts or num_experts.")
            if expert_intermediate_size is None:
                raise TypeError("MoE requires expert_intermediate_size when experts are not provided.")
            experts = [
                ExpertMLP(
                    hidden_size=hidden_size,
                    intermediate_size=expert_intermediate_size,
                    activation=expert_activation,
                    gated=expert_gated,
                    bias=bias,
                )
                for _ in range(num_experts)
            ]
        expert_list = list(experts)
        if not expert_list:
            raise ValueError("MoE requires at least one expert.")
        if top_k <= 0 or top_k > len(expert_list):
            raise ValueError("top_k must be in the range [1, num_experts].")

        self.hidden_size = hidden_size
        self.num_experts = len(expert_list)
        self.top_k = top_k
        self.experts = nn.ModuleList(expert_list)
        self.shared_expert = shared_expert
        self.return_router_outputs = return_router_outputs
        self.router = router or TopKRouter(
            hidden_size=hidden_size,
            num_experts=self.num_experts,
            top_k=top_k,
            score_function=router_score_function,
            normalize_topk=normalize_topk,
            route_scale=route_scale,
            bias=bias,
        )
        if self.router.num_experts != self.num_experts:
            raise ValueError("router.num_experts must match the number of experts.")
        if self.router.top_k != top_k:
            raise ValueError("router.top_k must match top_k.")

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        output_router_logits: bool = False,
        return_dict: bool = True,
    ) -> Any:
        if not isinstance(hidden_states, torch.Tensor):
            raise TypeError(f"hidden_states must be a torch.Tensor, got {type(hidden_states).__name__}.")
        if hidden_states.dim() < 2:
            raise ValueError("hidden_states must have at least 2 dimensions.")
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Expected hidden_states last dimension to be {self.hidden_size}, "
                f"got {hidden_states.shape[-1]}."
            )

        original_shape = hidden_states.shape
        flat = hidden_states.reshape(-1, self.hidden_size)
        router_outputs = self.router(flat, return_dict=True)
        routing_weights = router_outputs["routing_weights"]
        selected_experts = router_outputs["selected_experts"]

        routed = torch.zeros_like(flat)
        expert_load = torch.zeros(self.num_experts, device=flat.device, dtype=torch.float32)
        for expert_id, expert in enumerate(self.experts):
            token_mask = selected_experts == expert_id
            if not token_mask.any():
                routed = routed + expert(flat[:1]).sum() * 0.0
                continue
            token_pos, route_pos = token_mask.nonzero(as_tuple=True)
            expert_input = flat[token_pos]
            expert_output = expert(expert_input).to(flat.dtype)
            weight = routing_weights[token_pos, route_pos].unsqueeze(-1).to(flat.dtype)
            routed.index_add_(0, token_pos, expert_output * weight)
            expert_load[expert_id] = float(token_pos.numel())

        if self.shared_expert is not None:
            routed = routed + self.shared_expert(flat).to(flat.dtype)

        output = routed.reshape(original_shape)
        if return_dict:
            result = {
                "hidden_states": output,
                "routing_weights": routing_weights.reshape(*original_shape[:-1], self.top_k),
                "selected_experts": selected_experts.reshape(*original_shape[:-1], self.top_k),
                "expert_load": expert_load,
            }
            if output_router_logits or self.return_router_outputs:
                result["router_logits"] = router_outputs["router_logits"].reshape(*original_shape[:-1], self.num_experts)
                result["router_scores"] = router_outputs["router_scores"].reshape(*original_shape[:-1], self.num_experts)
            return result

        if output_router_logits or self.return_router_outputs:
            return (
                output,
                routing_weights.reshape(*original_shape[:-1], self.top_k),
                selected_experts.reshape(*original_shape[:-1], self.top_k),
                router_outputs["router_logits"].reshape(*original_shape[:-1], self.num_experts),
            )
        return output, routing_weights.reshape(*original_shape[:-1], self.top_k), selected_experts.reshape(
            *original_shape[:-1], self.top_k
        )


__all__ = ["MoE"]
