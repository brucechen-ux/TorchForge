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
        routed_scaling_factor: DeepSeek-style alias for ``route_scale``.
        router_score_correction_bias: Whether created routers maintain aux-loss-free
            per-expert correction bias for future routing decisions.
        router_n_group: Number of expert groups for group-limited routing.
        router_topk_group: Number of groups retained before top-k expert selection.
        router_bias_update_rate: Default update rate for score correction bias.
        return_aux_loss: Whether to return auxiliary load-balancing loss by default.
        aux_loss_alpha: Scale applied to the auxiliary load-balancing loss.
        expert_activation: Activation used by created ``ExpertMLP`` experts.
        expert_gated: Whether created experts use gated MLPs.
        bias: Whether created router and expert projections use bias.
        return_router_outputs: Whether router logits/scores are returned by default.

    Forward:
        ``hidden_states`` has shape ``(..., hidden_size)``.

    Returns:
        With ``return_dict=True``, returns ``hidden_states``, ``routing_weights``,
        ``selected_experts``, ``expert_load``, and optional router logits/scores,
        auxiliary loss, and router bias.
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
        routed_scaling_factor: Optional[float] = None,
        router_score_correction_bias: bool = False,
        router_n_group: Optional[int] = None,
        router_topk_group: Optional[int] = None,
        router_bias_update_rate: float = 1.0e-3,
        return_aux_loss: bool = False,
        aux_loss_alpha: float = 0.0,
        expert_activation: str = "silu",
        expert_gated: bool = True,
        bias: bool = False,
        return_router_outputs: bool = False,
        expert_clamp_limit: Optional[float] = None,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")
        if router_bias_update_rate <= 0.0:
            raise ValueError("router_bias_update_rate must be positive.")
        if aux_loss_alpha < 0.0:
            raise ValueError("aux_loss_alpha must be non-negative.")

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
                    clamp_limit=expert_clamp_limit,
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
        self.return_aux_loss = return_aux_loss
        self.aux_loss_alpha = aux_loss_alpha
        self.router_bias_update_rate = router_bias_update_rate
        self.router = router or TopKRouter(
            hidden_size=hidden_size,
            num_experts=self.num_experts,
            top_k=top_k,
            score_function=router_score_function,
            normalize_topk=normalize_topk,
            route_scale=route_scale,
            routed_scaling_factor=routed_scaling_factor,
            score_correction_bias=router_score_correction_bias,
            n_group=router_n_group,
            topk_group=router_topk_group,
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
        output_aux_loss: Optional[bool] = None,
        update_router_bias: bool = False,
        router_bias_update_rate: Optional[float] = None,
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
        want_aux_loss = self.return_aux_loss if output_aux_loss is None else output_aux_loss
        aux_loss = None
        if want_aux_loss:
            balance_selected_experts = torch.topk(
                router_outputs["router_scores"],
                k=self.top_k,
                dim=-1,
            ).indices
            aux_loss = _sequence_wise_balance_loss(
                router_outputs["router_scores"],
                balance_selected_experts,
                self.num_experts,
                top_k=self.top_k,
                leading_shape=original_shape[:-1],
                alpha=self.aux_loss_alpha,
            )

        router_bias = None
        if update_router_bias:
            rate = self.router_bias_update_rate if router_bias_update_rate is None else router_bias_update_rate
            router_bias = self.router.update_score_correction_bias(expert_load, update_rate=rate)

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
            if aux_loss is not None:
                result["aux_loss"] = aux_loss
            if router_bias is not None:
                result["router_bias"] = router_bias.detach().clone()
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


def _sequence_wise_balance_loss(
    router_scores: torch.Tensor,
    selected_experts: torch.Tensor,
    num_experts: int,
    *,
    top_k: int,
    leading_shape: torch.Size,
    alpha: float,
) -> torch.Tensor:
    """DeepSeek-V4 sequence-wise load-balancing loss (paper Section 2.1).

    Balances expert load within each individual sequence rather than over the
    whole batch. For each sequence, ``f_i = (N / K) * mean_t 1(i in top-k(t))``
    and ``P_i = mean_t (s_i,t / sum_j s_j,t)``; the loss is
    ``alpha * mean_seq sum_i f_i P_i``.
    """

    if alpha == 0.0:
        return router_scores.new_zeros(())

    # Recover the (num_sequences, seq_len) split. The last leading dim is the
    # sequence axis; anything before it indexes independent sequences.
    if len(leading_shape) >= 2:
        seq_len = int(leading_shape[-1])
        num_seq = 1
        for dim in leading_shape[:-1]:
            num_seq *= int(dim)
    else:
        seq_len = int(leading_shape[-1]) if len(leading_shape) == 1 else router_scores.shape[0]
        num_seq = 1

    scores = router_scores.reshape(num_seq, seq_len, num_experts)
    selected = selected_experts.reshape(num_seq, seq_len, -1)

    expert_mask = torch.zeros(num_seq, seq_len, num_experts, device=scores.device, dtype=scores.dtype)
    expert_mask.scatter_add_(-1, selected, torch.ones_like(selected, dtype=scores.dtype))
    # f_i normalized by top_k so that a perfectly balanced router yields f_i == 1.
    f_i = expert_mask.mean(dim=1) * (num_experts / top_k)
    normalized_scores = scores / scores.sum(dim=-1, keepdim=True).clamp_min(1.0e-9)
    p_i = normalized_scores.mean(dim=1)
    per_sequence = (f_i * p_i).sum(dim=-1)
    return (alpha * per_sequence.mean()).to(router_scores.dtype)


__all__ = ["MoE"]
