from __future__ import annotations

from typing import Optional

from torch import nn

from .moe import MoE
from .shared_expert import SharedExpertMLP


class GLM53FlashMoE(MoE):
    """GLM-5.3-Flash-style MoE with routed and always-on shared experts.

    The routed path is implemented by :class:`MoE`; this specialization builds
    the shared expert and exposes GLM's ``num_experts_per_tok`` naming.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_experts: int,
        expert_intermediate_size: int,
        shared_expert_intermediate_size: int,
        top_k: Optional[int] = None,
        num_experts_per_tok: Optional[int] = None,
        router_score_function: str = "sigmoid",
        normalize_topk: bool = True,
        routed_scaling_factor: float = 1.0,
        expert_activation: str = "silu",
        expert_gated: bool = True,
        bias: bool = False,
        router_score_correction_bias: bool = False,
        router_bias_update_rate: float = 1.0e-3,
        return_aux_loss: bool = False,
        aux_loss_alpha: float = 0.0,
        return_router_outputs: bool = False,
        expert_clamp_limit: Optional[float] = None,
        expert_beta_gate: float = 4.0,
        expert_beta_up: float = 25.0,
    ) -> None:
        resolved_top_k = _resolve_top_k(top_k, num_experts_per_tok)
        shared_expert: nn.Module = SharedExpertMLP(
            hidden_size=hidden_size,
            intermediate_size=shared_expert_intermediate_size,
            activation=expert_activation,
            gated=expert_gated,
            bias=bias,
            clamp_limit=expert_clamp_limit,
            beta_gate=expert_beta_gate,
            beta_up=expert_beta_up,
        )
        super().__init__(
            hidden_size=hidden_size,
            num_experts=num_experts,
            top_k=resolved_top_k,
            expert_intermediate_size=expert_intermediate_size,
            shared_expert=shared_expert,
            router_score_function=router_score_function,
            normalize_topk=normalize_topk,
            routed_scaling_factor=routed_scaling_factor,
            router_score_correction_bias=router_score_correction_bias,
            router_bias_update_rate=router_bias_update_rate,
            return_aux_loss=return_aux_loss,
            aux_loss_alpha=aux_loss_alpha,
            expert_activation=expert_activation,
            expert_beta_gate=expert_beta_gate,
            expert_beta_up=expert_beta_up,
            expert_gated=expert_gated,
            bias=bias,
            return_router_outputs=return_router_outputs,
            expert_clamp_limit=expert_clamp_limit,
        )
        self.num_experts_per_tok = resolved_top_k
        self.expert_intermediate_size = expert_intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size


def _resolve_top_k(top_k: Optional[int], num_experts_per_tok: Optional[int]) -> int:
    if top_k is None and num_experts_per_tok is None:
        raise TypeError("GLM53FlashMoE requires top_k or num_experts_per_tok.")
    if top_k is not None and num_experts_per_tok is not None and top_k != num_experts_per_tok:
        raise ValueError("top_k and num_experts_per_tok must match when both are provided.")
    resolved = top_k if top_k is not None else num_experts_per_tok
    if not isinstance(resolved, int) or resolved <= 0:
        raise ValueError("top_k must be a positive int.")
    return resolved


GLMMoE = GLM53FlashMoE


__all__ = ["GLM53FlashMoE", "GLMMoE"]
