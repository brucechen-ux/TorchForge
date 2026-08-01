from __future__ import annotations

import math
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn

from torchforge.common.nn import RMSNorm, SiTUGLU

from .quantile_router import QuantileBalancingRouter


class _SiTUGatedMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        beta_gate: float,
        beta_up: float,
        bias: bool,
    ) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.activation = SiTUGLU(beta_gate=beta_gate, beta_up=beta_up)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.activation((self.gate_proj(hidden_states), self.up_proj(hidden_states))))


class _PackedLatentExperts(nn.Module):
    def __init__(
        self,
        *,
        num_experts: int,
        latent_size: int,
        intermediate_size: int,
        beta_gate: float,
        beta_up: float,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.latent_size = latent_size
        self.intermediate_size = intermediate_size
        self.beta_gate = float(beta_gate)
        self.beta_up = float(beta_up)
        self.gate_weight = nn.Parameter(torch.empty(num_experts, intermediate_size, latent_size))
        self.up_weight = nn.Parameter(torch.empty(num_experts, intermediate_size, latent_size))
        self.down_weight = nn.Parameter(torch.empty(num_experts, latent_size, intermediate_size))
        for weight in (self.gate_weight, self.up_weight, self.down_weight):
            nn.init.kaiming_uniform_(weight.flatten(0, 1), a=math.sqrt(5))

    def forward(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        routed = torch.zeros_like(hidden_states)
        expert_load = torch.bincount(
            selected_experts.reshape(-1), minlength=self.num_experts
        ).to(torch.float32)
        for expert_index in range(self.num_experts):
            token_positions, route_positions = (selected_experts == expert_index).nonzero(as_tuple=True)
            if token_positions.numel() == 0:
                continue
            expert_input = hidden_states[token_positions]
            gate = F.linear(expert_input, self.gate_weight[expert_index])
            up = F.linear(expert_input, self.up_weight[expert_index])
            activated = (
                self.beta_gate * torch.tanh(gate / self.beta_gate)
                * torch.sigmoid(gate)
                * self.beta_up
                * torch.tanh(up / self.beta_up)
            )
            expert_output = F.linear(activated, self.down_weight[expert_index])
            weighted = expert_output * routing_weights[token_positions, route_positions].unsqueeze(-1)
            routed = routed.index_add(0, token_positions, weighted)
        return routed, expert_load


class StableLatentMoE(nn.Module):
    """Kimi-K3 Stable LatentMoE with shared full-width and routed latent paths."""

    def __init__(
        self,
        *,
        hidden_size: int,
        latent_size: int,
        num_experts: int,
        top_k: int,
        expert_intermediate_size: int,
        num_shared_experts: int = 2,
        shared_intermediate_size: Optional[int] = None,
        beta_gate: float = 4.0,
        beta_up: float = 25.0,
        histogram_bins: int = 256,
        rms_norm_eps: float = 1.0e-6,
        bias: bool = False,
    ) -> None:
        super().__init__()
        values = {
            "hidden_size": hidden_size,
            "latent_size": latent_size,
            "num_experts": num_experts,
            "top_k": top_k,
            "expert_intermediate_size": expert_intermediate_size,
            "num_shared_experts": num_shared_experts,
        }
        for name, value in values.items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive int, got {value!r}.")
        if top_k >= num_experts:
            raise ValueError("top_k must be less than num_experts.")
        shared_intermediate_size = (
            expert_intermediate_size if shared_intermediate_size is None else shared_intermediate_size
        )
        if shared_intermediate_size <= 0:
            raise ValueError("shared_intermediate_size must be positive.")
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_shared_experts = num_shared_experts
        self.latent_down = nn.Linear(hidden_size, latent_size, bias=bias)
        self.router = QuantileBalancingRouter(
            hidden_size=hidden_size,
            num_experts=num_experts,
            top_k=top_k,
            histogram_bins=histogram_bins,
            bias=bias,
        )
        self.experts = _PackedLatentExperts(
            num_experts=num_experts,
            latent_size=latent_size,
            intermediate_size=expert_intermediate_size,
            beta_gate=beta_gate,
            beta_up=beta_up,
        )
        self.routed_norm = RMSNorm(latent_size, eps=rms_norm_eps)
        self.latent_up = nn.Linear(latent_size, hidden_size, bias=bias)
        self.shared_experts = nn.ModuleList(
            _SiTUGatedMLP(
                hidden_size,
                shared_intermediate_size,
                beta_gate=beta_gate,
                beta_up=beta_up,
                bias=bias,
            )
            for _ in range(num_shared_experts)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        record_router_statistics: bool = True,
        return_dict: bool = True,
    ) -> Any:
        if not isinstance(hidden_states, torch.Tensor):
            raise TypeError("hidden_states must be a torch.Tensor.")
        if hidden_states.dim() != 3 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"hidden_states must have shape (batch, sequence, {self.hidden_size}), got {tuple(hidden_states.shape)}."
            )
        original_shape = hidden_states.shape
        flat = hidden_states.reshape(-1, self.hidden_size)
        router_output = self.router(
            flat,
            record_statistics=record_router_statistics,
            return_dict=True,
        )
        latent = self.latent_down(flat)
        routed, expert_load = self.experts(
            latent,
            router_output["routing_weights"],
            router_output["selected_experts"],
        )
        routed_output = self.latent_up(self.routed_norm(routed))
        shared_output = torch.zeros_like(flat)
        for expert in self.shared_experts:
            shared_output = shared_output + expert(flat)
        output = (shared_output + routed_output).view(original_shape)
        if not return_dict:
            return output, router_output["routing_weights"], router_output["selected_experts"]
        return {
            "hidden_states": output,
            "routing_weights": router_output["routing_weights"].view(
                *original_shape[:-1], self.top_k
            ),
            "selected_experts": router_output["selected_experts"].view(
                *original_shape[:-1], self.top_k
            ),
            "router_scores": router_output["router_scores"].view(
                *original_shape[:-1], self.num_experts
            ),
            "expert_load": expert_load,
            "router_bias": self.router.expert_bias.detach().clone(),
        }

    @torch.no_grad()
    def update_router_bias(self, *, distributed: bool = True) -> torch.Tensor:
        return self.router.update_bias(distributed=distributed)


__all__ = ["StableLatentMoE"]
