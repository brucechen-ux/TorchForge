from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn


class QuantileBalancingRouter(nn.Module):
    """Auxiliary-loss-free Kimi-K3 router with histogram quantile updates."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        histogram_bins: int = 256,
        histogram_min: float | None = None,
        histogram_max: float | None = None,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or num_experts <= 0:
            raise ValueError("hidden_size and num_experts must be positive.")
        if top_k <= 0 or top_k >= num_experts:
            raise ValueError("top_k must be in [1, num_experts).")
        if histogram_bins <= 1:
            raise ValueError("histogram_bins must be greater than one.")
        if (histogram_min is None) != (histogram_max is None):
            raise ValueError("histogram_min and histogram_max must either both be set or both be omitted.")
        if histogram_min is not None and histogram_min >= histogram_max:
            raise ValueError("histogram_min must be less than histogram_max.")
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.histogram_bins = histogram_bins
        self.histogram_min = None if histogram_min is None else float(histogram_min)
        self.histogram_max = None if histogram_max is None else float(histogram_max)
        self.proj = nn.Linear(hidden_size, num_experts, bias=bias)
        self.register_buffer(
            "expert_bias",
            torch.zeros(num_experts, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "margin_histogram",
            torch.zeros(num_experts, histogram_bins, dtype=torch.int64),
            persistent=False,
        )

    def _active_histogram_range(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.histogram_min is not None and self.histogram_max is not None:
            return (
                self.expert_bias.new_tensor(self.histogram_min),
                self.expert_bias.new_tensor(self.histogram_max),
            )
        return (
            self.expert_bias.min() - 1.0,
            self.expert_bias.max() + 1.0,
        )

    @torch.no_grad()
    def _record_required_biases(self, required_biases: torch.Tensor) -> None:
        histogram_min, histogram_max = self._active_histogram_range()
        scale = self.histogram_bins / (histogram_max - histogram_min)
        bin_indices = torch.floor((required_biases.float() - histogram_min) * scale).long()
        bin_indices = bin_indices.clamp(0, self.histogram_bins - 1)
        expert_indices = torch.arange(self.num_experts, device=required_biases.device).view(1, -1)
        flat_indices = expert_indices * self.histogram_bins + bin_indices
        counts = torch.bincount(
            flat_indices.reshape(-1),
            minlength=self.num_experts * self.histogram_bins,
        ).view(self.num_experts, self.histogram_bins)
        self.margin_histogram.add_(counts.to(self.margin_histogram.dtype))

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        record_statistics: bool = True,
        return_dict: bool = True,
    ) -> Any:
        if not isinstance(hidden_states, torch.Tensor):
            raise TypeError("hidden_states must be a torch.Tensor.")
        if hidden_states.dim() < 2 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(f"hidden_states last dimension must be {self.hidden_size}.")
        logits = F.linear(
            hidden_states.float(),
            self.proj.weight.float(),
            None if self.proj.bias is None else self.proj.bias.float(),
        )
        scores = torch.sigmoid(logits)
        selection_scores = scores + self.expert_bias.to(scores.dtype)
        top_values, top_indices = torch.topk(selection_scores, k=self.top_k + 1, dim=-1)
        selected_experts = top_indices[..., : self.top_k]
        cutoffs = top_values[..., self.top_k : self.top_k + 1]
        routing_weights = torch.gather(scores, dim=-1, index=selected_experts)
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-9)
        if self.training and record_statistics:
            required_biases = cutoffs.reshape(-1, 1) - scores.reshape(-1, self.num_experts)
            self._record_required_biases(required_biases.detach())
        routing_weights = routing_weights.to(hidden_states.dtype)
        if not return_dict:
            return routing_weights, selected_experts
        return {
            "routing_weights": routing_weights,
            "selected_experts": selected_experts,
            "router_logits": logits,
            "router_scores": scores,
            "selection_scores": selection_scores,
            "cutoffs": cutoffs,
        }

    @torch.no_grad()
    def update_bias(self, *, distributed: bool = True) -> torch.Tensor:
        """Apply the accumulated batch's quantile as the next-step bias."""

        if not self.training:
            return self.expert_bias.detach().clone()
        counts = self.margin_histogram.clone()
        if distributed and dist.is_available() and dist.is_initialized():
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        totals = counts.sum(dim=-1)
        if torch.any(totals <= 0):
            raise RuntimeError("QuantileBalancingRouter has no complete margin histogram to update from.")
        target = torch.ceil(totals * (self.top_k / self.num_experts)).clamp_min(1)
        cumulative = counts.cumsum(dim=-1)
        bin_indices = (cumulative >= target.unsqueeze(-1)).to(torch.int64).argmax(dim=-1)
        previous_indices = (bin_indices - 1).clamp_min(0)
        cumulative_before = cumulative.gather(1, previous_indices.unsqueeze(-1)).squeeze(-1)
        cumulative_before = torch.where(
            bin_indices == 0,
            torch.zeros_like(cumulative_before),
            cumulative_before,
        )
        selected_counts = counts.gather(1, bin_indices.unsqueeze(-1)).squeeze(-1).clamp_min(1)
        fraction = ((target - cumulative_before).float() / selected_counts.float()).clamp(0.0, 1.0)
        histogram_min, histogram_max = self._active_histogram_range()
        bin_width = (histogram_max - histogram_min) / self.histogram_bins
        next_bias = histogram_min + (bin_indices.float() + fraction) * bin_width
        next_bias = next_bias - next_bias.mean()
        self.expert_bias.copy_(next_bias.to(self.expert_bias.dtype))
        self.margin_histogram.zero_()
        return self.expert_bias.detach().clone()


__all__ = ["QuantileBalancingRouter"]
