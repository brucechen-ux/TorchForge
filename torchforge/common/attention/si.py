"""Streaming-Aware Indexing (SI) component for LSA.

Splits attention budget into contiguous regions (sink + sliding window)
and dynamic sparse selection for hardware-friendly memory access.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    input_dtype = x.dtype
    x_fp32 = x.float()
    variance = x_fp32.square().mean(-1, keepdim=True)
    x_fp32 = x_fp32 * torch.rsqrt(variance + eps)
    return weight * x_fp32.to(input_dtype)


class LightningIndexer(nn.Module):
    """Lightning Indexer for token scoring (DSA baseline).
    
    Args:
        hidden_size: Input hidden dimension.
        num_heads: Number of indexer heads.
        head_dim: Dimension per indexer head.
        rms_norm_eps: RMS normalization epsilon.
    """
    
    def __init__(self, hidden_size: int, num_heads: int, head_dim: int, rms_norm_eps: float = 1e-6) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.rms_norm_eps = rms_norm_eps
        
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, head_dim, bias=False)
        self.weights = nn.Parameter(torch.ones(num_heads))
        self.q_norm = nn.Parameter(torch.ones(head_dim))
        self.k_norm = nn.Parameter(torch.ones(head_dim))
    
    def forward(self, hidden_states: torch.Tensor, position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute indexer scores for all token pairs.
        
        Args:
            hidden_states: Shape (batch, seq_len, hidden_size).
            position_ids: Optional position IDs.
        
        Returns:
            scores: Shape (batch, seq_len, seq_len) where scores[b, t, s]
                   is the saliency score of token s for query t.
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states)
        
        q = _rms_norm(q, self.q_norm, self.rms_norm_eps)
        k = _rms_norm(k, self.k_norm, self.rms_norm_eps)
        
        k = k.unsqueeze(2)
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = F.relu(scores)
        
        weights = self.weights.view(1, 1, self.num_heads, 1)
        scores = (scores * weights).sum(dim=2)
        
        return scores


class StreamingAwareIndexer(nn.Module):
    """Streaming-Aware Indexing component.
    
    Splits attention budget into:
    1. Sink tokens: Fixed initial tokens (e.g., BOS)
    2. Sliding window: Recent contiguous tokens (hardware-friendly)
    3. Sparse selection: Dynamically selected important tokens
    
    Args:
        base_indexer: The underlying Lightning Indexer.
        total_budget: Total attention budget K.
        sink_size: Number of sink tokens.
        window_size: Size of the sliding window.
    """
    
    def __init__(
        self,
        base_indexer: LightningIndexer,
        total_budget: int,
        sink_size: int,
        window_size: int,
    ) -> None:
        super().__init__()
        self.base_indexer = base_indexer
        self.total_budget = total_budget
        self.sink_size = sink_size
        self.window_size = window_size
        self.sparse_budget = total_budget - sink_size - window_size
        
        if self.sparse_budget <= 0:
            raise ValueError(
                f"Invalid budget split: total={total_budget}, sink={sink_size}, "
                f"window={window_size}, sparse={self.sparse_budget}"
            )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Select tokens using streaming-aware strategy.
        
        Args:
            hidden_states: Shape (batch, seq_len, hidden_size).
            position_ids: Optional position IDs.
        
        Returns:
            selected_indices: Shape (batch, seq_len, total_budget).
            components: Dict with 'sink', 'window', 'sparse' indices.
        """
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        
        scores = self.base_indexer(hidden_states, position_ids)
        
        selected_indices = []
        components = {'sink': [], 'window': [], 'sparse': []}
        
        for t in range(seq_len):
            sink_idx = torch.arange(min(self.sink_size, t + 1), device=device)
            
            window_start = max(0, t - self.window_size + 1)
            window_idx = torch.arange(window_start, t + 1, device=device)
            
            if t + 1 > self.sink_size + self.window_size:
                sparse_region_end = window_start
                if sparse_region_end > self.sink_size:
                    region_scores = scores[:, t, self.sink_size:sparse_region_end]
                    k = min(self.sparse_budget, sparse_region_end - self.sink_size)
                    _, sparse_relative_idx = torch.topk(region_scores, k, dim=-1)
                    sparse_idx = sparse_relative_idx + self.sink_size
                else:
                    sparse_idx = torch.empty((batch_size, 0), dtype=torch.long, device=device)
            else:
                sparse_idx = torch.empty((batch_size, 0), dtype=torch.long, device=device)
            
            sink_idx = sink_idx.unsqueeze(0).expand(batch_size, -1)
            window_idx = window_idx.unsqueeze(0).expand(batch_size, -1)
            
            combined = torch.cat([sink_idx, window_idx, sparse_idx], dim=-1)
            
            if combined.shape[-1] < self.total_budget:
                padding = torch.full(
                    (batch_size, self.total_budget - combined.shape[-1]),
                    -1,
                    dtype=torch.long,
                    device=device,
                )
                combined = torch.cat([combined, padding], dim=-1)
            else:
                combined = combined[:, :self.total_budget]
            
            selected_indices.append(combined)
            components['sink'].append(sink_idx)
            components['window'].append(window_idx)
            components['sparse'].append(sparse_idx if sparse_idx.shape[-1] > 0 else None)
        
        selected_indices = torch.stack(selected_indices, dim=1)
        
        return selected_indices, components


__all__ = ["LightningIndexer", "StreamingAwareIndexer"]
