"""Cross-Layer Indexing (CLI) component for LSA.

Amortizes indexing computation by reusing indices across layer groups.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn


class CrossLayerIndexCache:
    """Cache for storing and retrieving cross-layer indices.
    
    Manages index reuse across layer groups for Cross-Layer Indexing.
    """
    
    def __init__(self):
        self.cache: Dict[int, torch.Tensor] = {}
    
    def store(self, group_id: int, indices: torch.Tensor) -> None:
        """Store indices for a layer group."""
        self.cache[group_id] = indices
    
    def retrieve(self, group_id: int) -> Optional[torch.Tensor]:
        """Retrieve cached indices for a layer group."""
        return self.cache.get(group_id)
    
    def clear(self) -> None:
        """Clear all cached indices."""
        self.cache.clear()


class CrossLayerIndexer(nn.Module):
    """Cross-Layer Indexing wrapper.
    
    Manages owner/reuse layer logic for index sharing across layer groups.
    
    Args:
        layer_idx: Current layer index.
        group_size: Number of layers per CLI group.
        cache: Shared CrossLayerIndexCache instance.
    """
    
    def __init__(self, layer_idx: int, group_size: int, cache: CrossLayerIndexCache) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.group_size = group_size
        self.cache = cache
        
        self.group_id = layer_idx // group_size
        self.is_owner = (layer_idx % group_size) == 0
    
    def should_compute(self) -> bool:
        """Check if this layer should compute indices."""
        return self.is_owner
    
    def store(self, indices: torch.Tensor) -> None:
        """Store indices in cache (owner layer only)."""
        if not self.is_owner:
            raise RuntimeError("Only owner layers can store indices")
        self.cache.store(self.group_id, indices)
    
    def retrieve(self) -> Optional[torch.Tensor]:
        """Retrieve cached indices (reuse layer only)."""
        if self.is_owner:
            raise RuntimeError("Owner layers should compute, not retrieve")
        return self.cache.retrieve(self.group_id)
    
    def compute_distillation_loss(
        self,
        indexer_scores: torch.Tensor,
        target_attention_weights: torch.Tensor,
        distillation_weight: float = 1.0,
    ) -> torch.Tensor:
        """Compute cross-layer distillation loss for CLI training.
        
        Args:
            indexer_scores: Shape (batch, seq_len, seq_len).
            target_attention_weights: Aggregated attention from group layers.
                Shape (batch, seq_len, seq_len).
            distillation_weight: Loss weight.
        
        Returns:
            loss: KL divergence between indexer scores and target attention.
        """
        if not self.is_owner:
            raise RuntimeError("Only owner layers can compute distillation loss")
        
        losses = []
        for t in range(indexer_scores.shape[1]):
            pred_scores = indexer_scores[:, t, :t+1]
            target_weights = target_attention_weights[:, t, :t+1]
            
            pred_dist = F.softmax(pred_scores, dim=-1)
            target_dist = target_weights / (target_weights.sum(dim=-1, keepdim=True) + 1e-10)
            
            kl = F.kl_div(
                pred_dist.log(),
                target_dist,
                reduction='batchmean',
                log_target=False,
            )
            losses.append(kl)
        
        return torch.stack(losses).mean() * distillation_weight


__all__ = ["CrossLayerIndexCache", "CrossLayerIndexer"]
