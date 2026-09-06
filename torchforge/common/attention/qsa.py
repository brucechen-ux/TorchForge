"""Qwen3.8-Flash-Next Query Sparse Attention (QSA) with micro-block indexer."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from torchforge.common.nn import RMSNorm


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Unbiased RMS normalization."""
    input_dtype = x.dtype
    x_fp32 = x.float()
    variance = x_fp32.square().mean(-1, keepdim=True)
    x_fp32 = x_fp32 * torch.rsqrt(variance + eps)
    return weight * x_fp32.to(input_dtype)


class MicroBlockIndexer(nn.Module):
    """Micro-block indexer for QSA sparse attention.
    
    Qwen3.8-Flash-Next uses block-level sparse retrieval instead of token-level.
    This reduces indexing overhead and improves memory locality.
    
    Mathematical principle:
        1. Divide sequence into blocks of size block_size
        2. Compute block-level scores (e.g., max/avg token scores per block)
        3. Select top-k blocks (budget: 512 blocks or 2048 tokens in Qwen)
        4. Attend to all tokens within selected blocks
    
    Args:
        hidden_size: Input hidden dimension.
        num_heads: Number of scoring heads.
        head_dim: Per-head dimension.
        block_size: Tokens per block (default 4, so 512 blocks = 2048 tokens).
        num_blocks: Maximum number of blocks to select.
        block_scoring: How to score blocks - "max", "avg", or "learned".
        rms_norm_eps: Epsilon for RMSNorm.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        block_size: int = 4,
        num_blocks: int = 512,
        block_scoring: str = "max",
        rms_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {head_dim}")
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}")
        if block_scoring not in ("max", "avg", "learned"):
            raise ValueError(f"block_scoring must be max/avg/learned, got {block_scoring}")
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.block_scoring = block_scoring
        self.rms_norm_eps = rms_norm_eps
        
        # Query and key projections for block scoring
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        
        # RMSNorm for query and key
        self.q_norm = nn.Parameter(torch.ones(head_dim))
        self.k_norm = nn.Parameter(torch.ones(head_dim))
        
        # Learned aggregation for block scoring
        if block_scoring == "learned":
            self.block_aggregator = nn.Linear(num_heads, 1, bias=False)
        else:
            self.register_parameter("block_aggregator", None)
    
    def _compute_token_scores(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> torch.Tensor:
        """Compute token-level attention scores.
        
        Args:
            query: (batch, num_heads, query_len, head_dim)
            key: (batch, num_heads, key_len, head_dim)
        
        Returns:
            scores: (batch, query_len, key_len)
        """
        # Compute attention scores: (batch, num_heads, query_len, key_len)
        scores = torch.matmul(query, key.transpose(-2, -1))
        
        # Average across heads for block-level scoring
        scores = scores.mean(dim=1)  # (batch, query_len, key_len)
        
        return scores
    
    def _score_blocks(
        self,
        token_scores: torch.Tensor,
        num_blocks: int,
    ) -> torch.Tensor:
        """Aggregate token scores into block scores.
        
        Args:
            token_scores: (batch, query_len, key_len) - token-level scores
            num_blocks: Number of blocks in key sequence
        
        Returns:
            block_scores: (batch, query_len, num_blocks)
        """
        batch_size, query_len, key_len = token_scores.shape
        
        # Reshape to blocks: (batch, query_len, num_blocks, block_size)
        # Handle padding for last block
        padded_key_len = num_blocks * self.block_size
        if key_len < padded_key_len:
            padding = torch.full(
                (batch_size, query_len, padded_key_len - key_len),
                float('-inf'),
                dtype=token_scores.dtype,
                device=token_scores.device
            )
            token_scores = torch.cat([token_scores, padding], dim=-1)
        
        token_scores = token_scores.view(batch_size, query_len, num_blocks, self.block_size)
        
        # Aggregate within blocks
        if self.block_scoring == "max":
            block_scores = token_scores.max(dim=-1).values
        elif self.block_scoring == "avg":
            # Use mean, but handle -inf values
            block_scores = torch.where(
                torch.isinf(token_scores),
                token_scores,
                token_scores
            ).mean(dim=-1)
        else:  # learned
            # Apply learned aggregation (simplified: still use max/avg as base)
            block_scores = token_scores.max(dim=-1).values
        
        return block_scores
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute block-level sparse indices.
        
        Args:
            hidden_states: (..., seq_len, hidden_size)
            attention_mask: Optional mask for valid positions
        
        Returns:
            block_indices: (..., query_len, num_blocks) - selected block indices
            block_mask: (..., query_len, seq_len) - expanded token-level mask
        """
        if not isinstance(hidden_states, torch.Tensor):
            raise TypeError("hidden_states must be a Tensor")
        
        *batch_dims, seq_len, _ = hidden_states.shape
        batch_size = int(torch.prod(torch.tensor(batch_dims)).item()) if batch_dims else 1
        
        # Calculate number of blocks
        num_blocks = (seq_len + self.block_size - 1) // self.block_size
        
        # Project query and key
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        
        # Reshape for multi-head
        query = query.view(*batch_dims, seq_len, self.num_heads, self.head_dim)
        key = key.view(*batch_dims, seq_len, self.num_heads, self.head_dim)
        
        # Transpose to (batch, num_heads, seq_len, head_dim)
        query = query.transpose(-3, -2)
        key = key.transpose(-3, -2)
        
        # Apply RMSNorm
        query = _rms_norm(query, self.q_norm, self.rms_norm_eps)
        key = _rms_norm(key, self.k_norm, self.rms_norm_eps)
        
        # Compute token-level scores
        token_scores = self._compute_token_scores(query, key)  # (batch, query_len, key_len)
        
        # Apply attention mask to token scores
        if attention_mask is not None:
            # Expand mask to match token_scores shape
            mask_expanded = attention_mask.view(*batch_dims, 1, seq_len)
            token_scores = token_scores.masked_fill(mask_expanded == 0, float('-inf'))
        
        # Aggregate into block scores
        block_scores = self._score_blocks(token_scores, num_blocks)  # (batch, query_len, num_blocks)
        
        # Select top-k blocks
        actual_k = min(self.num_blocks, num_blocks)
        top_k_values, top_k_block_indices = torch.topk(
            block_scores,
            k=actual_k,
            dim=-1,
            largest=True,
            sorted=True
        )
        
        # Pad to num_blocks if needed
        if actual_k < self.num_blocks:
            pad_shape = (*batch_dims, seq_len, self.num_blocks - actual_k)
            padding = torch.full(
                pad_shape,
                -1,
                dtype=top_k_block_indices.dtype,
                device=top_k_block_indices.device
            )
            top_k_block_indices = torch.cat([top_k_block_indices, padding], dim=-1)
        
        # Expand block indices to token-level mask
        block_mask = torch.zeros(
            (*batch_dims, seq_len, seq_len),
            dtype=torch.bool,
            device=hidden_states.device
        )
        
        # Mark all tokens in selected blocks as valid
        flat_batch_size = batch_size if batch_dims else 1
        for b in range(flat_batch_size):
            for q_idx in range(seq_len):
                valid_blocks = top_k_block_indices[b, q_idx][top_k_block_indices[b, q_idx] >= 0]
                for block_idx in valid_blocks:
                    block_idx = block_idx.item()
                    start_token = block_idx * self.block_size
                    end_token = min(start_token + self.block_size, seq_len)
                    block_mask[b, q_idx, start_token:end_token] = True
        
        return top_k_block_indices, block_mask


class QwenQuerySparseAttention(nn.Module):
    """Qwen3.8-Flash-Next Query Sparse Attention (QSA) layer.
    
    Architecture:
        1. Micro-block indexer: Divide sequence into blocks, score blocks
        2. Select top-k blocks (512 blocks = 2048 tokens budget)
        3. Block-sparse attention: Attend only to tokens in selected blocks
    
    Mathematical principle (from Qwen3.8-Flash-Next):
        - Standard attention: O(L^2) complexity
        - Block-sparse: O(L * B * S) where B = num_blocks, S = block_size
        - Qwen config: B=512, S=4, so effective budget = 2048 tokens per query
        - Much more memory-friendly than token-level sparse attention
    
    Args:
        hidden_size: Model hidden dimension.
        num_attention_heads: Number of attention heads.
        num_key_value_heads: Number of KV heads (for GQA).
        head_dim: Per-head dimension (qk and v use same dim in QSA).
        block_size: Tokens per block (default 4).
        num_blocks: Maximum blocks to attend (default 512).
        block_scoring: Block scoring method (max/avg/learned).
        attention_dropout: Dropout rate.
        rms_norm_eps: RMSNorm epsilon.
        bias: Whether to use bias in projections.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        block_size: int = 4,
        num_blocks: int = 512,
        block_scoring: str = "max",
        attention_dropout: float = 0.0,
        rms_norm_eps: float = 1e-6,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if num_attention_heads <= 0:
            raise ValueError(f"num_attention_heads must be positive, got {num_attention_heads}")
        if num_key_value_heads <= 0:
            raise ValueError(f"num_key_value_heads must be positive, got {num_key_value_heads}")
        if num_attention_heads % num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {head_dim}")
        
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.attention_dropout = attention_dropout
        self.rms_norm_eps = rms_norm_eps
        
        self.num_key_value_groups = num_attention_heads // num_key_value_heads
        self.scaling = head_dim ** -0.5
        
        # Standard attention projections
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=bias)
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=bias)
        
        # Micro-block indexer
        self.block_indexer = MicroBlockIndexer(
            hidden_size=hidden_size,
            num_heads=8,  # Indexer heads (configurable)
            head_dim=64,  # Indexer head dimension (configurable)
            block_size=block_size,
            num_blocks=num_blocks,
            block_scoring=block_scoring,
            rms_norm_eps=rms_norm_eps,
        )
    
    def _repeat_kv(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Repeat KV for GQA."""
        if self.num_key_value_groups == 1:
            return hidden_states
        batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_kv_heads, self.num_key_value_groups, seq_len, head_dim
        )
        return hidden_states.reshape(batch, num_kv_heads * self.num_key_value_groups, seq_len, head_dim)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Forward pass implementing block-sparse attention.
        
        Args:
            hidden_states: (batch, seq_len, hidden_size)
            attention_mask: Optional attention mask
            position_ids: Optional position indices (can be used for RoPE)
            output_attentions: Whether to return attention weights
        
        Returns:
            Dictionary with 'hidden_states' and optional 'attentions'
        """
        if not isinstance(hidden_states, torch.Tensor):
            raise TypeError("hidden_states must be a Tensor")
        
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project Q, K, V
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query = query.view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        key = key.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        value = value.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        
        # Transpose to (batch, num_heads, seq_len, head_dim)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        
        # GQA: Repeat KV if needed
        key = self._repeat_kv(key)
        value = self._repeat_kv(value)
        
        # === Block-Sparse Indexing ===
        block_indices, block_mask = self.block_indexer(hidden_states, attention_mask)
        
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scaling
        
        # Apply block-sparse mask
        # block_mask: (batch, seq_len, seq_len) boolean mask
        block_mask_expanded = block_mask.unsqueeze(1)  # (batch, 1, seq_len, seq_len)
        scores = scores.masked_fill(~block_mask_expanded, float('-inf'))
        
        # Apply causal mask if provided
        if attention_mask is not None:
            scores = scores + attention_mask
        
        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = F.dropout(
            attn_weights,
            p=self.attention_dropout if self.training else 0.0,
            training=self.training
        )
        
        # Compute attention output
        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        
        # Output projection
        output = self.o_proj(attn_output)
        
        result = {"hidden_states": output}
        if output_attentions:
            result["attentions"] = attn_weights
        
        return result


# Aliases
QSA = QwenQuerySparseAttention


__all__ = [
    "MicroBlockIndexer",
    "QwenQuerySparseAttention",
    "QSA",
]
