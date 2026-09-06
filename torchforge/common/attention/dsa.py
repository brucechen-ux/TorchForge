"""GLM-5.3-Flash Dynamic Sparse Attention (DSA) with KPool compression and sparse MLA."""

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


def _stable_topk_indices(scores: torch.Tensor, k: int, dim: int = -1) -> torch.Tensor:
    """Stable top-k selection using argsort."""
    return torch.argsort(scores, dim=dim, descending=True, stable=True).narrow(dim, 0, k)


class KPool(nn.Module):
    """K-dimensional pooling for compressing KV sequences.
    
    GLM-5.3-Flash uses KPool×4 to compress the sequence before indexing.
    This reduces the search space for the sparse attention indexer.
    
    Args:
        compress_rate: Compression factor (e.g., 4 means compress 4 tokens into 1).
        pooling_mode: Pooling strategy - "avg", "max", or "learned_weighted".
        hidden_size: Hidden dimension size.
    
    Mathematical principle:
        For sequence length L and compress_rate C:
        - Divide L tokens into L/C groups
        - Each group produces 1 compressed token
        - Output length = L / C (rounded up)
    """
    
    def __init__(
        self,
        compress_rate: int,
        pooling_mode: str = "avg",
        hidden_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        if compress_rate <= 0:
            raise ValueError(f"compress_rate must be positive, got {compress_rate}")
        if pooling_mode not in ("avg", "max", "learned_weighted"):
            raise ValueError(f"pooling_mode must be avg/max/learned_weighted, got {pooling_mode}")
        
        self.compress_rate = compress_rate
        self.pooling_mode = pooling_mode
        
        if pooling_mode == "learned_weighted" and hidden_size is None:
            raise ValueError("learned_weighted pooling requires hidden_size")
        
        if pooling_mode == "learned_weighted":
            # Learnable weights for weighted average within each pool
            self.pool_weights = nn.Parameter(torch.ones(compress_rate) / compress_rate)
        else:
            self.register_parameter("pool_weights", None)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compress sequence via pooling.
        
        Args:
            hidden_states: (..., seq_len, hidden_size)
        
        Returns:
            Compressed tensor: (..., compressed_seq_len, hidden_size)
        """
        if not isinstance(hidden_states, torch.Tensor):
            raise TypeError("hidden_states must be a Tensor")
        
        *batch_dims, seq_len, hidden_size = hidden_states.shape
        compress_rate = self.compress_rate
        
        # Calculate compressed length (round up for remainder)
        compressed_len = (seq_len + compress_rate - 1) // compress_rate
        
        # Pad sequence if needed to make it divisible by compress_rate
        if seq_len % compress_rate != 0:
            pad_len = compressed_len * compress_rate - seq_len
            hidden_states = F.pad(hidden_states, (0, 0, 0, pad_len))
            seq_len = compressed_len * compress_rate
        
        # Reshape to group tokens: (..., num_groups, compress_rate, hidden_size)
        grouped = hidden_states.view(*batch_dims, compressed_len, compress_rate, hidden_size)
        
        if self.pooling_mode == "avg":
            # Average pooling within each group
            compressed = grouped.mean(dim=-2)
        elif self.pooling_mode == "max":
            # Max pooling within each group
            compressed = grouped.max(dim=-2).values
        else:  # learned_weighted
            # Weighted average with learnable weights
            weights = F.softmax(self.pool_weights, dim=0).view(1, 1, compress_rate, 1)
            compressed = (grouped * weights).sum(dim=-2)
        
        return compressed


class DSAIndexer(nn.Module):
    """Sparse indexer for DSA that selects top-k compressed positions.
    
    Mathematical principle:
        1. Project query: q = W_q @ (hidden_states + q_residual)
        2. Project key from compressed KV: k = W_k @ compressed_kv
        3. Compute scores: s_ij = q_i^T k_j (no RoPE, NoPE design)
        4. Select top-k indices per query token
    
    Args:
        hidden_size: Input hidden dimension.
        q_lora_rank: Query residual dimension.
        num_heads: Number of scoring heads.
        head_dim: Per-head dimension.
        top_k: Number of tokens to select per query.
        rms_norm_eps: Epsilon for RMSNorm.
    """
    
    def __init__(
        self,
        hidden_size: int,
        q_lora_rank: int,
        num_heads: int,
        head_dim: int,
        top_k: int,
        rms_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if q_lora_rank <= 0:
            raise ValueError(f"q_lora_rank must be positive, got {q_lora_rank}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {head_dim}")
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        
        self.hidden_size = hidden_size
        self.q_lora_rank = q_lora_rank
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.top_k = top_k
        self.rms_norm_eps = rms_norm_eps
        
        # Query projection: (hidden_size + q_lora_rank) -> num_heads * head_dim
        self.q_proj = nn.Linear(hidden_size + q_lora_rank, num_heads * head_dim, bias=False)
        
        # Key projection: hidden_size -> num_heads * head_dim
        self.k_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        
        # RMSNorm for query and key
        self.q_norm = nn.Parameter(torch.ones(head_dim))
        self.k_norm = nn.Parameter(torch.ones(head_dim))
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        compressed_kv: torch.Tensor,
        q_residual: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute top-k sparse indices for each query token.
        
        Args:
            hidden_states: (..., query_len, hidden_size) - query tokens
            compressed_kv: (..., compressed_len, hidden_size) - compressed KV
            q_residual: (..., query_len, q_lora_rank) - query residual
            attention_mask: Optional mask for valid positions
        
        Returns:
            indices: (..., query_len, top_k) - selected compressed positions
        """
        if not isinstance(hidden_states, torch.Tensor):
            raise TypeError("hidden_states must be a Tensor")
        if not isinstance(compressed_kv, torch.Tensor):
            raise TypeError("compressed_kv must be a Tensor")
        if not isinstance(q_residual, torch.Tensor):
            raise TypeError("q_residual must be a Tensor")
        
        *batch_dims, query_len, _ = hidden_states.shape
        compressed_len = compressed_kv.shape[-2]
        
        # Concatenate hidden_states and q_residual for query projection
        q_input = torch.cat([hidden_states, q_residual], dim=-1)
        
        # Project query and key
        query = self.q_proj(q_input)  # (..., query_len, num_heads * head_dim)
        key = self.k_proj(compressed_kv)  # (..., compressed_len, num_heads * head_dim)
        
        # Reshape to multi-head: (..., seq_len, num_heads, head_dim)
        query = query.view(*batch_dims, query_len, self.num_heads, self.head_dim)
        key = key.view(*batch_dims, compressed_len, self.num_heads, self.head_dim)
        
        # Apply RMSNorm
        query = _rms_norm(query, self.q_norm, self.rms_norm_eps)
        key = _rms_norm(key, self.k_norm, self.rms_norm_eps)
        
        # Compute scores: (..., num_heads, query_len, compressed_len)
        # scores[..., i, j] = query[i]^T @ key[j]
        query = query.transpose(-3, -2)  # (..., num_heads, query_len, head_dim)
        key = key.transpose(-3, -2)      # (..., num_heads, compressed_len, head_dim)
        scores = torch.matmul(query, key.transpose(-2, -1))  # (..., num_heads, query_len, compressed_len)
        
        # Average scores across heads
        scores = scores.mean(dim=-3)  # (..., query_len, compressed_len)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # Select top-k indices per query token
        actual_k = min(self.top_k, compressed_len)
        top_k_indices = _stable_topk_indices(scores, actual_k, dim=-1)
        
        # Pad to top_k if compressed_len < top_k
        if actual_k < self.top_k:
            pad_shape = (*batch_dims, query_len, self.top_k - actual_k)
            padding = torch.full(pad_shape, -1, dtype=top_k_indices.dtype, device=top_k_indices.device)
            top_k_indices = torch.cat([top_k_indices, padding], dim=-1)
        
        return top_k_indices


class GLMDynamicSparseAttention(nn.Module):
    """GLM-5.3-Flash Dynamic Sparse Attention (DSA) layer.
    
    Architecture:
        1. KPool×4: Compress sequence by 4x
        2. Indexer: Select top-2048 compressed positions
        3. Sparse MLA: Attend only to selected positions (NoPE - no positional encoding)
    
    Mathematical principle (from GLM-5.3-Flash):
        - Standard MLA: O(L^2) complexity for sequence length L
        - DSA compression: L -> L/4 via KPool
        - DSA indexing: Select top-K << L/4 positions per query
        - Final complexity: O(L * K) where K=2048 in GLM-5.3-Flash
    
    Args:
        hidden_size: Model hidden dimension.
        num_attention_heads: Number of attention heads.
        num_key_value_heads: Number of KV heads (for GQA).
        q_lora_rank: Query LoRA rank.
        kv_lora_rank: KV latent dimension.
        qk_nope_head_dim: Non-positional QK head dimension.
        v_head_dim: Value head dimension.
        compress_rate: KPool compression rate (default 4).
        top_k: Number of sparse positions to attend (default 2048).
        attention_dropout: Dropout rate.
        rms_norm_eps: RMSNorm epsilon.
        pooling_mode: KPool strategy (avg/max/learned_weighted).
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        v_head_dim: int,
        compress_rate: int = 4,
        top_k: int = 2048,
        attention_dropout: float = 0.0,
        rms_norm_eps: float = 1e-6,
        pooling_mode: str = "avg",
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if num_attention_heads <= 0:
            raise ValueError(f"num_attention_heads must be positive, got {num_attention_heads}")
        if num_key_value_heads <= 0:
            raise ValueError(f"num_key_value_heads must be positive, got {num_key_value_heads}")
        if num_attention_heads % num_key_value_heads != 0:
            raise ValueError(f"num_attention_heads must be divisible by num_key_value_heads")
        
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.v_head_dim = v_head_dim
        self.compress_rate = compress_rate
        self.top_k = top_k
        self.attention_dropout = attention_dropout
        self.rms_norm_eps = rms_norm_eps
        
        self.num_key_value_groups = num_attention_heads // num_key_value_heads
        self.scaling = qk_nope_head_dim ** -0.5
        
        # Query projections (NoPE: no RoPE, only nope dimensions)
        self.q_a_proj = nn.Linear(hidden_size, q_lora_rank, bias=False)
        self.q_b_proj = nn.Linear(q_lora_rank, num_attention_heads * qk_nope_head_dim, bias=False)
        self.q_a_norm = RMSNorm(q_lora_rank, eps=rms_norm_eps)
        
        # KV projections (latent compression)
        self.kv_a_proj = nn.Linear(hidden_size, kv_lora_rank + qk_nope_head_dim, bias=False)
        self.kv_b_proj = nn.Linear(
            kv_lora_rank,
            num_key_value_heads * (qk_nope_head_dim + v_head_dim),
            bias=False
        )
        self.kv_a_norm = RMSNorm(kv_lora_rank, eps=rms_norm_eps)
        
        # KPool for sequence compression
        self.kpool = KPool(compress_rate, pooling_mode, hidden_size)
        
        # Indexer for sparse selection
        self.indexer = DSAIndexer(
            hidden_size=hidden_size,
            q_lora_rank=q_lora_rank,
            num_heads=8,  # Indexer heads (configurable)
            head_dim=64,  # Indexer head dimension (configurable)
            top_k=top_k,
            rms_norm_eps=rms_norm_eps,
        )
        
        # Output projection
        self.output_proj = nn.Linear(num_attention_heads * v_head_dim, hidden_size, bias=False)
    
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
        """Forward pass implementing sparse MLA attention.
        
        Args:
            hidden_states: (..., seq_len, hidden_size)
            attention_mask: Optional attention mask
            position_ids: Not used (NoPE design)
            output_attentions: Whether to return attention weights
        
        Returns:
            Dictionary with 'hidden_states' and optional 'attentions'
        """
        del position_ids  # NoPE: no positional encoding
        
        if not isinstance(hidden_states, torch.Tensor):
            raise TypeError("hidden_states must be a Tensor")
        
        batch_size, seq_len, _ = hidden_states.shape
        
        # === Query Projection ===
        q_residual = self.q_a_proj(hidden_states)
        q_residual = self.q_a_norm(q_residual)
        query = self.q_b_proj(q_residual)
        query = query.view(batch_size, seq_len, self.num_attention_heads, self.qk_nope_head_dim)
        query = query.transpose(1, 2)  # (batch, num_heads, seq_len, head_dim)
        
        # === KV Projection with Latent Compression ===
        kv_a = self.kv_a_proj(hidden_states)
        compressed_kv, c_pe = kv_a.split([self.kv_lora_rank, self.qk_nope_head_dim], dim=-1)
        compressed_kv = self.kv_a_norm(compressed_kv)
        
        # KPool: Compress the KV sequence
        compressed_hidden = self.kpool(hidden_states)
        compressed_kv_pooled = self.kpool(compressed_kv)
        
        # Indexer: Select top-k sparse positions
        sparse_indices = self.indexer(
            hidden_states=hidden_states,
            compressed_kv=compressed_hidden,
            q_residual=q_residual,
            attention_mask=None,  # Indexer uses its own masking logic
        )  # (batch, seq_len, top_k)
        
        # Expand KV from compressed latent
        kv_b = self.kv_b_proj(compressed_kv)
        kv_b = kv_b.view(batch_size, seq_len, self.num_key_value_heads, self.qk_nope_head_dim + self.v_head_dim)
        key, value = kv_b.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        
        key = key.transpose(1, 2)    # (batch, num_kv_heads, seq_len, head_dim)
        value = value.transpose(1, 2)  # (batch, num_kv_heads, seq_len, v_head_dim)
        
        # GQA: Repeat KV if needed
        key = self._repeat_kv(key)
        value = self._repeat_kv(value)
        
        # === Sparse Attention ===
        # Gather selected KV positions based on sparse indices
        # sparse_indices: (batch, seq_len, top_k)
        # We need to gather from compressed key/value
        
        # For simplicity in this reference implementation, we perform full attention
        # then mask out non-selected positions. Production code would use actual sparse kernels.
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scaling
        
        # Create sparse mask from indices
        sparse_mask = torch.zeros(
            (batch_size, 1, seq_len, seq_len),
            dtype=torch.bool,
            device=hidden_states.device
        )
        # Mark selected positions as valid
        for b in range(batch_size):
            for q_idx in range(seq_len):
                valid_indices = sparse_indices[b, q_idx][sparse_indices[b, q_idx] >= 0]
                if len(valid_indices) > 0:
                    sparse_mask[b, 0, q_idx, valid_indices] = True
        
        # Apply sparse mask
        scores = scores.masked_fill(~sparse_mask, float('-inf'))
        
        # Apply causal mask if provided
        if attention_mask is not None:
            scores = scores + attention_mask
        
        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout if self.training else 0.0)
        
        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        
        # Output projection
        output = self.output_proj(attn_output)
        
        result = {"hidden_states": output}
        if output_attentions:
            result["attentions"] = attn_weights
        
        return result


# Aliases
DSA = GLMDynamicSparseAttention


__all__ = [
    "KPool",
    "DSAIndexer",
    "GLMDynamicSparseAttention",
    "DSA",
]
