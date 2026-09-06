"""Qwen3.8-Flash-Next N-gram Embedding for lookup-based capacity scaling."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class NgramEmbedding(nn.Module):
    """N-gram lookup table for capacity scaling without extra compute.
    
    Qwen3.8-Flash-Next innovation: Add ~51B parameters via bigram/trigram lookup
    that provides memory-based capacity without increasing FLOPs significantly.
    
    Mathematical principle:
        Standard embedding: e_i = Lookup(token_i)
        N-gram embedding: e_i = Lookup(token_i) + Lookup(token_{i-1}, token_i) + Lookup(token_{i-2}, token_{i-1}, token_i)
        
        This adds contextual embeddings based on local token history.
    
    Architecture (Qwen config):
        - Base table size: 20,000,000 entries
        - ~51B total parameters for bigram + trigram
        - Injected at Layer 2 (not input embedding layer)
        - Can be offloaded to host memory with async prefetch
    
    Key properties:
        1. Deterministic lookup (hash-based indexing)
        2. Memory-bound, not compute-bound
        3. Captures local n-gram patterns directly
        4. Complements learned transformer representations
    
    Args:
        vocab_size: Vocabulary size for token IDs.
        hidden_size: Embedding dimension size.
        max_ngram: Maximum n-gram order (2 for bigram, 3 for trigram).
        table_size: Hash table size for n-gram embeddings.
        hash_function: Hash function for n-gram to index mapping.
        dropout: Dropout rate for n-gram embeddings.
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        max_ngram: int = 3,
        table_size: int = 20_000_000,
        hash_function: str = "murmur",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if max_ngram < 2:
            raise ValueError(f"max_ngram must be at least 2, got {max_ngram}")
        if max_ngram > 4:
            raise ValueError(f"max_ngram > 4 not supported, got {max_ngram}")
        if table_size <= 0:
            raise ValueError(f"table_size must be positive, got {table_size}")
        if hash_function not in ("murmur", "simple"):
            raise ValueError(f"hash_function must be murmur or simple, got {hash_function}")
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_ngram = max_ngram
        self.table_size = table_size
        self.hash_function = hash_function
        self.dropout = dropout
        
        # N-gram embedding tables (2-gram to max_ngram)
        # Each n-gram order gets its own table
        self.ngram_tables = nn.ModuleDict()
        for n in range(2, max_ngram + 1):
            # Each n-gram table maps hash indices to embeddings
            self.ngram_tables[str(n)] = nn.Embedding(table_size, hidden_size)
        
        # Learnable weights for combining different n-gram orders
        self.ngram_weights = nn.Parameter(torch.ones(max_ngram - 1))
        
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        """Initialize n-gram embedding tables."""
        for table in self.ngram_tables.values():
            nn.init.normal_(table.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.ngram_weights, 1.0 / (self.max_ngram - 1))
    
    def _hash_ngram_simple(self, ngram: torch.Tensor) -> torch.Tensor:
        """Simple polynomial hash for n-gram.
        
        Args:
            ngram: (..., n) tensor of token IDs
        
        Returns:
            hash_idx: (...,) tensor of hash indices in [0, table_size)
        """
        # Polynomial rolling hash: hash = (a[0] * p^(n-1) + a[1] * p^(n-2) + ... + a[n-1]) % table_size
        n = ngram.shape[-1]
        p = 31  # Prime base
        
        hash_val = torch.zeros_like(ngram[..., 0])
        power = 1
        
        for i in range(n - 1, -1, -1):
            hash_val = (hash_val + ngram[..., i] * power) % self.table_size
            power = (power * p) % self.table_size
        
        return hash_val
    
    def _hash_ngram_murmur(self, ngram: torch.Tensor) -> torch.Tensor:
        """MurmurHash-inspired hash for n-gram (simplified PyTorch version).
        
        Args:
            ngram: (..., n) tensor of token IDs
        
        Returns:
            hash_idx: (...,) tensor of hash indices in [0, table_size)
        """
        # Simplified MurmurHash for better distribution
        n = ngram.shape[-1]
        
        # Combine n-gram tokens with mixing
        h = torch.zeros_like(ngram[..., 0], dtype=torch.long)
        c1 = 0xcc9e2d51
        c2 = 0x1b873593
        
        for i in range(n):
            k = ngram[..., i].long()
            k = (k * c1) & 0xFFFFFFFF
            k = ((k << 15) | (k >> 17)) & 0xFFFFFFFF
            k = (k * c2) & 0xFFFFFFFF
            
            h = h ^ k
            h = ((h << 13) | (h >> 19)) & 0xFFFFFFFF
            h = (h * 5 + 0xe6546b64) & 0xFFFFFFFF
        
        # Finalization
        h = h ^ n
        h = h ^ (h >> 16)
        h = (h * 0x85ebca6b) & 0xFFFFFFFF
        h = h ^ (h >> 13)
        h = (h * 0xc2b2ae35) & 0xFFFFFFFF
        h = h ^ (h >> 16)
        
        # Modulo to table size
        return h % self.table_size
    
    def _get_ngram_indices(
        self,
        input_ids: torch.Tensor,
        n: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract n-grams and compute their hash indices.
        
        Args:
            input_ids: (batch, seq_len) token IDs
            n: N-gram order (2, 3, 4, ...)
        
        Returns:
            hash_indices: (batch, seq_len) hash indices for each position
            valid_mask: (batch, seq_len) boolean mask for valid n-grams
        """
        batch_size, seq_len = input_ids.shape
        
        # Pad input_ids to handle boundary (pad with vocab_size as special marker)
        padding = torch.full(
            (batch_size, n - 1),
            self.vocab_size,
            dtype=input_ids.dtype,
            device=input_ids.device
        )
        padded_ids = torch.cat([padding, input_ids], dim=1)
        
        # Extract n-grams using sliding window
        # For position i, n-gram is [i-(n-1), ..., i-1, i]
        ngrams = []
        for offset in range(n):
            ngrams.append(padded_ids[:, offset:offset + seq_len])
        ngrams = torch.stack(ngrams, dim=-1)  # (batch, seq_len, n)
        
        # Compute hash indices
        if self.hash_function == "murmur":
            hash_indices = self._hash_ngram_murmur(ngrams)
        else:  # simple
            hash_indices = self._hash_ngram_simple(ngrams)
        
        # Valid mask: positions where all n tokens are real (not padding)
        valid_mask = (ngrams != self.vocab_size).all(dim=-1)
        
        return hash_indices, valid_mask
    
    def forward(
        self,
        input_ids: torch.Tensor,
        return_dict: bool = True,
    ) -> torch.Tensor | dict:
        """Compute n-gram embeddings for input tokens.
        
        Args:
            input_ids: (batch, seq_len) tensor of token IDs
            return_dict: If True, return dict with per-order embeddings
        
        Returns:
            ngram_embeddings: (batch, seq_len, hidden_size) combined n-gram embeddings
            Or dict with per-order embeddings if return_dict=True
        """
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError(f"input_ids must be a Tensor, got {type(input_ids).__name__}")
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be 2D (batch, seq_len), got shape {input_ids.shape}")
        
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Initialize combined embedding
        combined_embedding = torch.zeros(
            (batch_size, seq_len, self.hidden_size),
            dtype=torch.float32,
            device=device
        )
        
        ngram_components = {}
        
        # Process each n-gram order
        weights = F.softmax(self.ngram_weights, dim=0)
        
        for idx, n in enumerate(range(2, self.max_ngram + 1)):
            # Get hash indices for n-grams
            hash_indices, valid_mask = self._get_ngram_indices(input_ids, n)
            
            # Lookup embeddings from n-gram table
            table = self.ngram_tables[str(n)]
            ngram_emb = table(hash_indices)  # (batch, seq_len, hidden_size)
            
            # Mask invalid positions (e.g., first few tokens don't have full context)
            ngram_emb = ngram_emb * valid_mask.unsqueeze(-1).float()
            
            # Apply dropout
            if self.training and self.dropout > 0:
                ngram_emb = F.dropout(ngram_emb, p=self.dropout)
            
            # Weight and accumulate
            weighted_emb = weights[idx] * ngram_emb
            combined_embedding = combined_embedding + weighted_emb
            
            ngram_components[f"{n}gram"] = ngram_emb
        
        if return_dict:
            return {
                "ngram_embedding": combined_embedding,
                "components": ngram_components,
            }
        else:
            return combined_embedding


class NgramEmbeddingLayer(nn.Module):
    """N-gram embedding injection layer for Qwen architecture.
    
    This module is inserted at a specific layer (e.g., Layer 2 in Qwen)
    to inject n-gram embeddings into the hidden states.
    
    Args:
        vocab_size: Vocabulary size.
        hidden_size: Model hidden dimension.
        max_ngram: Maximum n-gram order (default 3 for trigram).
        table_size: N-gram hash table size (default 20M).
        hash_function: Hash function (murmur or simple).
        dropout: Dropout rate.
        injection_mode: How to inject - "add" or "concat_project".
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        max_ngram: int = 3,
        table_size: int = 20_000_000,
        hash_function: str = "murmur",
        dropout: float = 0.0,
        injection_mode: str = "add",
    ) -> None:
        super().__init__()
        if injection_mode not in ("add", "concat_project"):
            raise ValueError(f"injection_mode must be add or concat_project, got {injection_mode}")
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.injection_mode = injection_mode
        
        # N-gram embedding module
        self.ngram_embedding = NgramEmbedding(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            max_ngram=max_ngram,
            table_size=table_size,
            hash_function=hash_function,
            dropout=dropout,
        )
        
        # Projection if using concat mode
        if injection_mode == "concat_project":
            self.projection = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        else:
            self.register_parameter("projection", None)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Inject n-gram embeddings into hidden states.
        
        Args:
            hidden_states: (batch, seq_len, hidden_size) current hidden states
            input_ids: (batch, seq_len) original token IDs
        
        Returns:
            enhanced_hidden_states: (batch, seq_len, hidden_size)
        """
        if not isinstance(hidden_states, torch.Tensor):
            raise TypeError("hidden_states must be a Tensor")
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError("input_ids must be a Tensor")
        
        # Get n-gram embeddings
        ngram_emb = self.ngram_embedding(input_ids, return_dict=False)
        
        # Inject into hidden states
        if self.injection_mode == "add":
            # Simple addition (Qwen's approach)
            return hidden_states + ngram_emb
        else:  # concat_project
            # Concatenate and project back to hidden_size
            concatenated = torch.cat([hidden_states, ngram_emb], dim=-1)
            return self.projection(concatenated)


__all__ = [
    "NgramEmbedding",
    "NgramEmbeddingLayer",
]
