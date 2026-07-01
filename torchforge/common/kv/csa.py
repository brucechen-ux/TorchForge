from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

from .indexer import (
    CompressedKVIndexer,
    _apply_rotary_pos_emb,
    _rms_norm,
    _rotary_embeddings,
    _validate_hidden_states,
    _validate_position_ids,
    _validate_positive_float,
    _validate_positive_int,
    _validate_rotary_factor,
)


class CSACompressor(nn.Module):
    """Compress hidden states and build sparse block bias for compressed KV attention.

    Args:
        hidden_size: Size of the input hidden-state dimension.
        q_lora_rank: Size of the query residual consumed by the internal indexer.
        head_dim: Dimension of each compressed KV entry.
        index_num_heads: Number of index scoring heads.
        index_head_dim: Per-head dimension used by the internal indexer.
        index_top_k: Number of compressed KV entries selected for each token.
        compress_rate: Number of source tokens represented by each compressed KV entry.
        partial_rotary_factor: Fraction of ``head_dim`` that receives compressed RoPE.
        rope_theta: RoPE theta used for compressed positions.
        rms_norm_eps: Epsilon used by RMS normalization.
        indexer: Optional prebuilt ``CompressedKVIndexer``.
        index_topk: Deprecated alias for ``index_top_k``.

    Forward:
        ``hidden_states`` has shape ``(batch, sequence_length, hidden_size)``.
        ``q_residual`` has shape ``(batch, sequence_length, q_lora_rank)``.
        ``position_ids`` has shape ``(batch, sequence_length)``.

    Returns:
        ``compressed_kv`` with shape ``(batch, 1, compressed_length, head_dim)`` and
        ``block_bias`` with shape ``(batch, 1, sequence_length, compressed_length)``.
        ``block_bias`` is ``None`` when no compressed entries are available.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        q_lora_rank: int,
        head_dim: int,
        index_num_heads: int,
        index_head_dim: int,
        compress_rate: int,
        partial_rotary_factor: float,
        rope_theta: float,
        index_top_k: Optional[int] = None,
        rms_norm_eps: float = 1e-6,
        indexer: Optional[CompressedKVIndexer] = None,
        index_topk: Optional[int] = None,
    ) -> None:
        super().__init__()
        if index_top_k is None:
            if index_topk is None:
                raise TypeError("CSACompressor missing required argument: index_top_k.")
            index_top_k = index_topk
        elif index_topk is not None and index_topk != index_top_k:
            raise ValueError("Pass only one of index_top_k or index_topk, or pass matching values.")
        _validate_positive_int("hidden_size", hidden_size)
        _validate_positive_int("q_lora_rank", q_lora_rank)
        _validate_positive_int("head_dim", head_dim)
        _validate_positive_int("index_num_heads", index_num_heads)
        _validate_positive_int("index_head_dim", index_head_dim)
        _validate_positive_int("index_top_k", index_top_k)
        _validate_positive_int("compress_rate", compress_rate)
        _validate_rotary_factor("partial_rotary_factor", partial_rotary_factor, head_dim)
        _validate_positive_float("rope_theta", rope_theta)
        _validate_positive_float("rms_norm_eps", rms_norm_eps)
        _validate_rotary_factor("partial_rotary_factor", partial_rotary_factor, index_head_dim)
        self.hidden_size = hidden_size
        self.q_lora_rank = q_lora_rank
        self.compress_rate = compress_rate
        self.head_dim = head_dim
        self.partial_rotary_factor = partial_rotary_factor
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.kv_proj = nn.Linear(hidden_size, 2 * head_dim, bias=False)
        self.gate_proj = nn.Linear(hidden_size, 2 * head_dim, bias=False)
        self.position_bias = nn.Parameter(torch.empty(compress_rate, 2 * head_dim))
        self.kv_norm_weight = nn.Parameter(torch.ones(head_dim))
        self.indexer = indexer or CompressedKVIndexer(
            hidden_size=hidden_size,
            q_lora_rank=q_lora_rank,
            num_heads=index_num_heads,
            head_dim=index_head_dim,
            top_k=index_top_k,
            compress_rate=compress_rate,
            partial_rotary_factor=partial_rotary_factor,
            rope_theta=rope_theta,
            rms_norm_eps=rms_norm_eps,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        q_residual: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        _validate_hidden_states("hidden_states", hidden_states, self.hidden_size)
        _validate_hidden_states("q_residual", q_residual, self.q_lora_rank)
        if hidden_states.shape[:-1] != q_residual.shape[:-1]:
            raise ValueError(
                "q_residual must match hidden_states except for the last dimension, "
                f"got {tuple(q_residual.shape)} and {tuple(hidden_states.shape)}."
            )
        _validate_position_ids(position_ids, hidden_states.shape[:-1])
        batch, seq_len = hidden_states.shape[:2]
        kv = self.kv_proj(hidden_states)
        gate = self.gate_proj(hidden_states)
        usable = (kv.shape[1] // self.compress_rate) * self.compress_rate
        chunk_kv, chunk_gate = kv[:, :usable], gate[:, :usable]
        if chunk_kv.shape[1] > 0:
            n_windows = chunk_kv.shape[1] // self.compress_rate
            ratio = self.compress_rate
            chunk_kv = chunk_kv.view(batch, n_windows, ratio, -1)
            chunk_gate = chunk_gate.view(batch, n_windows, ratio, -1) + self.position_bias
            new_kv = chunk_kv.new_zeros((batch, n_windows, 2 * ratio, self.head_dim))
            new_gate = chunk_gate.new_full((batch, n_windows, 2 * ratio, self.head_dim), float("-inf"))
            new_kv[:, :, ratio:] = chunk_kv[..., self.head_dim :]
            new_gate[:, :, ratio:] = chunk_gate[..., self.head_dim :]
            if n_windows > 1:
                new_kv[:, 1:, :ratio] = chunk_kv[:, :-1, :, : self.head_dim]
                new_gate[:, 1:, :ratio] = chunk_gate[:, :-1, :, : self.head_dim]
            compressed = _rms_norm(
                (new_kv * new_gate.softmax(dim=2, dtype=torch.float32).to(new_kv.dtype)).sum(dim=2),
                self.kv_norm_weight,
                self.rms_norm_eps,
            )
            positions = torch.arange(n_windows, device=compressed.device)
            positions = (positions * self.compress_rate).unsqueeze(0).expand(batch, -1)
            cos, sin = _rotary_embeddings(
                compressed,
                position_ids=positions,
                head_dim=self.head_dim,
                partial_rotary_factor=self.partial_rotary_factor,
                theta=self.rope_theta,
            )
            compressed = _apply_rotary_pos_emb(compressed.unsqueeze(1), cos, sin).squeeze(1)
        else:
            compressed = chunk_kv.new_zeros((batch, 0, self.head_dim))

        compressed_kv = compressed.unsqueeze(1)
        compressed_len = compressed_kv.shape[2]
        if compressed_len == 0:
            return compressed_kv, None

        top_k_indices = self.indexer(hidden_states, q_residual=q_residual, position_ids=position_ids)
        valid = top_k_indices >= 0
        safe_indices = torch.where(valid, top_k_indices, torch.full_like(top_k_indices, compressed_len))
        block_bias = compressed_kv.new_full((batch, 1, seq_len, compressed_len + 1), float("-inf"))
        block_bias.scatter_(-1, safe_indices.unsqueeze(1), 0.0)
        return compressed_kv, block_bias[..., :compressed_len]
