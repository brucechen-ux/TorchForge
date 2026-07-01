from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

from .indexer import (
    _apply_rotary_pos_emb,
    _rms_norm,
    _rotary_embeddings,
    _validate_hidden_states,
    _validate_position_ids,
    _validate_positive_float,
    _validate_positive_int,
    _validate_rotary_factor,
)


class HCACompressor(nn.Module):
    """Compress hidden states into heavily compressed KV entries.

    Args:
        hidden_size: Size of the input hidden-state dimension.
        head_dim: Dimension of each compressed KV entry.
        compress_rate: Number of source tokens represented by each compressed KV entry.
        partial_rotary_factor: Fraction of ``head_dim`` that receives compressed RoPE.
        rope_theta: RoPE theta used for compressed positions.
        rms_norm_eps: Epsilon used by RMS normalization.

    Forward:
        ``hidden_states`` has shape ``(batch, sequence_length, hidden_size)``.
        ``position_ids`` has shape ``(batch, sequence_length)``.

    Returns:
        ``compressed_kv`` with shape ``(batch, 1, compressed_length, head_dim)`` and
        ``block_bias`` with shape ``(batch, 1, sequence_length, compressed_length)``.
        ``block_bias`` is ``None`` when no compressed causal bias is needed.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        head_dim: int,
        compress_rate: int,
        partial_rotary_factor: float,
        rope_theta: float,
        rms_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        _validate_positive_int("hidden_size", hidden_size)
        _validate_positive_int("head_dim", head_dim)
        _validate_positive_int("compress_rate", compress_rate)
        _validate_rotary_factor("partial_rotary_factor", partial_rotary_factor, head_dim)
        _validate_positive_float("rope_theta", rope_theta)
        _validate_positive_float("rms_norm_eps", rms_norm_eps)
        self.hidden_size = hidden_size
        self.compress_rate = compress_rate
        self.head_dim = head_dim
        self.partial_rotary_factor = partial_rotary_factor
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.kv_proj = nn.Linear(hidden_size, head_dim, bias=False)
        self.gate_proj = nn.Linear(hidden_size, head_dim, bias=False)
        self.position_bias = nn.Parameter(torch.empty(compress_rate, head_dim))
        self.kv_norm_weight = nn.Parameter(torch.ones(head_dim))

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        _validate_hidden_states("hidden_states", hidden_states, self.hidden_size)
        _validate_position_ids(position_ids, hidden_states.shape[:-1])
        batch = hidden_states.shape[0]
        kv = self.kv_proj(hidden_states)
        gate = self.gate_proj(hidden_states)
        usable = (kv.shape[1] // self.compress_rate) * self.compress_rate
        chunk_kv, chunk_gate = kv[:, :usable], gate[:, :usable]
        if chunk_kv.shape[1] > 0:
            n_windows = chunk_kv.shape[1] // self.compress_rate
            chunk_kv = chunk_kv.view(batch, n_windows, self.compress_rate, -1)
            chunk_gate = chunk_gate.view(batch, n_windows, self.compress_rate, -1) + self.position_bias
            compressed = _rms_norm(
                (chunk_kv * chunk_gate.softmax(dim=2, dtype=torch.float32).to(chunk_kv.dtype)).sum(dim=2),
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
        seq_len = position_ids.shape[1]
        if seq_len == 1 or compressed_len == 0:
            return compressed_kv, None

        entry_indices = torch.arange(compressed_len, device=compressed_kv.device)
        causal_threshold = (position_ids + 1) // self.compress_rate
        block_bias = compressed_kv.new_zeros((batch, 1, seq_len, compressed_len))
        block_bias = block_bias.masked_fill(
            entry_indices.view(1, 1, 1, -1) >= causal_threshold.unsqueeze(1).unsqueeze(-1),
            float("-inf"),
        )
        return compressed_kv, block_bias
