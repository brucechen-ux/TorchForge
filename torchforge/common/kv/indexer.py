from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


def _rotate_half_interleaved(x: torch.Tensor) -> torch.Tensor:
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


def _apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos.repeat_interleave(2, dim=-1).unsqueeze(1)
    sin = sin.repeat_interleave(2, dim=-1).unsqueeze(1)
    rope_dim = cos.shape[-1]
    nope, rope = x[..., :-rope_dim], x[..., -rope_dim:]
    rotated = (rope.float() * cos) + (_rotate_half_interleaved(rope).float() * sin)
    return torch.cat([nope, rotated.to(x.dtype)], dim=-1)


def _stable_topk_indices(scores: torch.Tensor, k: int, dim: int = -1) -> torch.Tensor:
    return torch.argsort(scores, dim=dim, descending=True, stable=True).narrow(dim, 0, k)


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    input_dtype = x.dtype
    x_fp32 = x.to(torch.float32)
    x_fp32 = x_fp32 * torch.rsqrt(x_fp32.square().mean(-1, keepdim=True) + eps)
    return weight * x_fp32.to(input_dtype)


def _rotary_embeddings(
    x: torch.Tensor,
    *,
    position_ids: torch.Tensor,
    head_dim: int,
    partial_rotary_factor: float,
    theta: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    dim = int(head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=x.device) / dim))
    inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
    position_ids_expanded = position_ids[:, None, :].float()
    freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
    return freqs.cos().to(dtype=x.dtype), freqs.sin().to(dtype=x.dtype)


class _IndexerScorer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, head_dim: int) -> None:
        super().__init__()
        self.softmax_scale = head_dim**-0.5
        self.weights_scaling = num_heads**-0.5
        self.weights_proj = nn.Linear(hidden_size, num_heads, bias=False)

    def forward(self, query: torch.Tensor, compressed_kv: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        scores = torch.matmul(query.float(), compressed_kv.transpose(-1, -2).float().unsqueeze(1))
        scores = F.relu(scores) * self.softmax_scale
        weights = self.weights_proj(hidden_states).float() * self.weights_scaling
        return (scores * weights.unsqueeze(-1)).sum(dim=2)


class CompressedKVIndexer(nn.Module):
    """Select compressed KV entries for sparse attention.

    Args:
        hidden_size: Size of the input hidden-state dimension.
        q_lora_rank: Size of the query residual consumed by the index query projection.
        num_heads: Number of index scoring heads.
        head_dim: Per-head dimension used by the compressed indexer.
        top_k: Number of compressed KV entries selected for each token.
        compress_rate: Number of source tokens represented by each compressed KV entry.
        partial_rotary_factor: Fraction of ``head_dim`` that receives compressed RoPE.
        rope_theta: RoPE theta used for compressed positions.
        rms_norm_eps: Epsilon used by RMS normalization.
        topk: Deprecated alias for ``top_k``.

    Forward:
        ``hidden_states`` has shape ``(..., sequence_length, hidden_size)``.
        ``q_residual`` has shape ``(..., sequence_length, q_lora_rank)``.
        ``position_ids`` has shape ``(..., sequence_length)``.

    Returns:
        Integer selected-entry indices with shape ``(..., sequence_length, top_k)``.
        Invalid causal selections are filled with ``-1``.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        q_lora_rank: int,
        num_heads: int,
        head_dim: int,
        compress_rate: int,
        partial_rotary_factor: float,
        rope_theta: float,
        top_k: Optional[int] = None,
        rms_norm_eps: float = 1e-6,
        topk: Optional[int] = None,
    ) -> None:
        super().__init__()
        if top_k is None:
            if topk is None:
                raise TypeError("CompressedKVIndexer missing required argument: top_k.")
            top_k = topk
        elif topk is not None and topk != top_k:
            raise ValueError("Pass only one of top_k or topk, or pass matching values.")
        _validate_positive_int("hidden_size", hidden_size)
        _validate_positive_int("q_lora_rank", q_lora_rank)
        _validate_positive_int("num_heads", num_heads)
        _validate_positive_int("head_dim", head_dim)
        _validate_positive_int("top_k", top_k)
        _validate_positive_int("compress_rate", compress_rate)
        _validate_rotary_factor("partial_rotary_factor", partial_rotary_factor, head_dim)
        _validate_positive_float("rope_theta", rope_theta)
        _validate_positive_float("rms_norm_eps", rms_norm_eps)
        self.compress_rate = compress_rate
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.top_k = top_k
        self.partial_rotary_factor = partial_rotary_factor
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.hidden_size = hidden_size
        self.q_lora_rank = q_lora_rank
        self.kv_proj = nn.Linear(hidden_size, 2 * head_dim, bias=False)
        self.gate_proj = nn.Linear(hidden_size, 2 * head_dim, bias=False)
        self.position_bias = nn.Parameter(torch.zeros(compress_rate, 2 * head_dim))
        self.kv_norm_weight = nn.Parameter(torch.ones(head_dim))
        self.q_b_proj = nn.Linear(q_lora_rank, num_heads * head_dim, bias=False)
        self.scorer = _IndexerScorer(hidden_size, num_heads, head_dim)

    def _compress(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch = hidden_states.shape[0]
        kv = self.kv_proj(hidden_states)
        gate = self.gate_proj(hidden_states)
        usable = (kv.shape[1] // self.compress_rate) * self.compress_rate
        chunk_kv, chunk_gate = kv[:, :usable], gate[:, :usable]
        if chunk_kv.shape[1] == 0:
            return chunk_kv.new_zeros((batch, 0, self.head_dim))

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
        return _apply_rotary_pos_emb(compressed.unsqueeze(1), cos, sin).squeeze(1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        q_residual: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        _validate_hidden_states("hidden_states", hidden_states, self.hidden_size)
        _validate_hidden_states("q_residual", q_residual, self.q_lora_rank)
        if hidden_states.shape[:-1] != q_residual.shape[:-1]:
            raise ValueError(
                "q_residual must match hidden_states except for the last dimension, "
                f"got {tuple(q_residual.shape)} and {tuple(hidden_states.shape)}."
            )
        _validate_position_ids(position_ids, hidden_states.shape[:-1])
        batch, seq_len = hidden_states.shape[:2]
        compressed_kv = self._compress(hidden_states)
        compressed_len = compressed_kv.shape[1]
        if compressed_len == 0:
            return position_ids.new_empty((batch, seq_len, 0))

        cos_q, sin_q = _rotary_embeddings(
            hidden_states,
            position_ids=position_ids,
            head_dim=self.head_dim,
            partial_rotary_factor=self.partial_rotary_factor,
            theta=self.rope_theta,
        )
        query = self.q_b_proj(q_residual).view(batch, seq_len, -1, self.head_dim).transpose(1, 2)
        query = _apply_rotary_pos_emb(query, cos_q, sin_q).transpose(1, 2)
        index_scores = self.scorer(query, compressed_kv, hidden_states)
        top_k = min(self.top_k, compressed_len)
        causal_threshold = (position_ids + 1) // self.compress_rate
        entry_indices = torch.arange(compressed_len, device=index_scores.device)
        future_mask = entry_indices.view(1, 1, -1) >= causal_threshold.unsqueeze(-1)
        index_scores = index_scores.masked_fill(future_mask, float("-inf"))
        top_k_indices = _stable_topk_indices(index_scores, top_k, dim=-1)
        invalid = top_k_indices >= causal_threshold.unsqueeze(-1)
        return torch.where(invalid, torch.full_like(top_k_indices, -1), top_k_indices)


def _validate_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive int, got {value!r}.")


def _validate_positive_float(name: str, value: float) -> None:
    if not isinstance(value, (float, int)) or float(value) <= 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}.")


def _validate_rotary_factor(name: str, value: float, head_dim: int) -> None:
    _validate_positive_float(name, value)
    if float(value) > 1.0:
        raise ValueError(f"{name} must be in the range (0, 1], got {value!r}.")
    rotary_dim = int(head_dim * float(value))
    if rotary_dim <= 0:
        raise ValueError(f"head_dim * {name} must be at least 1.")
    if rotary_dim % 2 != 0:
        raise ValueError(f"head_dim * {name} must be even, got {rotary_dim}.")


def _validate_hidden_states(name: str, value: torch.Tensor, hidden_size: int) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}.")
    if value.dim() != 3:
        raise ValueError(f"{name} must have shape (batch, sequence_length, hidden_size).")
    if value.shape[-1] != hidden_size:
        raise ValueError(f"{name} last dimension must be {hidden_size}, got {value.shape[-1]}.")


def _validate_position_ids(position_ids: torch.Tensor, expected_shape: Tuple[int, ...]) -> None:
    if not isinstance(position_ids, torch.Tensor):
        raise TypeError(f"position_ids must be a torch.Tensor, got {type(position_ids).__name__}.")
    if tuple(position_ids.shape) != tuple(expected_shape):
        raise ValueError(f"position_ids must have shape {tuple(expected_shape)}, got {tuple(position_ids.shape)}.")
