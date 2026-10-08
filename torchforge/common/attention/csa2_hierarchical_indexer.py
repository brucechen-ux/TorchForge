from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .csa2_quantization import fake_quantize_fp4
from .indexer import _stable_topk_indices, _validate_hidden_states, _validate_positive_int
from .rotary import apply_rotary_interleaved_single


def _apply_rope(
    x: torch.Tensor,
    position_ids: torch.Tensor,
    rope_dim: int,
    theta: float,
    *,
    original_seq_len: int = 0,
    factor: float = 1.0,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
    inverse: bool = False,
) -> torch.Tensor:
    """Trailing, interleaved RoPE; optional YaRN follows the released V4.1 code."""
    freqs = theta ** (-torch.arange(0, rope_dim, 2, device=x.device).float() / rope_dim)
    if original_seq_len > 0:
        def corrected_dim(rotations: float) -> float:
            return rope_dim * math.log(original_seq_len / (rotations * 2 * math.pi)) / (2 * math.log(theta))

        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), rope_dim - 1)
        ramp = ((torch.arange(rope_dim // 2, device=x.device) - low) / max(high - low, 1e-3)).clamp(0, 1)
        freqs = freqs * (1 - ramp) + freqs / factor * ramp
    angles = position_ids.float().unsqueeze(-1) * freqs
    sin = angles.sin()
    return apply_rotary_interleaved_single(x, angles.cos(), -sin if inverse else sin)


def _validate_positions(name: str, positions: torch.Tensor, shape: Tuple[int, int]) -> None:
    if not isinstance(positions, torch.Tensor) or tuple(positions.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}.")
    if positions.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"{name} must contain integer positions.")
    if (positions < 0).any():
        raise ValueError(f"{name} must be nonnegative.")


def _select_indices(scores: torch.Tensor, top_k: int, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Fixed-width routes; masked, missing and padded positions become -1."""
    width = min(top_k, scores.shape[-1])
    selected = _stable_topk_indices(scores, width)
    valid = torch.isfinite(scores.gather(-1, selected))
    if positions is not None:
        selected = positions.gather(-1, selected)
        valid = valid & (selected >= 0)
    selected = selected.masked_fill(~valid, -1)
    return F.pad(selected, (0, top_k - width), value=-1)


class HierarchicalSparseIndexer(nn.Module):
    """Weighted ReLU scoring, shared across heads, with per-query candidate pools.

    Inputs: hidden states [B,S,D], query residual [B,S,R], a SINGLE shared key
    head [B,1,C,I], query positions [B,S], and block-end positions [B,C].
    Outputs: routes [B,S,K] and candidate positions [B,S,P], using -1 for padding.
    Reindex accepts a pool explicitly, so different layers keep independent Q
    and weight projections. No full-context score matrix is built on that path.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        q_lora_rank: int,
        index_num_heads: int,
        index_head_dim: int,
        top_k: int,
        block_size: int = 8,
        num_candidate_blocks: int = 2048,
        rope_dim: Optional[int] = None,
        rope_theta: float = 160000.0,
        rms_norm_eps: float = 1e-20,
        quantize: bool = True,
        pin_recent_block: bool = False,
        original_seq_len: int = 0,
        rope_factor: float = 1.0,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
    ) -> None:
        super().__init__()
        for name, value in (("hidden_size", hidden_size), ("q_lora_rank", q_lora_rank),
                            ("index_num_heads", index_num_heads), ("index_head_dim", index_head_dim),
                            ("top_k", top_k), ("block_size", block_size),
                            ("num_candidate_blocks", num_candidate_blocks)):
            _validate_positive_int(name, value)
        if top_k > block_size * num_candidate_blocks:
            raise ValueError("top_k cannot exceed the candidate pool capacity.")
        rope_dim = min(64, index_head_dim) if rope_dim is None else rope_dim
        if rope_dim <= 0 or rope_dim % 2 or rope_dim > index_head_dim:
            raise ValueError("rope_dim must be positive, even, and <= index_head_dim.")
        if not math.isfinite(rope_theta) or rope_theta <= 1 or rms_norm_eps <= 0:
            raise ValueError("rope_theta must exceed 1 and rms_norm_eps must be positive.")
        if original_seq_len < 0 or not math.isfinite(rope_factor) or rope_factor < 1:
            raise ValueError("original_seq_len must be nonnegative and rope_factor must be >= 1.")
        if not (math.isfinite(beta_fast) and math.isfinite(beta_slow) and beta_fast >= beta_slow > 0):
            raise ValueError("YaRN requires finite beta_fast >= beta_slow > 0.")
        if quantize and index_head_dim % 32:
            raise ValueError("Quantized index_head_dim must be divisible by 32.")
        self.hidden_size = hidden_size
        self.q_lora_rank = q_lora_rank
        self.index_num_heads = index_num_heads
        self.index_head_dim = index_head_dim
        self.top_k = top_k
        self.block_size = block_size
        self.num_candidate_blocks = num_candidate_blocks
        self.rope_dim = rope_dim
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.quantize = quantize
        self.pin_recent_block = pin_recent_block
        self.rope_options = dict(original_seq_len=original_seq_len, factor=rope_factor,
                                 beta_fast=beta_fast, beta_slow=beta_slow)
        self.indexer_q_proj = nn.Linear(q_lora_rank, index_num_heads * index_head_dim, bias=False)
        self.weights_proj = nn.Linear(hidden_size, index_num_heads, bias=False)
        self.candidate_pool_positions: Optional[torch.Tensor] = None
        self.candidate_query_positions: Optional[torch.Tensor] = None
        self.indexer_q_proj.muon_head_shape = (index_num_heads, index_head_dim, q_lora_rank)

    @staticmethod
    def distillation_loss(scores: torch.Tensor, target_attention_weights: torch.Tensor) -> torch.Tensor:
        """计算有效 query 的平均 KL；teacher 支持 [B,S,K] 或 [B,S,H,K]。"""
        if scores.ndim != 3 or not scores.is_floating_point():
            raise ValueError("scores must be a floating-point tensor with shape (batch, sequence, keys).")
        target = target_attention_weights.detach().float()
        if target.ndim == 4:
            target = target.sum(dim=2)
        if target.shape != scores.shape or target.device != scores.device:
            raise ValueError("Teacher attention must match the indexer score shape and device.")
        if not torch.isfinite(target).all() or (target < 0).any():
            raise ValueError("Teacher attention must be finite and nonnegative.")
        if torch.isnan(scores).any() or torch.isposinf(scores).any():
            raise ValueError("Indexer scores may contain only finite values or negative infinity.")
        visible = torch.isfinite(scores)
        target = target.masked_fill(~visible, 0)
        mass = target.sum(dim=-1, keepdim=True)
        active = mass.squeeze(-1) > 0
        if not active.any():
            return scores.masked_fill(~visible, 0).sum() * 0
        target = target[active] / mass[active]
        log_prediction = scores.float()[active].log_softmax(dim=-1)
        positive = target > 0
        log_prediction = log_prediction.masked_fill(~positive, 0)
        log_target = target.clamp_min(torch.finfo(target.dtype).tiny).log()
        return (target * (log_target - log_prediction)).sum(dim=-1).mean()

    def _query(self, q_residual: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        batch, seq_len = q_residual.shape[:2]
        q = self.indexer_q_proj(q_residual).view(batch, seq_len, self.index_num_heads, self.index_head_dim)
        q = _apply_rope(q.transpose(1, 2), position_ids, self.rope_dim,
                        self.rope_theta, **self.rope_options).transpose(1, 2)
        if self.quantize:
            q = fake_quantize_fp4(q, block_size=32, scale_format="e8m0")
        return q

    def scores(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        indexer_k: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        key_end_position_ids: torch.Tensor,
        candidate_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Differentiable logits for indexer training; hard top-k itself has no gradient."""
        _validate_hidden_states("hidden_states", hidden_states, self.hidden_size)
        _validate_hidden_states("q_residual", q_residual, self.q_lora_rank)
        batch, seq_len = hidden_states.shape[:2]
        if q_residual.shape[:2] != (batch, seq_len):
            raise ValueError("q_residual must match the hidden-state batch and sequence dimensions.")
        if indexer_k.ndim != 4 or indexer_k.shape[:2] != (batch, 1) or indexer_k.shape[-1] != self.index_head_dim:
            raise ValueError("indexer_k must have shape (batch, 1, compressed_length, index_head_dim).")
        length = indexer_k.shape[2]
        _validate_positions("position_ids", position_ids, (batch, seq_len))
        _validate_positions("key_end_position_ids", key_end_position_ids, (batch, length))
        tensors = (q_residual, indexer_k, position_ids, key_end_position_ids)
        if any(t.device != hidden_states.device for t in tensors):
            raise ValueError("Indexer inputs must be on the same device.")
        q = self._query(q_residual, position_ids)
        weights = self.weights_proj(hidden_states).float()
        weights = weights * (self.index_head_dim ** -0.5 * self.index_num_heads ** -0.5)
        if candidate_positions is None:
            with torch.autocast(device_type=q.device.type, enabled=False):
                logits = torch.einsum("bshd,bcd->bshc", q.float(), indexer_k[:, 0].float())
            scores = (logits.relu() * weights.unsqueeze(-1)).sum(dim=2)
            visible = key_end_position_ids[:, None, :] <= position_ids[:, :, None]
        else:
            if candidate_positions.ndim != 3 or candidate_positions.shape[:2] != (batch, seq_len):
                raise ValueError("candidate_positions must have shape (batch, query_length, pool_size).")
            if candidate_positions.dtype != torch.long or candidate_positions.device != hidden_states.device:
                raise ValueError("candidate_positions must be int64 on the input device.")
            if ((candidate_positions < -1) | (candidate_positions >= length)).any():
                raise ValueError("Candidate position is outside the shared KV cache.")
            # Append a sentinel to safely gather -1, including when length == 0.
            keys = F.pad(indexer_k[:, 0], (0, 0, 0, 1))
            ends = F.pad(key_end_position_ids, (0, 1))
            safe = torch.where(candidate_positions >= 0, candidate_positions, length)
            batch_indices = torch.arange(batch, device=keys.device)[:, None, None]
            gathered = keys[batch_indices, safe]
            with torch.autocast(device_type=q.device.type, enabled=False):
                logits = torch.einsum("bshd,bspd->bshp", q.float(), gathered.float())
            scores = (logits.relu() * weights.unsqueeze(-1)).sum(dim=2)
            visible = (candidate_positions >= 0) & (ends[batch_indices, safe] <= position_ids[:, :, None])
        return scores.masked_fill(~visible, float("-inf"))

    def _blockwise_candidate_selection(self, scores: torch.Tensor) -> torch.Tensor:
        batch, seq_len, length = scores.shape
        if length == 0:
            return torch.empty((batch, seq_len, 0), dtype=torch.long, device=scores.device)
        padded = F.pad(scores, (0, -length % self.block_size), value=float("-inf"))
        blocks = padded.unflatten(-1, (-1, self.block_size)).amax(-1)
        if self.pin_recent_block:
            # Released reference detail: retain the newest causally visible block.
            entries = torch.arange(length, device=scores.device)
            last = torch.where(torch.isfinite(scores), entries, -1).amax(-1)
            block_ids = torch.arange(blocks.shape[-1], device=scores.device)
            newest = (last[..., None] >= 0) & (block_ids == last[..., None] // self.block_size)
            blocks = blocks.masked_fill(newest, float("inf"))
        count = min(self.num_candidate_blocks, blocks.shape[-1])
        chosen = _stable_topk_indices(blocks, count)
        reachable = blocks.gather(-1, chosen) > -float("inf")
        offsets = torch.arange(self.block_size, device=scores.device)
        pool = (chosen[..., None] * self.block_size + offsets).flatten(-2)
        safe = pool.clamp_max(length - 1)
        valid = reachable.repeat_interleave(self.block_size, -1) & (pool < length)
        valid = valid & torch.isfinite(scores.gather(-1, safe))
        return pool.masked_fill(~valid, -1)

    def forward_full_mode(
        self, q_residual: torch.Tensor, indexer_k: torch.Tensor, *,
        hidden_states: torch.Tensor, position_ids: torch.Tensor,
        key_end_position_ids: torch.Tensor, build_candidate_pool: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        scores = self.scores(hidden_states, q_residual, indexer_k,
                             position_ids=position_ids, key_end_position_ids=key_end_position_ids)
        pool = self._blockwise_candidate_selection(scores) if build_candidate_pool else None
        self.candidate_pool_positions = pool
        self.candidate_query_positions = position_ids.clone() if pool is not None else None
        # The Full layer's own selection is over ALL positions, not its candidate pool.
        return _select_indices(scores, self.top_k), pool

    def forward_reindex_mode(
        self, q_residual: torch.Tensor, indexer_k: torch.Tensor, *,
        hidden_states: torch.Tensor, position_ids: torch.Tensor,
        key_end_position_ids: torch.Tensor,
        candidate_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if candidate_positions is None:
            if self.candidate_pool_positions is None or self.candidate_query_positions is None:
                raise RuntimeError("Reindex requires a candidate pool from the Full layer.")
            if not torch.equal(position_ids, self.candidate_query_positions):
                raise ValueError("Cached candidate pool belongs to different query positions.")
            candidate_positions = self.candidate_pool_positions
        scores = self.scores(hidden_states, q_residual, indexer_k, position_ids=position_ids,
                             key_end_position_ids=key_end_position_ids, candidate_positions=candidate_positions)
        return _select_indices(scores, self.top_k, candidate_positions)

    def clear_cache(self) -> None:
        self.candidate_pool_positions = None
        self.candidate_query_positions = None


__all__ = ["HierarchicalSparseIndexer"]
