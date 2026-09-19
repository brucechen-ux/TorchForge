"""DeepSeek-V4.1 CSA2 reference attention (report sections 2.2--2.4.4).

Pure PyTorch components: shared-key/value MQA, layer-local SWA, learned
non-overlapping compression, cross-layer reuse, and optional hierarchical
selection. Quantization emulates dequantized values, not packed storage.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from torchforge.common.nn import RMSNorm

from .csa2_hierarchical_indexer import HierarchicalSparseIndexer, _apply_rope, _validate_positions
from .csa2_quantization import fake_quantize_fp4, fake_quantize_swa
from .indexer import _validate_hidden_states, _validate_positive_int, _validate_rotary_factor
from .mla import GroupedLinear


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    value = x.float()
    value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)
    return (value * weight.float()).to(x.dtype)


def _validate_sequence(hidden_states: torch.Tensor, position_ids: torch.Tensor, hidden_size: int) -> None:
    _validate_hidden_states("hidden_states", hidden_states, hidden_size)
    _validate_positions("position_ids", position_ids, tuple(hidden_states.shape[:2]))
    if hidden_states.shape[1] == 0:
        raise ValueError("A CSA2 input chunk must contain at least one token.")
    if position_ids.device != hidden_states.device:
        raise ValueError("position_ids and hidden_states must be on the same device.")
    if position_ids.shape[1] > 1 and not torch.all(position_ids[:, 1:] == position_ids[:, :-1] + 1):
        raise ValueError("CSA2 accepts contiguous, unpadded sequences; split packed samples into separate calls.")


def _check_continuation(positions: torch.Tensor, next_position: Optional[torch.Tensor], name: str) -> None:
    if next_position is not None:
        if positions.device != next_position.device or not torch.equal(positions[:, 0], next_position):
            raise ValueError(f"{name} state does not match the batch or next contiguous position; use a fresh state.")


class CSA2Mode(Enum):
    FULL = "full"
    REINDEX = "reindex"
    REUSE = "reuse"


@dataclass
class CSA2LayerState:
    """Per-layer, per-request temporal state; pass explicitly for chunked decode.

    Full layers own global KV and incomplete compression blocks. Every layer
    owns its SWA tail. Cross-layer sharing is handled separately by SharedCache.
    Tensor graphs are preserved; inference callers should use no_grad().
    """

    main_kv: Optional[torch.Tensor] = None
    indexer_k: Optional[torch.Tensor] = None
    compressed_position_ids: Optional[torch.Tensor] = None
    block_end_position_ids: Optional[torch.Tensor] = None
    pending_hidden: Optional[torch.Tensor] = None
    pending_positions: Optional[torch.Tensor] = None
    next_global_position: Optional[torch.Tensor] = None
    swa_kv: Optional[torch.Tensor] = None
    swa_position_ids: Optional[torch.Tensor] = None
    next_swa_position: Optional[torch.Tensor] = None
    layer_idx: Optional[int] = None


class CSA2SharedCache:
    """Cross-layer states for one request/micro-batch, never a process-global cache.

    main_kv [B,1,C,D] is used as BOTH key and value; indexer_k [B,1,C,I]
    has one shared head. Routes [B,S,K] and candidates [B,S,P] are query-specific.
    Start positions determine RoPE, end positions determine causal visibility.
    """

    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        self.main_kv: Optional[torch.Tensor] = None
        self.indexer_k: Optional[torch.Tensor] = None
        self.top_k_indices: Optional[torch.Tensor] = None
        self.candidate_pool_positions: Optional[torch.Tensor] = None
        self.compressed_position_ids: Optional[torch.Tensor] = None
        self.block_end_position_ids: Optional[torch.Tensor] = None
        self.query_position_ids: Optional[torch.Tensor] = None
        self.main_source_layer_idx: Optional[int] = None
        self.index_source_layer_idx: Optional[int] = None
        self.source_layer_idx: Optional[int] = None
        self.signature: Optional[Tuple] = None

    def update_main_kv(
        self, main_kv: torch.Tensor, indexer_k: torch.Tensor,
        compressed_position_ids: torch.Tensor, layer_idx: int, *,
        block_end_position_ids: torch.Tensor, signature: Tuple,
    ) -> None:
        self.main_kv = main_kv
        self.indexer_k = indexer_k
        self.compressed_position_ids = compressed_position_ids
        self.block_end_position_ids = block_end_position_ids
        self.main_source_layer_idx = layer_idx
        self.source_layer_idx = layer_idx
        self.signature = signature
        # Old routes and candidate pools cannot be used against a new KV source.
        self.top_k_indices = self.candidate_pool_positions = self.query_position_ids = None
        self.index_source_layer_idx = None

    def update_indices(self, top_k_indices: torch.Tensor, layer_idx: int, *, position_ids: torch.Tensor) -> None:
        self.top_k_indices = top_k_indices
        self.index_source_layer_idx = self.source_layer_idx = layer_idx
        self.query_position_ids = position_ids.clone()


class CSA2Compressor(nn.Module):
    """CSA2 global branch with static Full, Reindex or Reuse responsibilities.

    Decoder Full layers must set is_decoder=True and receive encoder_hidden_states
    (CED report eq. 1). Encoder Full layers compress their own inputs. The return
    value is (shared_kv [B,1,C,D], indices [B,S,K]); -1 denotes an invalid route.
    Quantization can be disabled to study the full-precision architecture.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        compress_rate: int,
        top_k: int,
        q_lora_rank: int,
        index_num_heads: int,
        index_head_dim: int,
        partial_rotary_factor: float = 0.125,
        rope_theta: float = 160000.0,
        rms_norm_eps: float = 1e-20,
        mode: CSA2Mode = CSA2Mode.FULL,
        shared_cache: Optional[CSA2SharedCache] = None,
        is_decoder: bool = False,
        hierarchical: bool = False,
        block_size: int = 8,
        num_candidate_blocks: int = 2048,
        pin_recent_block: bool = False,
        quantize: bool = True,
        original_seq_len: int = 0,
        rope_factor: float = 1.0,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
    ) -> None:
        super().__init__()
        for name, value in (("hidden_size", hidden_size), ("num_attention_heads", num_attention_heads),
                            ("num_key_value_heads", num_key_value_heads), ("head_dim", head_dim),
                            ("compress_rate", compress_rate), ("top_k", top_k), ("q_lora_rank", q_lora_rank),
                            ("index_num_heads", index_num_heads), ("index_head_dim", index_head_dim)):
            _validate_positive_int(name, value)
        if num_key_value_heads != 1:
            raise ValueError("DeepSeek-V4.1 CSA2 uses shared-key/value MQA: num_key_value_heads must be 1.")
        _validate_rotary_factor("partial_rotary_factor", partial_rotary_factor, head_dim)
        if not math.isfinite(rope_theta) or rope_theta <= 1:
            raise ValueError("rope_theta must be finite and greater than 1.")
        if not math.isfinite(rms_norm_eps) or rms_norm_eps <= 0:
            raise ValueError("rms_norm_eps must be finite and positive.")
        if not isinstance(original_seq_len, int) or original_seq_len < 0:
            raise ValueError("original_seq_len must be a nonnegative integer.")
        if not math.isfinite(rope_factor) or rope_factor < 1:
            raise ValueError("rope_factor must be finite and >= 1.")
        if not (math.isfinite(beta_fast) and math.isfinite(beta_slow) and beta_fast >= beta_slow > 0):
            raise ValueError("YaRN requires finite beta_fast >= beta_slow > 0.")
        rope_dim = int(head_dim * partial_rotary_factor)
        if rope_dim > index_head_dim:
            raise ValueError("The same RoPE dimension must fit both main and indexer heads.")
        try:
            mode = CSA2Mode(mode)
        except ValueError as exc:
            raise ValueError("mode must be full, reindex, or reuse.") from exc
        if mode != CSA2Mode.FULL and shared_cache is None:
            raise ValueError("Reindex and Reuse modes require a shared_cache.")
        if hierarchical and not is_decoder:
            raise ValueError("The V4.1 hierarchical indexer is used only in decoder layers.")
        if is_decoder and compress_rate != 1:
            raise ValueError("V4.1 CED decoder global KV uses compress_rate=1.")
        if quantize and (head_dim % 32 or index_head_dim % 32):
            raise ValueError("Quantized main/SWA and index dimensions must be divisible by 32.")
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = 1
        self.head_dim = head_dim
        self.compress_rate = compress_rate
        self.top_k = top_k
        self.q_lora_rank = q_lora_rank
        self.index_num_heads = index_num_heads
        self.index_head_dim = index_head_dim
        self.rope_dim = rope_dim
        self.partial_rotary_factor = partial_rotary_factor
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.mode = mode
        self.shared_cache = shared_cache if shared_cache is not None else CSA2SharedCache()
        self.is_decoder = is_decoder
        self.hierarchical = hierarchical
        self.quantize = quantize
        self.rope_options = dict(original_seq_len=original_seq_len, factor=rope_factor,
                                 beta_fast=beta_fast, beta_slow=beta_slow)
        self.signature = (head_dim, index_head_dim, compress_rate, rope_dim, rope_theta,
                          original_seq_len, rope_factor, beta_fast, beta_slow, quantize, is_decoder)
        if mode == CSA2Mode.FULL:
            self.kv_proj = nn.Linear(hidden_size, head_dim, bias=False)
            if compress_rate > 1:
                self.gate_proj = nn.Linear(hidden_size, head_dim, bias=False)
            self.kv_norm_weight = nn.Parameter(torch.ones(head_dim))
            self.indexer_k_proj = nn.Linear(head_dim, index_head_dim, bias=False)
            self.indexer_k_norm_weight = nn.Parameter(torch.ones(index_head_dim))
        self.indexer = None
        if mode != CSA2Mode.REUSE:
            self.indexer = HierarchicalSparseIndexer(
                hidden_size=hidden_size, q_lora_rank=q_lora_rank,
                index_num_heads=index_num_heads, index_head_dim=index_head_dim,
                top_k=top_k, block_size=block_size, num_candidate_blocks=num_candidate_blocks,
                pin_recent_block=pin_recent_block,
                rope_dim=rope_dim, rope_theta=rope_theta, rms_norm_eps=rms_norm_eps,
                quantize=quantize, original_seq_len=original_seq_len,
                rope_factor=rope_factor, beta_fast=beta_fast, beta_slow=beta_slow,
            )

    def _rope(self, x: torch.Tensor, positions: torch.Tensor, *, inverse: bool = False) -> torch.Tensor:
        return _apply_rope(x, positions, self.rope_dim, self.rope_theta,
                           inverse=inverse, **self.rope_options)

    def prepare_global(
        self, hidden_states: torch.Tensor, position_ids: torch.Tensor, layer_idx: int,
        *, state: Optional[CSA2LayerState] = None,
    ) -> None:
        """Project a Full layer's global cache without running Q, indexing or SWA.

        For CED pass final encoder states here. Later forward(reuse_global=True)
        can run decoder/replay queries against these caches without overwriting KV.
        """
        if self.mode != CSA2Mode.FULL:
            raise RuntimeError("Only a Full layer can prepare global KV.")
        if not isinstance(layer_idx, int) or layer_idx < 0:
            raise ValueError("layer_idx must be a nonnegative integer.")
        _validate_sequence(hidden_states, position_ids, self.hidden_size)
        if state is not None:
            if state.layer_idx not in (None, layer_idx):
                raise ValueError("CSA2LayerState belongs to a different layer.")
            _check_continuation(position_ids, state.next_global_position, "Global KV")
        elif (position_ids[:, 0] % self.compress_rate != 0).any():
            raise ValueError("A fresh compression sequence must start at a block boundary.")
        if state is not None and state.next_global_position is None and (position_ids[:, 0] % self.compress_rate != 0).any():
            raise ValueError("A fresh compression sequence must start at a block boundary.")
        source = hidden_states
        source_positions = position_ids
        if state is not None and state.pending_hidden is not None:
            source = torch.cat((state.pending_hidden, source), dim=1)
            source_positions = torch.cat((state.pending_positions, source_positions), dim=1)
        batch, length = source.shape[:2]
        count = length // self.compress_rate
        usable = count * self.compress_rate
        x = source[:, :usable]
        if self.compress_rate > 1:
            # Per-token/channel gating, no overlap, no absolute compression bias.
            with torch.autocast(device_type=x.device.type, enabled=False):
                values = F.linear(x.float(), self.kv_proj.weight.float())
                gates = F.linear(x.float(), self.gate_proj.weight.float())
                values = values.reshape(batch, count, self.compress_rate, self.head_dim)
                gates = gates.reshape_as(values)
                latent = (values * gates.softmax(dim=2)).sum(dim=2).to(hidden_states.dtype)
        else:
            latent = self.kv_proj(x)
        latent = _rms_norm(latent, self.kv_norm_weight, self.rms_norm_eps)
        starts = source_positions[:, :usable:self.compress_rate]
        ends = source_positions[:, self.compress_rate - 1:usable:self.compress_rate]
        # The index key comes from normalized, UNROTATED main latent, before FP4.
        indexer_k = self.indexer_k_proj(latent)
        indexer_k = _rms_norm(indexer_k, self.indexer_k_norm_weight, self.rms_norm_eps)
        indexer_k = self._rope(indexer_k.unsqueeze(1), starts)
        main_kv = self._rope(latent.unsqueeze(1), starts)
        if self.quantize:
            indexer_k = fake_quantize_fp4(indexer_k, block_size=32, scale_format="e8m0")
            main_kv = fake_quantize_fp4(main_kv, block_size=16, scale_format="e4m3")
        if state is not None:
            if state.main_kv is not None:
                main_kv = torch.cat((state.main_kv, main_kv), dim=2)
                indexer_k = torch.cat((state.indexer_k, indexer_k), dim=2)
                starts = torch.cat((state.compressed_position_ids, starts), dim=1)
                ends = torch.cat((state.block_end_position_ids, ends), dim=1)
            state.main_kv, state.indexer_k = main_kv, indexer_k
            state.compressed_position_ids, state.block_end_position_ids = starts, ends
            state.pending_hidden = source[:, usable:].clone()
            state.pending_positions = source_positions[:, usable:].clone()
            state.next_global_position = position_ids[:, -1] + 1
            state.layer_idx = layer_idx
        self.shared_cache.update_main_kv(main_kv, indexer_k, starts, layer_idx,
                                        block_end_position_ids=ends, signature=self.signature)

    def forward(
        self, hidden_states: torch.Tensor, q_residual: torch.Tensor,
        position_ids: torch.Tensor, layer_idx: int, *,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_position_ids: Optional[torch.Tensor] = None,
        state: Optional[CSA2LayerState] = None,
        reuse_global: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _validate_sequence(hidden_states, position_ids, self.hidden_size)
        if not isinstance(layer_idx, int) or layer_idx < 0:
            raise ValueError("layer_idx must be a nonnegative integer.")
        _validate_hidden_states("q_residual", q_residual, self.q_lora_rank)
        if q_residual.shape[:2] != hidden_states.shape[:2] or q_residual.device != hidden_states.device:
            raise ValueError("q_residual must match the query batch, sequence and device.")
        if self.mode != CSA2Mode.FULL and (encoder_hidden_states is not None or encoder_position_ids is not None or reuse_global):
            raise ValueError("Only Full layers accept encoder inputs or reuse_global.")
        if reuse_global and (encoder_hidden_states is not None or encoder_position_ids is not None):
            raise ValueError("reuse_global reads existing KV; omit encoder inputs.")
        if self.mode == CSA2Mode.FULL and not reuse_global:
            if self.is_decoder:
                if encoder_hidden_states is None or encoder_position_ids is None:
                    raise ValueError("CED Full layers require final encoder_hidden_states and encoder_position_ids.")
                source, source_positions = encoder_hidden_states, encoder_position_ids
            else:
                if encoder_hidden_states is not None or encoder_position_ids is not None:
                    raise ValueError("Encoder CSA2 must compress its own hidden states.")
                source, source_positions = hidden_states, position_ids
            self.prepare_global(source, source_positions, layer_idx, state=state)
        cache = self.shared_cache
        if cache.main_kv is None or cache.indexer_k is None or cache.signature != self.signature:
            raise RuntimeError("Missing or incompatible global KV; run the matching Full layer first.")
        if cache.main_kv.shape[0] != hidden_states.shape[0] or cache.main_kv.device != hidden_states.device:
            raise ValueError("The shared cache belongs to a different batch or device.")
        if self.mode == CSA2Mode.FULL:
            if cache.main_source_layer_idx != layer_idx:
                raise RuntimeError("reuse_global requires this Full layer's own cache.")
        elif cache.main_source_layer_idx >= layer_idx:
            raise RuntimeError("Shared global KV must come from a preceding Full layer.")
        if self.mode == CSA2Mode.REUSE:
            if cache.top_k_indices is None or cache.query_position_ids is None:
                raise RuntimeError("Reuse requires indices from a Full or Reindex layer.")
            if cache.index_source_layer_idx >= layer_idx or not torch.equal(cache.query_position_ids, position_ids):
                raise ValueError("Reuse requires a preceding index source for these exact query positions.")
            if cache.top_k_indices.shape[-1] != self.top_k:
                raise ValueError("Reuse top_k must match the index-producing layer.")
            return cache.main_kv, cache.top_k_indices
        if self.mode == CSA2Mode.REINDEX and self.hierarchical:
            if cache.candidate_pool_positions is None or not torch.equal(cache.query_position_ids, position_ids):
                raise RuntimeError("Hierarchical Reindex requires the Full layer's pool for these queries.")
            indices = self.indexer.forward_reindex_mode(
                q_residual, cache.indexer_k, hidden_states=hidden_states,
                position_ids=position_ids, key_end_position_ids=cache.block_end_position_ids,
                candidate_positions=cache.candidate_pool_positions,
            )
        else:
            indices, pool = self.indexer.forward_full_mode(
                q_residual, cache.indexer_k, hidden_states=hidden_states,
                position_ids=position_ids, key_end_position_ids=cache.block_end_position_ids,
                build_candidate_pool=self.mode == CSA2Mode.FULL and self.hierarchical,
            )
            if self.mode == CSA2Mode.FULL:
                cache.candidate_pool_positions = pool
        cache.update_indices(indices, layer_idx, position_ids=position_ids)
        return cache.main_kv, indices


class CSA2Attention(nn.Module):
    """V4.1 attention over concatenated selected global KV and independent local KV.

    q_residual may be supplied by an external normalized low-rank query path;
    otherwise the layer computes it internally. KV is shared as key AND value.
    One softmax includes both branches and a learned zero-value sink per head.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        q_lora_rank: int,
        compressor: CSA2Compressor,
        window_size: int = 128,
        o_groups: int = 8,
        o_lora_rank: int = 1024,
    ) -> None:
        super().__init__()
        for name, value in (("window_size", window_size), ("o_groups", o_groups), ("o_lora_rank", o_lora_rank)):
            _validate_positive_int(name, value)
        for name, value in (("hidden_size", hidden_size), ("num_attention_heads", num_attention_heads),
                            ("num_key_value_heads", num_key_value_heads), ("head_dim", head_dim),
                            ("q_lora_rank", q_lora_rank)):
            if getattr(compressor, name) != value:
                raise ValueError(f"Attention {name} must match its compressor.")
        if num_attention_heads % o_groups:
            raise ValueError("num_attention_heads must be divisible by o_groups.")
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.q_lora_rank = q_lora_rank
        self.compressor = compressor
        self.window_size = window_size
        self.o_groups = o_groups
        self.q_a_proj = nn.Linear(hidden_size, q_lora_rank, bias=False)
        self.q_a_norm = RMSNorm(q_lora_rank, eps=compressor.rms_norm_eps)
        self.q_proj = nn.Linear(q_lora_rank, num_attention_heads * head_dim, bias=False)
        self.swa_kv_proj = nn.Linear(hidden_size, head_dim, bias=False)
        self.swa_norm_weight = nn.Parameter(torch.ones(head_dim))
        self.sink_logits = nn.Parameter(torch.zeros(num_attention_heads))
        self.o_a_proj = GroupedLinear(num_attention_heads * head_dim // o_groups,
                                      o_groups * o_lora_rank, o_groups)
        self.o_b_proj = nn.Linear(o_groups * o_lora_rank, hidden_size, bias=False)

    def forward(
        self, hidden_states: torch.Tensor, q_residual: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None, layer_idx: int = 0, *,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_position_ids: Optional[torch.Tensor] = None,
        state: Optional[CSA2LayerState] = None,
        reuse_global: bool = False,
    ) -> torch.Tensor:
        if position_ids is None:
            raise ValueError("CSA2Attention requires explicit position_ids.")
        _validate_sequence(hidden_states, position_ids, self.hidden_size)
        if state is not None:
            if state.layer_idx not in (None, layer_idx):
                raise ValueError("CSA2LayerState belongs to a different layer.")
            _check_continuation(position_ids, state.next_swa_position, "SWA")
        batch, seq_len = hidden_states.shape[:2]
        if q_residual is None:
            q_residual = self.q_a_norm(self.q_a_proj(hidden_states))
        main_kv, indices = self.compressor(
            hidden_states, q_residual, position_ids, layer_idx,
            encoder_hidden_states=encoder_hidden_states, encoder_position_ids=encoder_position_ids,
            state=state, reuse_global=reuse_global,
        )
        query = self.q_proj(q_residual).view(batch, seq_len, self.num_attention_heads, self.head_dim)
        query = self.compressor._rope(query.transpose(1, 2), position_ids).transpose(1, 2)
        local = _rms_norm(self.swa_kv_proj(hidden_states), self.swa_norm_weight,
                          self.compressor.rms_norm_eps)
        local = self.compressor._rope(local.unsqueeze(1), position_ids).squeeze(1)
        if self.compressor.quantize:
            local = fake_quantize_swa(local)
        local_positions = position_ids
        if state is not None and state.swa_kv is not None:
            local = torch.cat((state.swa_kv, local), dim=1)
            local_positions = torch.cat((state.swa_position_ids, local_positions), dim=1)
        past_length = local.shape[1] - seq_len
        # Only gather W local and K global entries per query, with no head copies.
        offsets = torch.arange(self.window_size, device=local.device)
        local_indices = past_length + torch.arange(seq_len, device=local.device)[:, None] - offsets
        local_valid = local_indices >= 0
        selected_local = local[:, local_indices.clamp_min(0)]
        # Sentinel makes early queries and empty global caches safe.
        global_values = F.pad(main_kv[:, 0], (0, 0, 0, 1))
        safe = torch.where(indices >= 0, indices, main_kv.shape[2])
        batch_indices = torch.arange(batch, device=local.device)[:, None, None]
        selected_global = global_values[batch_indices, safe]
        values = torch.cat((selected_local, selected_global), dim=2)
        valid = torch.cat((local_valid[None].expand(batch, -1, -1), indices >= 0), dim=-1)
        with torch.autocast(device_type=query.device.type, enabled=False):
            logits = torch.einsum("bshd,bskd->bshk", query.float(), values.float()) * self.head_dim ** -0.5
            logits = logits.masked_fill(~valid.unsqueeze(2), float("-inf"))
            sinks = self.sink_logits.float().view(1, 1, -1, 1).expand(batch, seq_len, -1, -1)
            probabilities = torch.cat((logits, sinks), dim=-1).softmax(dim=-1)[..., :-1]
            output = torch.einsum("bshk,bskd->bshd", probabilities, values.float()).to(query.dtype)
        output = self.compressor._rope(output.transpose(1, 2), position_ids, inverse=True).transpose(1, 2)
        grouped = output.reshape(batch, seq_len, self.o_groups, -1)
        result = self.o_b_proj(self.o_a_proj(grouped).flatten(2))
        if state is not None:
            # Keep only the local history needed by the next query.
            keep = min(self.window_size - 1, local.shape[1])
            state.swa_kv = (local[:, -keep:] if keep else local[:, :0]).clone()
            state.swa_position_ids = (local_positions[:, -keep:] if keep else local_positions[:, :0]).clone()
            state.next_swa_position = position_ids[:, -1] + 1
            state.layer_idx = layer_idx
        return result


__all__ = ["CSA2Mode", "CSA2LayerState", "CSA2SharedCache", "CSA2Compressor", "CSA2Attention"]
