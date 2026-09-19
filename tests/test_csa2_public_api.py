"""Behavioral specifications for the report-aligned CSA2 public API.

These cover the old Full/Reindex/Reuse, attention and hierarchical scenarios,
with corrected MQA shapes, explicit causal positions and numerical references.
"""

import pytest
import torch
import torch.nn.functional as F

from torchforge.common.attention import (
    CSA2Attention, CSA2Compressor, CSA2LayerState, CSA2Mode,
    CSA2SharedCache, HierarchicalSparseIndexer,
)
from torchforge.common.attention.csa2_quantization import (
    _e2m1, _e4m3, fake_quantize_fp4, fake_quantize_swa,
)


def config(**overrides):
    return dict(dict(hidden_size=16, num_attention_heads=4, num_key_value_heads=1,
                     head_dim=8, compress_rate=2, top_k=8, q_lora_rank=8,
                     index_num_heads=2, index_head_dim=8, partial_rotary_factor=0.5,
                     rms_norm_eps=1e-6, quantize=False), **overrides)


def inputs(length=7, start=0):
    return torch.randn(2, length, 16), torch.randn(2, length, 8), torch.arange(start, start + length).expand(2, -1)


def attention(compressor):
    return CSA2Attention(hidden_size=16, num_attention_heads=4, num_key_value_heads=1,
                         head_dim=compressor.head_dim, q_lora_rank=8, compressor=compressor,
                         window_size=3, o_groups=2, o_lora_rank=4)


def manual_rope(x, positions, dim=4, theta=160000.0, inverse=False):
    angles = positions.float()[..., None] * theta ** (-torch.arange(0, dim, 2).float() / dim)
    while angles.ndim < x.ndim:
        angles = angles.unsqueeze(-2)
    even, odd = x[..., -dim::2], x[..., -dim + 1::2]
    sine = angles.sin() * (-1 if inverse else 1)
    rotated = torch.stack((even * angles.cos() - odd * sine,
                           odd * angles.cos() + even * sine), dim=-1).flatten(-2)
    return torch.cat((x[..., :-dim], rotated), dim=-1)


def norm(x, weight, eps=1e-6):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * weight


def test_full_mode_compression_matches_learned_nonoverlapping_reference():
    torch.manual_seed(2)
    module = CSA2Compressor(**config())
    x, qr, pos = inputs()
    main, indices = module(x, qr, pos, 2)
    raw = F.linear(x[:, :6], module.kv_proj.weight).reshape(2, 3, 2, 8)
    gates = F.linear(x[:, :6], module.gate_proj.weight).reshape_as(raw)
    latent = norm((raw * gates.softmax(2)).sum(2), module.kv_norm_weight)
    expected_main = manual_rope(latent, pos[:, :6:2]).unsqueeze(1)
    expected_k = norm(F.linear(latent, module.indexer_k_proj.weight), module.indexer_k_norm_weight)
    expected_k = manual_rope(expected_k, pos[:, :6:2]).unsqueeze(1)
    torch.testing.assert_close(main, expected_main)
    torch.testing.assert_close(module.shared_cache.indexer_k, expected_k)
    assert main.shape == (2, 1, 3, 8)
    assert indices.shape == (2, 7, 8)
    assert not hasattr(module, "position_bias")
    assert (indices[:, 0] == -1).all()
    assert ((indices < (pos[..., None] + 1) // 2) | (indices == -1)).all()


def test_ratio_one_has_no_compression_gate():
    module = CSA2Compressor(**config(compress_rate=1))
    assert not hasattr(module, "gate_proj")
    x, qr, pos = inputs(1)
    main, idx = module(x, qr, pos, 0)
    torch.testing.assert_close(main[:, 0], norm(module.kv_proj(x), module.kv_norm_weight))
    assert (idx[..., 0] == 0).all()
    assert (idx[..., 1:] == -1).all()


def test_reindex_and_reuse_share_tensors_and_keep_distinct_owners():
    cache = CSA2SharedCache()
    full = CSA2Compressor(**config(), shared_cache=cache)
    reindex = CSA2Compressor(**config(), shared_cache=cache, mode=CSA2Mode.REINDEX)
    reuse = CSA2Compressor(**config(), shared_cache=cache, mode=CSA2Mode.REUSE)
    x, qr, pos = inputs()
    main, _ = full(x, qr, pos, 2)
    keys = cache.indexer_k
    main2, idx2 = reindex(x, qr + 1, pos, 3)
    main3, idx3 = reuse(x, qr, pos, 4)
    assert main is main2 is main3
    assert cache.indexer_k is keys
    assert idx3 is idx2
    assert cache.main_source_layer_idx == 2
    assert cache.index_source_layer_idx == 3
    assert not list(reuse.parameters())
    assert not hasattr(reindex, "kv_proj")
    with pytest.raises(ValueError, match="query positions"):
        reuse(x, qr, pos + 1, 4)
    cache.clear()
    with pytest.raises(RuntimeError, match="Missing"):
        reuse(x, qr, pos, 4)


def test_mqa_and_configuration_validation():
    with pytest.raises(ValueError, match="MQA"):
        CSA2Compressor(**config(num_key_value_heads=2))
    with pytest.raises(ValueError, match="shared_cache"):
        CSA2Compressor(**config(mode=CSA2Mode.REINDEX))
    with pytest.raises(ValueError, match="decoder"):
        CSA2Compressor(**config(hierarchical=True))
    with pytest.raises(ValueError, match="divisible by 32"):
        CSA2Compressor(**config(quantize=True))


def test_empty_global_cache_has_finite_local_attention_and_backward():
    layer = attention(CSA2Compressor(**config()))
    x, _, pos = inputs(1)
    x.requires_grad_()
    output = layer(x, position_ids=pos)
    assert output.shape == x.shape and output.isfinite().all()
    output.square().sum().backward()
    assert x.grad is not None and x.grad.isfinite().all()
    for parameter in (layer.q_a_proj.weight, layer.q_proj.weight, layer.swa_kv_proj.weight,
                      layer.sink_logits, layer.o_a_proj.weight, layer.o_b_proj.weight):
        assert parameter.grad is not None and parameter.grad.isfinite().all()


def test_attention_matches_joint_mqa_swa_sink_inverse_rope_reference():
    torch.manual_seed(3)
    layer = attention(CSA2Compressor(**config()))
    x, qr, pos = inputs(7)
    actual = layer(x, qr, pos)
    c = layer.compressor
    q = manual_rope(layer.q_proj(qr).view(2, 7, 4, 8), pos)
    local = manual_rope(norm(layer.swa_kv_proj(x), layer.swa_norm_weight), pos)
    global_kv = c.shared_cache.main_kv[:, 0]
    result = torch.empty(2, 7, 4, 8)
    for b in range(2):
        for t in range(7):
            # All complete blocks fit in top_k, so there is no learned selection ambiguity.
            values = torch.cat((local[b, max(0, t-2):t+1], global_kv[b, :(t+1)//2]), dim=0)
            logits = q[b, t] @ values.T / 8 ** 0.5
            p = torch.cat((logits, layer.sink_logits[:, None]), dim=-1).softmax(-1)[:, :-1]
            result[b, t] = p @ values
    result = manual_rope(result, pos, inverse=True).reshape(2, 7, 2, 16)
    weights = layer.o_a_proj.weight.view(2, 4, 16)
    expected = layer.o_b_proj(torch.einsum("bsgd,grd->bsgr", result, weights).flatten(2))
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


def test_future_tokens_do_not_change_prefix():
    torch.manual_seed(4)
    layer = attention(CSA2Compressor(**config(top_k=2)))
    x, qr, pos = inputs(9)
    a = layer(x, qr, pos)
    changed_x, changed_q = x.clone(), qr.clone()
    changed_x[:, 5:] += 50
    changed_q[:, 5:] -= 20
    b = layer(changed_x, changed_q, pos)
    torch.testing.assert_close(a[:, :5], b[:, :5])


@pytest.mark.parametrize("quantize", [False, True])
def test_chunked_decode_matches_prefill_across_three_modes(quantize):
    torch.manual_seed(5)
    cache = CSA2SharedCache()
    cfg = config(head_dim=32, index_head_dim=32, partial_rotary_factor=0.125, quantize=quantize)
    layers = [attention(CSA2Compressor(**cfg, mode=mode, shared_cache=cache))
              for mode in (CSA2Mode.FULL, CSA2Mode.REINDEX, CSA2Mode.REUSE)]
    x, _, pos = inputs(9)
    full = x
    with torch.no_grad():
        for i, layer in enumerate(layers):
            full = layer(full, position_ids=pos, layer_idx=i)
        cache.clear()
        states = [CSA2LayerState() for _ in layers]
        pieces = []
        for lo, hi in [(0, 3), (3, 4), (4, 6), (6, 9)]:
            chunk = x[:, lo:hi]
            for i, layer in enumerate(layers):
                chunk = layer(chunk, position_ids=pos[:, lo:hi], layer_idx=i, state=states[i])
            pieces.append(chunk)
        torch.testing.assert_close(full, torch.cat(pieces, 1), atol=2e-5, rtol=2e-4)
    assert states[0].main_kv.shape[2] == 4
    assert states[0].pending_hidden.shape[1] == 1
    assert all(state.swa_kv.shape[1] == 2 for state in states)


def test_ced_global_source_and_prepared_cache_replay():
    torch.manual_seed(6)
    c = CSA2Compressor(**config(is_decoder=True, compress_rate=1, hierarchical=True))
    x, qr, pos = inputs()
    enc = torch.randn_like(x)
    with pytest.raises(ValueError, match="final encoder"):
        c(x, qr, pos, 20)
    main, _ = c(x, qr, pos, 20, encoder_hidden_states=enc, encoder_position_ids=pos)
    changed, _ = c(x + 100, qr, pos, 20, encoder_hidden_states=enc, encoder_position_ids=pos)
    torch.testing.assert_close(main, changed)
    c.prepare_global(enc, pos, 20)
    prepared = c.shared_cache.main_kv
    layer = attention(c)
    output = layer(x[:, -2:], qr[:, -2:], pos[:, -2:], 20, reuse_global=True)
    assert output.isfinite().all()
    assert c.shared_cache.main_kv is prepared


def indexer(**overrides):
    args = dict(hidden_size=16, q_lora_rank=8, index_num_heads=2, index_head_dim=8,
                top_k=2, block_size=2, num_candidate_blocks=2, rope_dim=4, quantize=False)
    return HierarchicalSparseIndexer(**dict(args, **overrides))


def test_lightning_scores_sum_all_heads_and_receive_gradients():
    module = indexer()
    x, qr, _ = inputs(1)
    x.requires_grad_()
    qr.requires_grad_()
    keys = torch.randn(2, 1, 3, 8, requires_grad=True)
    pos = torch.zeros(2, 1, dtype=torch.long)
    ends = torch.zeros(2, 3, dtype=torch.long)
    scores = module.scores(x, qr, keys, position_ids=pos, key_end_position_ids=ends)
    q = module.indexer_q_proj(qr).view(2, 1, 2, 8)
    weights = module.weights_proj(x) / (8 * 2) ** .5
    expected = sum(weights[:, :, h, None] * (q[:, :, h] @ keys[:, 0].transpose(1, 2)).relu()
                   for h in range(2))
    torch.testing.assert_close(scores, expected)
    scores.sum().backward()
    assert module.indexer_q_proj.weight.grad is not None
    assert module.weights_proj.weight.grad is not None


def test_candidate_pools_are_per_query_and_never_include_future_or_padding():
    module = indexer(pin_recent_block=False, num_candidate_blocks=1)
    scores = torch.tensor([[[8., 7., 1., -float("inf"), -float("inf")],
                            [1., 2., 9., 8., 0.],
                            [-float("inf")] * 5]])
    pool = module._blockwise_candidate_selection(scores)
    assert pool.tolist() == [[[0, 1], [2, 3], [-1, -1]]]
    module = indexer(num_candidate_blocks=3)
    pool = module._blockwise_candidate_selection(torch.tensor([[[0., 1., 2., -float("inf"), -float("inf")]]]))
    assert ((pool == -1) | (pool < 3)).all()


def test_reindex_scores_only_shared_candidates_and_masks_invisible_entries():
    module = indexer()
    x, qr, pos = inputs(3, start=5)
    keys = torch.randn(2, 1, 9, 8)
    ends = torch.arange(9).expand(2, -1)
    full, pool = module.forward_full_mode(qr, keys, hidden_states=x, position_ids=pos,
                                         key_end_position_ids=ends)
    assert full.shape == (2, 3, 2)
    assert pool.shape == (2, 3, 4)
    other = indexer()
    subset_scores = other.scores(x, qr, keys, position_ids=pos, key_end_position_ids=ends,
                                 candidate_positions=pool)
    dense_scores = other.scores(x, qr, keys, position_ids=pos, key_end_position_ids=ends)
    expected = dense_scores.gather(-1, pool.clamp_min(0)).masked_fill(pool < 0, -float("inf"))
    torch.testing.assert_close(subset_scores, expected)
    idx = other.forward_reindex_mode(qr, keys, hidden_states=x, position_ids=pos,
                                     key_end_position_ids=ends, candidate_positions=pool)
    assert ((idx < 0) | (idx <= pos[..., None])).all()
    assert ((idx[..., None] == pool[:, :, None, :]).any(-1) | (idx < 0)).all()
    module.clear_cache()
    with pytest.raises(RuntimeError, match="candidate pool"):
        module.forward_reindex_mode(qr, keys, hidden_states=x, position_ids=pos, key_end_position_ids=ends)


def test_hierarchical_pool_is_integrated_across_decoder_layers():
    cache = CSA2SharedCache()
    cfg = config(compress_rate=1, is_decoder=True, hierarchical=True, top_k=2,
                 block_size=2, num_candidate_blocks=2)
    full = CSA2Compressor(**cfg, shared_cache=cache)
    reindex = CSA2Compressor(**cfg, shared_cache=cache, mode=CSA2Mode.REINDEX)
    reuse = CSA2Compressor(**cfg, shared_cache=cache, mode=CSA2Mode.REUSE)
    x, qr, pos = inputs(9)
    main, _ = full(x, qr, pos, 20, encoder_hidden_states=x + 1, encoder_position_ids=pos)
    pool = cache.candidate_pool_positions
    _, idx = reindex(x, qr, pos, 24)
    same, route = reuse(x, qr, pos, 25)
    assert cache.candidate_pool_positions is pool and same is main and route is idx
    assert ((idx[..., None] == pool[:, :, None, :]).any(-1) | (idx < 0)).all()


def test_quantization_formats_rounding_zero_and_straight_through_gradient():
    torch.testing.assert_close(_e2m1(torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5., -5.])),
                               torch.tensor([0., 1., 1., 2., 2., 4., 4., -4.]))
    torch.testing.assert_close(_e4m3(torch.tensor([0., 2.**-9, 1., 448., 500.])),
                               torch.tensor([0., 2.**-9, 1., 448., 448.]))
    x = torch.zeros(2, 32, requires_grad=True)
    for fmt, size in [("e4m3", 16), ("e8m0", 32)]:
        y = fake_quantize_fp4(x, block_size=size, scale_format=fmt)
        assert torch.equal(y, x) and y.isfinite().all()
        y.sum().backward()
    assert torch.equal(x.grad, torch.full_like(x, 2.))
    assert torch.equal(fake_quantize_swa(x), x)
