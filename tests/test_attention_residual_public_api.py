from __future__ import annotations

import torch

from torchforge.common.residual import BlockAttentionResidual, BlockAttentionResidualState


def test_block_attention_residual_tracks_only_completed_blocks_and_partial_sum() -> None:
    residual = BlockAttentionResidual(hidden_size=4, num_layers=2, block_size=1)
    with torch.no_grad():
        residual.pseudo_queries.zero_()
    embedding = torch.ones(1, 2, 4)
    state = residual.init_state(embedding)

    first_read = residual(state, layer_index=0, sublayer_index=0)["hidden_states"]
    state = residual.update(state, first_read * 2.0, layer_complete=False)
    second_read = residual(state, layer_index=0, sublayer_index=1)["hidden_states"]
    state = residual.update(state, second_read * 3.0, layer_complete=True)

    assert isinstance(state, BlockAttentionResidualState)
    assert len(state.completed_blocks) == 1
    assert state.current_module_count == 0
    assert torch.allclose(second_read, torch.full_like(second_read, 1.5))


def test_block_attention_residual_finalizer_includes_partial_block() -> None:
    residual = BlockAttentionResidual(hidden_size=4, num_layers=2, block_size=2)
    state = residual.init_state(torch.randn(1, 3, 4))
    state = residual.update(state, torch.randn(1, 3, 4), layer_complete=True)

    output = residual.finalize(state)

    assert output["hidden_states"].shape == (1, 3, 4)
    assert output["attention_weights"].shape[-1] == 2


def test_block_attention_residual_sums_layer_outputs_not_sublayer_states() -> None:
    residual = BlockAttentionResidual(hidden_size=2, num_layers=1, block_size=1)
    embedding = torch.zeros(1, 1, 2)
    state = residual.init_state(embedding)
    state = residual.update(state, torch.full_like(embedding, 2.0), layer_complete=False)
    state = residual.update(state, torch.full_like(embedding, 3.0), layer_complete=True)

    assert len(state.completed_blocks) == 1
    assert torch.equal(state.completed_blocks[0], torch.full_like(embedding, 5.0))


def test_block_attention_residual_matches_explicit_softmax_reference() -> None:
    residual = BlockAttentionResidual(hidden_size=2, num_layers=2, block_size=1)
    embedding = torch.tensor([[[1.0, 2.0]]])
    state = residual.init_state(embedding)
    state = residual.update(state, torch.tensor([[[2.0, 0.0]]]), layer_complete=False)
    state = residual.update(state, torch.tensor([[[0.0, 3.0]]]), layer_complete=True)
    state = residual.update(state, torch.tensor([[[-1.0, 1.0]]]), layer_complete=False)
    with torch.no_grad():
        residual.pseudo_queries[3].copy_(torch.tensor([0.25, -0.5]))

    actual = residual(state, layer_index=1, sublayer_index=1)["hidden_states"]
    sources = torch.stack(
        [embedding, torch.tensor([[[2.0, 3.0]]]), torch.tensor([[[-1.0, 1.0]]])],
        dim=-2,
    )
    normalized = sources * torch.rsqrt(
        sources.square().mean(dim=-1, keepdim=True) + residual.rms_norm_eps
    )
    weights = torch.softmax(
        torch.einsum("...sh,h->...s", normalized, residual.pseudo_queries[3]),
        dim=-1,
    )
    expected = torch.sum(sources * weights.unsqueeze(-1), dim=-2)

    assert torch.allclose(actual, expected, atol=1.0e-6, rtol=1.0e-6)
