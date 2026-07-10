from __future__ import annotations

import torch
from torch import nn

from torchforge.common.attention import CSACompressor


class _FixedIndexer(nn.Module):
    def __init__(self, indices: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("indices", indices)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        q_residual: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.indices.expand(hidden_states.shape[0], -1, -1)


def test_public_csa_compressor_can_be_instantiated_directly() -> None:
    compressor = CSACompressor(
        hidden_size=16,
        q_lora_rank=4,
        head_dim=8,
        index_num_heads=2,
        index_head_dim=4,
        index_top_k=1,
        compress_rate=2,
        partial_rotary_factor=0.5,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
    )
    compressor.eval()

    batch_size = 2
    seq_length = 4
    hidden_states = torch.randn(batch_size, seq_length, 16)
    q_residual = torch.randn(batch_size, seq_length, 4)
    position_ids = torch.arange(seq_length).unsqueeze(0).expand(batch_size, -1)

    compressed_kv, block_bias = compressor(
        hidden_states,
        q_residual=q_residual,
        position_ids=position_ids,
    )

    assert compressed_kv.shape == (batch_size, 1, seq_length // 2, 8)
    assert block_bias is not None
    assert block_bias.shape == (batch_size, 1, seq_length, seq_length // 2)


def test_csa_block_bias_exposes_only_indexed_topk_entries() -> None:
    fixed_indices = torch.tensor(
        [
            [-1, -1],
            [0, -1],
            [1, 0],
            [2, 0],
            [1, 2],
            [0, 2],
        ],
        dtype=torch.long,
    ).unsqueeze(0)
    compressor = CSACompressor(
        hidden_size=8,
        q_lora_rank=4,
        head_dim=4,
        index_num_heads=1,
        index_head_dim=4,
        index_top_k=2,
        compress_rate=2,
        partial_rotary_factor=0.5,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        indexer=_FixedIndexer(fixed_indices),
    )
    hidden_states = torch.randn(1, 6, 8)
    q_residual = torch.randn(1, 6, 4)
    position_ids = torch.arange(6).unsqueeze(0)

    compressed_kv, block_bias = compressor(
        hidden_states,
        q_residual=q_residual,
        position_ids=position_ids,
    )

    assert compressed_kv.shape == (1, 1, 3, 4)
    assert block_bias is not None
    visible = torch.isfinite(block_bias[0, 0])
    expected = torch.tensor(
        [
            [False, False, False],
            [True, False, False],
            [True, True, False],
            [True, False, True],
            [False, True, True],
            [True, False, True],
        ]
    )
    assert torch.equal(visible, expected)
