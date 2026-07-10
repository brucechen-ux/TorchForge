from __future__ import annotations

import torch

from torchforge.common.attention import HCACompressor


def test_public_hca_compressor_can_be_instantiated_directly() -> None:
    compressor = HCACompressor(
        hidden_size=16,
        head_dim=8,
        compress_rate=2,
        partial_rotary_factor=0.5,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
    )
    compressor.eval()

    batch_size = 2
    seq_length = 4
    hidden_states = torch.randn(batch_size, seq_length, 16)
    position_ids = torch.arange(seq_length).unsqueeze(0).expand(batch_size, -1)

    compressed_kv, block_bias = compressor(hidden_states, position_ids=position_ids)

    assert compressed_kv.shape == (batch_size, 1, seq_length // 2, 8)
    assert block_bias is not None
    assert block_bias.shape == (batch_size, 1, seq_length, seq_length // 2)


def test_hca_block_bias_exposes_only_completed_compressed_blocks() -> None:
    compressor = HCACompressor(
        hidden_size=8,
        head_dim=4,
        compress_rate=2,
        partial_rotary_factor=0.5,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
    )
    hidden_states = torch.randn(1, 6, 8)
    position_ids = torch.arange(6).unsqueeze(0)

    compressed_kv, block_bias = compressor(hidden_states, position_ids=position_ids)

    assert compressed_kv.shape == (1, 1, 3, 4)
    assert block_bias is not None
    visible = torch.isfinite(block_bias[0, 0])
    expected = torch.tensor(
        [
            [False, False, False],
            [True, False, False],
            [True, False, False],
            [True, True, False],
            [True, True, False],
            [True, True, True],
        ]
    )
    assert torch.equal(visible, expected)
