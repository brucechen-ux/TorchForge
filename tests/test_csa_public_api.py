from __future__ import annotations

import torch

from torchforge.common.kv import CSACompressor


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
