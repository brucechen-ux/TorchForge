from __future__ import annotations

import torch

from torchforge.common.attention import CompressedKVIndexer


def test_public_compressed_kv_indexer_can_be_instantiated_directly() -> None:
    indexer = CompressedKVIndexer(
        hidden_size=16,
        q_lora_rank=4,
        num_heads=2,
        head_dim=4,
        top_k=1,
        compress_rate=2,
        partial_rotary_factor=0.5,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
    )
    indexer.eval()

    batch_size = 2
    seq_length = 4
    hidden_states = torch.randn(batch_size, seq_length, 16)
    q_residual = torch.randn(batch_size, seq_length, 4)
    position_ids = torch.arange(seq_length).unsqueeze(0).expand(batch_size, -1)

    indices = indexer(hidden_states, q_residual=q_residual, position_ids=position_ids)

    assert indices.shape == (batch_size, seq_length, 1)
    assert indices.dtype == position_ids.dtype
