from __future__ import annotations

import torch

from torchforge.common.embedding import Embedding, RotaryEmbedding


def test_embedding_public_api_forward_shape() -> None:
    embedding = Embedding(vocab_size=11, hidden_size=7)
    input_ids = torch.tensor([[1, 2, 3]])
    output = embedding(input_ids)
    assert output.shape == (1, 3, 7)


def test_rotary_embedding_public_api_forward_shape() -> None:
    rotary = RotaryEmbedding(head_dim=8, partial_rotary_factor=0.5, rope_theta=10000.0)
    position_ids = torch.arange(4).unsqueeze(0)
    cos, sin = rotary(position_ids)
    assert cos.shape == sin.shape == (1, 4, 2)

