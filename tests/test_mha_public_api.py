from __future__ import annotations

import torch

from torchforge.common.attention import MHA
from torchforge.common.embedding import RotaryEmbedding


def test_public_mha_can_be_instantiated_directly() -> None:
    attention = MHA(
        hidden_size=16,
        num_attention_heads=4,
        head_dim=4,
        value_head_dim=4,
        rotary=True,
        rotary_layout="standard",
        rotary_application="full",
    )
    attention.eval()

    batch_size = 2
    seq_length = 3
    hidden_states = torch.randn(batch_size, seq_length, 16)
    position_ids = torch.arange(seq_length).unsqueeze(0).expand(batch_size, -1)
    cos, sin = RotaryEmbedding(head_dim=4)(position_ids)

    outputs = attention(hidden_states, position_embeddings=(cos, sin), output_attentions=True)

    assert outputs["hidden_states"].shape == (batch_size, seq_length, 16)
    assert outputs["attentions"] is not None
    assert outputs["attentions"].shape == (batch_size, 4, seq_length, seq_length)

    hidden_out, attentions = attention(hidden_states, position_embeddings=(cos, sin), return_dict=False)
    assert hidden_out.shape == (batch_size, seq_length, 16)
    assert attentions is None
