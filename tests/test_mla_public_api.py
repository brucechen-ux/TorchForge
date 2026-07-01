from __future__ import annotations

import torch

from torchforge.common.attention import MLA


def test_public_mla_can_be_instantiated_directly() -> None:
    attention = MLA(
        hidden_size=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        q_lora_rank=None,
        kv_lora_rank=None,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=8,
        query_projection_type="direct",
        kv_projection_type="direct_kv",
        kv_final_norm="none",
        kv_value_mode="projected_value",
        rotary_layout="interleaved",
        rotary_application="partial_trailing",
        position_source="tuple",
        repeat_kv=True,
        output_projection_type="linear",
    )
    attention.eval()

    batch_size = 2
    seq_length = 3
    hidden_states = torch.randn(batch_size, seq_length, 16)
    cos = torch.randn(batch_size, seq_length, 2)
    sin = torch.randn(batch_size, seq_length, 2)

    outputs = attention(hidden_states, position_embeddings=(cos, sin))

    assert outputs["hidden_states"].shape == (batch_size, seq_length, 16)

    hidden_out, attentions = attention(hidden_states, position_embeddings=(cos, sin), return_dict=False)
    assert hidden_out.shape == (batch_size, seq_length, 16)
    assert attentions is None
