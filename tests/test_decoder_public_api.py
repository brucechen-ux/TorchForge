from __future__ import annotations

import torch

from torchforge.common.decoder import DecoderLayer


def test_decoder_layer_dense_forward_shape() -> None:
    layer = DecoderLayer(
        hidden_size=16,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=32,
        q_lora_rank=4,
        kv_lora_rank=4,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=8,
        ffn_type="dense",
    )
    hidden_states = torch.randn(2, 3, 16)
    cos = torch.randn(2, 3, 2)
    sin = torch.randn(2, 3, 2)
    output = layer(hidden_states, position_embeddings=(cos, sin), return_dict=False)
    assert output.shape == (2, 3, 16)


def test_decoder_layer_moe_forward_shape() -> None:
    layer = DecoderLayer(
        hidden_size=16,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=32,
        q_lora_rank=4,
        kv_lora_rank=4,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=8,
        ffn_type="moe",
        num_experts=4,
        num_experts_per_tok=2,
    )
    hidden_states = torch.randn(2, 3, 16)
    cos = torch.randn(2, 3, 2)
    sin = torch.randn(2, 3, 2)
    output = layer(hidden_states, position_embeddings=(cos, sin), return_dict=False)
    assert output.shape == (2, 3, 16)

