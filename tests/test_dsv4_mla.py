from __future__ import annotations

from types import SimpleNamespace

import torch

from torchforge.patches import DSV4MLAPatch


def test_dsv4_mla_forward_and_component_lifecycle_sliding_attention() -> None:
    config = SimpleNamespace(
        hidden_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        partial_rotary_factor=0.5,
        q_lora_rank=4,
        attention_dropout=0.0,
        layer_types=["sliding_attention"],
        o_groups=2,
        o_lora_rank=4,
        rms_norm_eps=1e-6,
    )
    attention = DSV4MLAPatch.build(config)
    attention.eval()

    call_order: list[str] = []
    expected_order = [
        "query_projection",
        "kv_projection",
        "rotary",
        "kv_augment",
        "attention_bias",
        "attention_backend",
        "output_projection",
    ]
    handles = [
        getattr(attention, name).register_forward_hook(
            lambda module, inputs, output, component_name=name: call_order.append(component_name)
        )
        for name in expected_order
    ]

    try:
        batch_size = 2
        seq_length = 3
        rotary_dim = int(config.head_dim * config.partial_rotary_factor)
        hidden_states = torch.randn(batch_size, seq_length, config.hidden_size)
        cos = torch.randn(batch_size, seq_length, rotary_dim // 2)
        sin = torch.randn(batch_size, seq_length, rotary_dim // 2)

        outputs = attention(hidden_states, position_embeddings={"main": (cos, sin)})

        assert outputs["hidden_states"].shape == (batch_size, seq_length, config.hidden_size)
        assert call_order == expected_order
    finally:
        for handle in handles:
            handle.remove()
