from __future__ import annotations

from types import SimpleNamespace

import torch

from torchforge.patches import DSV3MLAPatch


def test_dsv3_mla_forward_and_component_lifecycle() -> None:
    config = SimpleNamespace(
        hidden_size=16,
        num_attention_heads=2,
        num_key_value_heads=2,
        q_lora_rank=4,
        kv_lora_rank=4,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=8,
        attention_dropout=0.0,
        rope_interleave=True,
        attention_bias=False,
        rms_norm_eps=1e-6,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    )
    attention = DSV3MLAPatch.build(config)
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
        hidden_states = torch.randn(batch_size, seq_length, config.hidden_size)
        cos = torch.randn(batch_size, seq_length, config.qk_rope_head_dim)
        sin = torch.randn(batch_size, seq_length, config.qk_rope_head_dim)

        outputs = attention(hidden_states, position_embeddings=(cos, sin))

        assert outputs["hidden_states"].shape == (batch_size, seq_length, config.hidden_size)
        assert call_order == expected_order
    finally:
        for handle in handles:
            handle.remove()
