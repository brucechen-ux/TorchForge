from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch

from torchforge.common.attention import MLA
from torchforge.patches import DSV3MLAPatch, DSV4MLAPatch


EXPECTED_ORDER = [
    "query_projection",
    "kv_projection",
    "rotary",
    "kv_augment",
    "attention_bias",
    "attention_backend",
    "output_projection",
]


def _record_lifecycle_order(core: MLA) -> tuple[list[str], list[Any]]:
    call_order: list[str] = []
    handles = [
        getattr(core, name).register_forward_hook(
            lambda module, inputs, output, component_name=name: call_order.append(component_name)
        )
        for name in EXPECTED_ORDER
    ]
    return call_order, handles


def _remove_hooks(handles: list[Any]) -> None:
    for handle in handles:
        handle.remove()


def test_patch_swap_dsv3_and_dsv4_use_same_public_mla_type() -> None:
    dsv3_config = SimpleNamespace(
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
    dsv4_config = SimpleNamespace(
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

    dsv3_attention = DSV3MLAPatch.build(dsv3_config)
    dsv4_attention = DSV4MLAPatch.build(dsv4_config)
    dsv3_attention.eval()
    dsv4_attention.eval()

    assert type(dsv3_attention) is MLA
    assert type(dsv4_attention) is MLA

    dsv3_order, dsv3_handles = _record_lifecycle_order(dsv3_attention)
    dsv4_order, dsv4_handles = _record_lifecycle_order(dsv4_attention)
    try:
        batch_size = 2
        seq_length = 3

        dsv3_hidden = torch.randn(batch_size, seq_length, dsv3_config.hidden_size)
        dsv3_cos = torch.randn(batch_size, seq_length, dsv3_config.qk_rope_head_dim)
        dsv3_sin = torch.randn(batch_size, seq_length, dsv3_config.qk_rope_head_dim)
        dsv3_output = dsv3_attention(dsv3_hidden, position_embeddings=(dsv3_cos, dsv3_sin))
        assert dsv3_output["hidden_states"].shape == (batch_size, seq_length, dsv3_config.hidden_size)
        assert dsv3_order == EXPECTED_ORDER

        dsv4_hidden = torch.randn(batch_size, seq_length, dsv4_config.hidden_size)
        dsv4_rotary_dim = int(dsv4_config.head_dim * dsv4_config.partial_rotary_factor)
        dsv4_cos = torch.randn(batch_size, seq_length, dsv4_rotary_dim // 2)
        dsv4_sin = torch.randn(batch_size, seq_length, dsv4_rotary_dim // 2)
        dsv4_output = dsv4_attention(dsv4_hidden, position_embeddings={"main": (dsv4_cos, dsv4_sin)})
        assert dsv4_output["hidden_states"].shape == (batch_size, seq_length, dsv4_config.hidden_size)
        assert dsv4_order == EXPECTED_ORDER
    finally:
        _remove_hooks(dsv3_handles)
        _remove_hooks(dsv4_handles)
