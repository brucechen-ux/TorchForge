from __future__ import annotations

import torch
import torch.nn.functional as F

from torchforge.common.attention import GatedMLA


def _attention() -> GatedMLA:
    return GatedMLA(
        hidden_size=16,
        num_heads=2,
        q_lora_rank=8,
        kv_lora_rank=4,
        head_dim=4,
        value_head_dim=4,
        attention_backend="reference",
    )


def test_gated_mla_is_nope_and_directly_instantiable() -> None:
    attention = _attention().eval()
    hidden_states = torch.randn(2, 6, 16)

    first = attention(hidden_states, position_ids=torch.arange(6).unsqueeze(0))["hidden_states"]
    second = attention(hidden_states, position_ids=torch.arange(100, 106).unsqueeze(0))["hidden_states"]

    assert first.shape == hidden_states.shape
    assert torch.equal(first, second)


def test_gated_mla_is_causal() -> None:
    attention = _attention().eval()
    original = torch.randn(1, 6, 16)
    changed = original.clone()
    changed[:, 4:] = torch.randn_like(changed[:, 4:])

    first = attention(original)["hidden_states"]
    second = attention(changed)["hidden_states"]

    assert torch.allclose(first[:, :4], second[:, :4], atol=1.0e-5, rtol=1.0e-5)


def test_gated_mla_reference_keeps_attention_outputs_in_fp32() -> None:
    attention = _attention().eval()
    hidden_states = torch.randn(2, 5, 16)
    key_mask = torch.ones(2, 5, dtype=torch.bool)
    outputs = attention(hidden_states, attention_mask=key_mask, output_attentions=True)

    assert outputs["attentions"].dtype == torch.float32
    assert torch.isfinite(outputs["hidden_states"]).all()


def test_gated_mla_applies_channel_gate_before_output_projection() -> None:
    attention = _attention().eval()
    hidden_states = torch.randn(1, 4, 16)
    with torch.no_grad():
        attention.output_gate.weight.zero_()
    output = attention(hidden_states)["hidden_states"]
    query_latent = attention.q_a_norm(attention.q_a_proj(hidden_states))
    kv_latent = attention.kv_a_norm(attention.kv_a_proj(hidden_states))
    query = attention._packed_linear(query_latent, attention.q_b_weight)
    key = attention._packed_linear(kv_latent, attention.k_weight)
    value = attention._packed_linear(kv_latent, attention.v_weight)
    ungated, _ = attention._reference_attention(query, key, value, None, False)
    expected = F.linear(0.5 * ungated.flatten(2), attention.output_proj.weight.float())

    assert torch.allclose(output, expected, atol=1.0e-6, rtol=1.0e-6)
