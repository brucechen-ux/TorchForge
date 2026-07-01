from __future__ import annotations

from types import SimpleNamespace

import torch

from torchforge.common.attention.mla import MLATensors
from torchforge.patches import DSV3MLAPatch, DSV4MLAPatch


def _dsv3_config() -> SimpleNamespace:
    return SimpleNamespace(
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


def _dsv4_config() -> SimpleNamespace:
    return SimpleNamespace(
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


def _assert_tensor_contract(tensor: torch.Tensor, *, dtype: torch.dtype, device: torch.device) -> None:
    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == dtype
    assert tensor.device == device


def test_dsv3_component_contracts() -> None:
    model_config = _dsv3_config()
    attention = DSV3MLAPatch.build(model_config)
    attention.eval()

    batch_size = 2
    seq_length = 3
    hidden_states = torch.randn(batch_size, seq_length, model_config.hidden_size)
    dtype = hidden_states.dtype
    device = hidden_states.device
    cos = torch.randn(batch_size, seq_length, model_config.qk_rope_head_dim, dtype=dtype, device=device)
    sin = torch.randn(batch_size, seq_length, model_config.qk_rope_head_dim, dtype=dtype, device=device)

    query = attention.query_projection(hidden_states)
    _assert_tensor_contract(query, dtype=dtype, device=device)
    assert query.shape == (
        batch_size,
        model_config.num_attention_heads,
        seq_length,
        model_config.qk_nope_head_dim + model_config.qk_rope_head_dim,
    )

    key, value = attention.kv_projection(hidden_states)
    _assert_tensor_contract(key, dtype=dtype, device=device)
    _assert_tensor_contract(value, dtype=dtype, device=device)
    assert key.shape == query.shape
    assert value.shape == (
        batch_size,
        model_config.num_attention_heads,
        seq_length,
        model_config.v_head_dim,
    )

    query, key = attention.rotary(query, key, position_embeddings=(cos, sin))
    tensors = MLATensors(query=query, key=key, value=value)
    tensors = attention.kv_augment(tensors, hidden_states=hidden_states)
    assert isinstance(tensors, MLATensors)
    tensors = attention.attention_bias(tensors, attention_mask=None)
    assert isinstance(tensors, MLATensors)
    assert tensors.attention_bias is None

    attention_output, attention_weights = attention.attention_backend(
        tensors.query,
        tensors.key,
        tensors.value,
        attention_bias=tensors.attention_bias,
        output_attentions=True,
    )
    _assert_tensor_contract(attention_output, dtype=dtype, device=device)
    assert attention_weights is not None
    _assert_tensor_contract(attention_weights, dtype=dtype, device=device)
    assert attention_output.shape == (
        batch_size,
        seq_length,
        model_config.num_attention_heads,
        model_config.v_head_dim,
    )

    output = attention.output_projection(attention_output)
    _assert_tensor_contract(output, dtype=dtype, device=device)
    assert output.shape == (batch_size, seq_length, model_config.hidden_size)


def test_dsv4_sliding_component_contracts() -> None:
    model_config = _dsv4_config()
    attention = DSV4MLAPatch.build(model_config)
    attention.eval()

    batch_size = 2
    seq_length = 3
    rotary_dim = int(model_config.head_dim * model_config.partial_rotary_factor)
    hidden_states = torch.randn(batch_size, seq_length, model_config.hidden_size)
    dtype = hidden_states.dtype
    device = hidden_states.device
    cos = torch.randn(batch_size, seq_length, rotary_dim // 2, dtype=dtype, device=device)
    sin = torch.randn(batch_size, seq_length, rotary_dim // 2, dtype=dtype, device=device)

    query = attention.query_projection(hidden_states)
    _assert_tensor_contract(query, dtype=dtype, device=device)
    assert query.shape == (
        batch_size,
        model_config.num_attention_heads,
        seq_length,
        model_config.head_dim,
    )

    key, value = attention.kv_projection(hidden_states)
    _assert_tensor_contract(key, dtype=dtype, device=device)
    _assert_tensor_contract(value, dtype=dtype, device=device)
    assert key.shape == (
        batch_size,
        model_config.num_key_value_heads,
        seq_length,
        model_config.head_dim,
    )
    assert value.shape == key.shape

    query, key = attention.rotary(query, key, position_embeddings={"main": (cos, sin)})
    tensors = MLATensors(query=query, key=key, value=value)
    tensors = attention.kv_augment(tensors, hidden_states=hidden_states)
    assert isinstance(tensors, MLATensors)
    assert tensors.value.shape == tensors.key.shape
    tensors = attention.attention_bias(tensors, attention_mask=None)
    assert isinstance(tensors, MLATensors)
    assert tensors.attention_bias is None

    attention_output, attention_weights = attention.attention_backend(
        tensors.query,
        tensors.key,
        tensors.value,
        attention_bias=tensors.attention_bias,
        output_attentions=True,
    )
    _assert_tensor_contract(attention_output, dtype=dtype, device=device)
    assert attention_weights is not None
    _assert_tensor_contract(attention_weights, dtype=dtype, device=device)
    assert attention_output.shape == (
        batch_size,
        seq_length,
        model_config.num_attention_heads,
        model_config.head_dim,
    )

    output = attention.output_projection(attention_output)
    _assert_tensor_contract(output, dtype=dtype, device=device)
    assert output.shape == (batch_size, seq_length, model_config.hidden_size)
