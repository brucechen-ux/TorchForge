from __future__ import annotations

import torch
from torch import nn

from torchforge.common.attention import MLA
from torchforge.common.attention.mla import KVAugment, MLAConfig, MLATensors


class _StaticCompressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.compressed = nn.Parameter(torch.randn(1, 1, 2, 4), requires_grad=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        compressed = self.compressed.expand(hidden_states.shape[0], -1, -1, -1)
        block_bias = compressed.new_zeros((hidden_states.shape[0], 1, hidden_states.shape[1], compressed.shape[2]))
        return compressed, block_bias


class _QueryProjectionStub:
    def __init__(self) -> None:
        self.last_q_residual = torch.randn(1, 3, 4)


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


def test_mla_kv_augment_uses_compressed_entries_as_key_and_value() -> None:
    config = MLAConfig(
        hidden_size=8,
        num_attention_heads=1,
        num_key_value_heads=1,
        q_lora_rank=4,
        kv_lora_rank=None,
        qk_nope_head_dim=2,
        qk_rope_head_dim=2,
        v_head_dim=4,
        kv_value_policy="value_equals_key_after_position",
    )
    augment = KVAugment(config, compressor=_StaticCompressor(), query_projection=_QueryProjectionStub())
    tensors = MLATensors(
        query=torch.randn(1, 1, 3, 4),
        key=torch.randn(1, 1, 3, 4),
        value=torch.randn(1, 1, 3, 4),
    )
    position_ids = torch.arange(3).unsqueeze(0)

    output = augment(tensors, hidden_states=torch.randn(1, 3, 8), position_ids=position_ids)

    assert output.key.shape == (1, 1, 5, 4)
    assert torch.equal(output.value, output.key)
    assert output.state["original_kv_len"] == 3
    assert output.state["compressed_kv_len"] == 2
