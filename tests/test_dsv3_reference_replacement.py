from __future__ import annotations

import torch

from experiments.dsv3_reference.config import DSV3ReferenceConfig
from experiments.dsv3_reference.model import DSV3ReferenceModel
from experiments.dsv3_torchforge.model import ComponentConfig, DSV3TorchForgeModel


def _copy_reference_weights(reference: DSV3ReferenceModel, target: DSV3TorchForgeModel) -> None:
    target.embed_tokens.load_state_dict(reference.embed_tokens.state_dict())
    target.norm.load_state_dict(reference.norm.state_dict())
    target.lm_head.load_state_dict(reference.lm_head.state_dict())

    for reference_layer, target_layer in zip(reference.layers, target.layers):
        target_layer.input_norm.load_state_dict(reference_layer.input_norm.state_dict())
        target_layer.post_attention_norm.load_state_dict(reference_layer.post_attention_norm.state_dict())
        target_layer.mlp.load_state_dict(reference_layer.mlp.state_dict())

        source_attention = reference_layer.self_attn
        target_attention = target_layer.self_attn.attention
        target_attention.query_projection.q_a_proj.load_state_dict(source_attention.q_a_proj.state_dict())
        target_attention.query_projection.q_b_proj.load_state_dict(source_attention.q_b_proj.state_dict())
        target_attention.query_projection.q_a_norm_weight.data.copy_(source_attention.q_a_norm_weight.data)
        target_attention.kv_projection.kv_a_proj_with_mqa.load_state_dict(
            source_attention.kv_a_proj_with_mqa.state_dict()
        )
        target_attention.kv_projection.kv_a_norm_weight.data.copy_(source_attention.kv_a_norm_weight.data)
        target_attention.kv_projection.kv_b_proj.load_state_dict(source_attention.kv_b_proj.state_dict())
        target_attention.output_projection.o_proj.load_state_dict(source_attention.o_proj.state_dict())


def test_dsv3_reference_model_torchforge_mla_replacement_matches() -> None:
    torch.manual_seed(1234)
    config = DSV3ReferenceConfig(
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        q_lora_rank=4,
        kv_lora_rank=4,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=8,
        intermediate_size=32,
    )
    reference = DSV3ReferenceModel(config)
    replacement = DSV3TorchForgeModel(
        config,
        ComponentConfig(attention="torchforge", norm="pytorch", ffn="pytorch", kv="pytorch"),
    )
    _copy_reference_weights(reference, replacement)
    reference.eval()
    replacement.eval()

    input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    labels = input_ids.roll(shifts=-1, dims=1)
    reference_outputs = reference(input_ids, labels=labels)
    replacement_outputs = replacement(input_ids, labels=labels)

    assert reference_outputs["logits"].shape == replacement_outputs["logits"].shape == (2, 4, config.vocab_size)
    torch.testing.assert_close(replacement_outputs["logits"], reference_outputs["logits"], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(replacement_outputs["loss"], reference_outputs["loss"], rtol=1e-5, atol=1e-5)

