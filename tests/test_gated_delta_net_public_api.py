from __future__ import annotations

import pytest
import torch

from torchforge.common.attention import (
    GatedDeltaNet,
    GatedDeltaNetState,
    Qwen3NextGatedDeltaNet,
)


def _attention(**kwargs: object) -> Qwen3NextGatedDeltaNet:
    arguments = {
        "hidden_size": 16,
        "num_heads": 2,
        "head_dim": 4,
        "value_head_dim": 5,
        "beta_bias": -0.5,
        "decay_bias": 1.0,
        "backend": "reference",
    }
    arguments.update(kwargs)
    return Qwen3NextGatedDeltaNet(**arguments)


def test_gated_delta_net_forward_backward_and_alias() -> None:
    assert GatedDeltaNet is Qwen3NextGatedDeltaNet
    attention = _attention()
    hidden_states = torch.randn(2, 7, 16, requires_grad=True)

    outputs = attention(hidden_states)
    outputs["hidden_states"].square().mean().backward()

    assert outputs["hidden_states"].shape == hidden_states.shape
    assert isinstance(outputs["state"], GatedDeltaNetState)
    assert outputs["recurrent_state"].shape == (2, 2, 4, 5)
    assert outputs["recurrent_state"].dtype == torch.float32
    assert torch.all(outputs["alpha"] > 0.0)
    assert torch.all(outputs["alpha"] < 1.0)
    assert torch.all(outputs["beta"] > 0.0)
    assert torch.all(outputs["beta"] < 1.0)
    assert torch.isfinite(outputs["hidden_states"]).all()
    assert hidden_states.grad is not None
    assert torch.isfinite(hidden_states.grad).all()


def test_gated_delta_net_is_causal() -> None:
    attention = _attention().eval()
    original = torch.randn(1, 8, 16)
    changed = original.clone()
    changed[:, 5:] = torch.randn_like(changed[:, 5:])

    original_output = attention(original)["hidden_states"]
    changed_output = attention(changed)["hidden_states"]

    assert torch.allclose(original_output[:, :5], changed_output[:, :5], atol=1.0e-6, rtol=1.0e-5)


def test_gated_delta_net_segmented_matches_full_sequence() -> None:
    torch.manual_seed(9)
    attention = _attention().eval()
    hidden_states = torch.randn(1, 9, 16)

    full = attention(hidden_states)["hidden_states"]
    first = attention(hidden_states[:, :4])
    second = attention(hidden_states[:, 4:], state=first["state"])
    segmented = torch.cat((first["hidden_states"], second["hidden_states"]), dim=1)
    direct = attention(hidden_states[:, 4:], recurrent_state=first["recurrent_state"])

    assert torch.allclose(full, segmented, atol=1.0e-6, rtol=1.0e-5)
    assert torch.allclose(second["hidden_states"], direct["hidden_states"], atol=1.0e-6, rtol=1.0e-5)


def test_gated_delta_net_segmented_matches_full_sequence_gradients() -> None:
    torch.manual_seed(19)
    attention = _attention().eval()
    full_input = torch.randn(1, 7, 16, requires_grad=True)
    segmented_input = full_input.detach().clone().requires_grad_(True)

    full_output = attention(full_input)["hidden_states"]
    full_output.square().sum().backward()
    full_input_grad = full_input.grad.detach().clone()
    full_weight_grad = attention.q_proj.weight.grad.detach().clone()
    attention.zero_grad(set_to_none=True)

    first = attention(segmented_input[:, :3])
    second = attention(segmented_input[:, 3:], state=first["state"])
    segmented_output = torch.cat((first["hidden_states"], second["hidden_states"]), dim=1)
    segmented_output.square().sum().backward()

    assert torch.allclose(full_output, segmented_output, atol=1.0e-6, rtol=1.0e-5)
    assert torch.allclose(full_input_grad, segmented_input.grad, atol=1.0e-6, rtol=1.0e-5)
    assert torch.allclose(full_weight_grad, attention.q_proj.weight.grad, atol=1.0e-6, rtol=1.0e-5)


def test_gated_delta_net_tuple_output() -> None:
    hidden_states = torch.randn(1, 3, 16)

    output, state = _attention()(hidden_states, return_dict=False)

    assert output.shape == hidden_states.shape
    assert isinstance(state, GatedDeltaNetState)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"hidden_size": 0}, "hidden_size"),
        ({"num_heads": 0}, "num_heads"),
        ({"head_dim": 0}, "head_dim"),
        ({"backend": "fused"}, "backend"),
    ],
)
def test_gated_delta_net_rejects_invalid_configuration(arguments: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _attention(**arguments)


def test_gated_delta_net_validates_input_and_state() -> None:
    attention = _attention()
    hidden_states = torch.randn(2, 3, 16)
    wrong_shape = torch.zeros(2, 2, 4, 4)
    wrong_dtype = torch.zeros(2, 2, 4, 5, dtype=torch.float64)

    with pytest.raises(ValueError, match="hidden_states must have shape"):
        attention(torch.randn(2, 3, 15))
    with pytest.raises(ValueError, match="recurrent_state must have shape"):
        attention(hidden_states, recurrent_state=wrong_shape)
    with pytest.raises(ValueError, match="dtype torch.float32"):
        attention(hidden_states, recurrent_state=wrong_dtype)
    with pytest.raises(ValueError, match="only one"):
        attention(
            hidden_states,
            state=GatedDeltaNetState(torch.zeros(2, 2, 4, 5)),
            recurrent_state=torch.zeros(2, 2, 4, 5),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="device validation requires CUDA")
def test_gated_delta_net_rejects_state_on_wrong_device() -> None:
    attention = _attention().cuda()
    hidden_states = torch.randn(1, 2, 16, device="cuda")

    with pytest.raises(ValueError, match="must be on"):
        attention(hidden_states, recurrent_state=torch.zeros(1, 2, 4, 5))
