from __future__ import annotations

import pytest
import torch

from torchforge.common.attention import KDAState, KimiDeltaAttention


def _attention() -> KimiDeltaAttention:
    return KimiDeltaAttention(
        hidden_size=16,
        num_heads=2,
        head_dim=4,
        value_head_dim=4,
        short_conv_kernel_size=3,
        decay_rank=4,
        chunk_size=4,
        tile_size=2,
        backend="reference",
    )


def test_kda_public_forward_and_decay_bounds() -> None:
    attention = _attention()
    hidden_states = torch.randn(2, 7, 16, requires_grad=True)

    outputs = attention(hidden_states)
    outputs["hidden_states"].sum().backward()

    assert outputs["hidden_states"].shape == hidden_states.shape
    assert isinstance(outputs["state"], KDAState)
    assert torch.all(outputs["log_decay"] < 0.0)
    assert torch.all(outputs["log_decay"] > -5.0)
    assert hidden_states.grad is not None
    assert torch.isfinite(hidden_states.grad).all()


def test_kda_decay_respects_closed_floating_point_bounds() -> None:
    attention = _attention()
    with torch.no_grad():
        attention.decay_bias.fill_(1.0e6)
    lower = attention(torch.zeros(1, 2, 16))["log_decay"]
    with torch.no_grad():
        attention.decay_bias.fill_(-1.0e6)
    upper = attention(torch.zeros(1, 2, 16))["log_decay"]

    assert torch.all(lower >= attention.g_min)
    assert torch.all(lower <= 0.0)
    assert torch.all(upper >= attention.g_min)
    assert torch.all(upper <= 0.0)


def test_kda_segmented_state_matches_full_sequence() -> None:
    torch.manual_seed(7)
    attention = _attention().eval()
    hidden_states = torch.randn(1, 9, 16)

    full = attention(hidden_states)["hidden_states"]
    first = attention(hidden_states[:, :4])
    second = attention(hidden_states[:, 4:], state=first["state"])
    segmented = torch.cat((first["hidden_states"], second["hidden_states"]), dim=1)

    assert torch.allclose(full, segmented, atol=1.0e-5, rtol=1.0e-5)


def test_kda_segmented_state_matches_full_sequence_gradients() -> None:
    torch.manual_seed(17)
    attention = _attention().eval()
    full_input = torch.randn(1, 9, 16, requires_grad=True)
    segmented_input = full_input.detach().clone().requires_grad_(True)

    full_output = attention(full_input)["hidden_states"]
    full_output.square().sum().backward()
    full_input_grad = full_input.grad.detach().clone()
    full_weight_grad = attention.q_weight.grad.detach().clone()
    attention.zero_grad(set_to_none=True)

    first = attention(segmented_input[:, :4])
    second = attention(segmented_input[:, 4:], state=first["state"])
    segmented_output = torch.cat((first["hidden_states"], second["hidden_states"]), dim=1)
    segmented_output.square().sum().backward()

    assert torch.allclose(full_output, segmented_output, atol=1.0e-5, rtol=1.0e-5)
    assert torch.allclose(full_input_grad, segmented_input.grad, atol=1.0e-5, rtol=1.0e-5)
    assert torch.allclose(full_weight_grad, attention.q_weight.grad, atol=1.0e-5, rtol=1.0e-5)


def test_kda_is_causal() -> None:
    attention = _attention().eval()
    original = torch.randn(1, 8, 16)
    changed = original.clone()
    changed[:, 5:] = torch.randn_like(changed[:, 5:])

    original_output = attention(original)["hidden_states"]
    changed_output = attention(changed)["hidden_states"]

    assert torch.allclose(original_output[:, :5], changed_output[:, :5], atol=1.0e-5, rtol=1.0e-5)


def test_fla_backend_fails_explicitly_without_cuda() -> None:
    attention = _attention()
    attention.backend = "fla"

    with pytest.raises(RuntimeError, match="requires CUDA"):
        attention(torch.randn(1, 2, 16))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="BF16 CUDA validation requires a GPU")
def test_kda_reference_bf16_forward_backward_is_finite() -> None:
    attention = _attention().cuda().to(torch.bfloat16)
    hidden_states = torch.randn(1, 7, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    output = attention(hidden_states)["hidden_states"]
    output.float().square().mean().backward()

    assert torch.isfinite(output).all()
    assert torch.isfinite(hidden_states.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FLA validation requires a GPU")
def test_fla_backend_matches_reference_bf16_output_and_gradient() -> None:
    pytest.importorskip("fla.ops.kda")
    arguments = {
        "hidden_size": 64,
        "num_heads": 2,
        "head_dim": 32,
        "value_head_dim": 32,
        "short_conv_kernel_size": 3,
        "decay_rank": 8,
        "chunk_size": 32,
        "tile_size": 16,
    }
    torch.manual_seed(31)
    reference = (
        KimiDeltaAttention(**arguments, backend="reference").cuda().to(torch.bfloat16)
    )
    accelerated = KimiDeltaAttention(**arguments, backend="fla").cuda().to(torch.bfloat16)
    accelerated.load_state_dict(reference.state_dict())
    reference_input = torch.randn(
        1,
        35,
        64,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    accelerated_input = reference_input.detach().clone().requires_grad_(True)

    reference_output = reference(reference_input)["hidden_states"]
    accelerated_output = accelerated(accelerated_input)["hidden_states"]
    reference_output.float().square().mean().backward()
    accelerated_output.float().square().mean().backward()

    torch.testing.assert_close(accelerated_output, reference_output, atol=0.08, rtol=0.08)
    torch.testing.assert_close(
        accelerated_input.grad,
        reference_input.grad,
        atol=0.08,
        rtol=0.08,
    )
    torch.testing.assert_close(
        accelerated.q_weight.grad,
        reference.q_weight.grad,
        atol=0.08,
        rtol=0.08,
    )
