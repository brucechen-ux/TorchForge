"""Public API checks for the DeepSeek-V4.1-Flash Single-Pass mHC reference."""

import pytest
import torch

from torchforge.common.residual import SinglePassMHC, SinglePassMHCBlock


@pytest.fixture
def config():
    return {"num_branches": 4, "hidden_size": 32, "rms_norm_eps": 1e-6}


def make_state(config, batch=2, seq=5):
    return torch.randn(batch, seq, config["num_branches"], config["hidden_size"])


def test_initialization_and_state(config):
    mhc = SinglePassMHC(**config)
    hidden = torch.randn(2, 5, config["hidden_size"])
    state = mhc.init_state(hidden)
    assert state.shape == (2, 5, config["num_branches"], config["hidden_size"])
    assert torch.allclose(state, hidden.unsqueeze(-2).expand_as(state))
    initial_a = mhc.initial_input_weights(state)
    assert initial_a.shape == (2, 5, 1, config["num_branches"])
    assert torch.equal(initial_a[..., 0], torch.ones_like(initial_a[..., 0]))
    assert torch.count_nonzero(initial_a[..., 1:]) == 0


def test_report_coefficient_constraints(config):
    mhc = SinglePassMHC(**config, sinkhorn_iters=20)
    A, B, C = mhc._predict_coefficients(make_state(config))
    n = config["num_branches"]
    assert A.shape == (2, 5, 1, n)
    assert B.shape == (2, 5, n, n)
    assert C.shape == (2, 5, n, 1)
    assert torch.all((A >= 0) & (A <= 1))
    assert torch.all((C >= 0) & (C <= 2))
    assert torch.allclose(B.sum(dim=-1), torch.ones_like(B.sum(dim=-1)), atol=1e-5)
    assert torch.allclose(B.sum(dim=-2), torch.ones_like(B.sum(dim=-2)), atol=1e-5)


def test_delayed_input_map_and_update_formula(config):
    mhc = SinglePassMHC(**config)
    X = make_state(config)
    A_prev = torch.randn(2, 5, 1, config["num_branches"])
    block_output = torch.randn(2, 5, config["hidden_size"])
    A, B, C = mhc._predict_coefficients(X)
    X_hat = mhc.input_mix(X, A_prev)
    X_next = mhc.update_state(X, block_output, B, C)
    expected_hat = torch.matmul(A_prev.float(), X.float()).squeeze(-2).to(X.dtype)
    expected_next = (
        torch.matmul(B.float(), X.float())
        + C.float() * block_output.float().unsqueeze(-2)
    ).to(X.dtype)
    assert torch.allclose(X_hat, expected_hat)
    assert torch.allclose(X_next, expected_next)
    assert A.shape == A_prev.shape


def test_explicit_a_is_transferable_between_instances(config):
    X = make_state(config)
    first = SinglePassMHC(**config)
    second = SinglePassMHC(**config)
    A_prev = first.initial_input_weights(X)
    _, _, coefficients = first(
        X,
        torch.zeros(2, 5, config["hidden_size"]),
        A_prev=A_prev,
        return_coefficients=True,
    )
    mixed = second.input_mix(X, coefficients["A"])
    assert mixed.shape == (2, 5, config["hidden_size"])
    assert not hasattr(first, "prev_A")
    assert not hasattr(second, "prev_A")


def test_single_pass_block_returns_next_a(config):
    mhc = SinglePassMHC(**config)
    block = torch.nn.Linear(config["hidden_size"], config["hidden_size"], bias=False)
    wrapper = SinglePassMHCBlock(mhc=mhc, block=block)
    X = make_state(config)
    A_prev = mhc.initial_input_weights(X)
    X_next, A_next, details = wrapper(X, A_prev=A_prev, return_coefficients=True)
    assert X_next.shape == X.shape
    assert A_next.shape == A_prev.shape
    assert details["hidden_states"].shape == (2, 5, config["hidden_size"])
    assert details["block_output"].shape == (2, 5, config["hidden_size"])


def test_hc_eps_variant_preserves_public_shapes(config):
    mhc = SinglePassMHC(**config, hc_eps=1e-6)
    A, B, C = mhc._predict_coefficients(make_state(config))
    assert A.shape[-2:] == (1, config["num_branches"])
    assert B.shape[-2:] == (config["num_branches"], config["num_branches"])
    assert C.shape[-2:] == (config["num_branches"], 1)
