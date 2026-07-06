from __future__ import annotations

import torch

from torchforge.common.loss import CausalLMLoss


def test_causal_lm_loss_returns_scalar() -> None:
    loss = CausalLMLoss()
    logits = torch.randn(2, 5, 7)
    labels = torch.randint(0, 7, (2, 5))
    output = loss(logits, labels)
    assert output.dim() == 0
    assert output.item() > 0.0


def test_causal_lm_loss_matches_manual_shift() -> None:
    torch.manual_seed(0)
    loss = CausalLMLoss()
    logits = torch.randn(1, 4, 6)
    labels = torch.randint(0, 6, (1, 4))
    expected = torch.nn.functional.cross_entropy(
        logits[:, :-1, :].reshape(-1, 6), labels[:, 1:].reshape(-1)
    )
    torch.testing.assert_close(loss(logits, labels), expected)


def test_causal_lm_loss_ignore_index_excludes_tokens() -> None:
    loss = CausalLMLoss(ignore_index=-100)
    logits = torch.randn(1, 3, 5)
    labels = torch.tensor([[0, -100, -100]])
    # Only the position predicting label[1] contributes, and it is ignored -> all ignored.
    output = loss(logits, labels)
    assert torch.isnan(output)
