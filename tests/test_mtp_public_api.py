from __future__ import annotations

import torch
from torch import nn

from torchforge.common.embedding import Embedding
from torchforge.common.lm_head import LMHead
from torchforge.common.mtp import MultiTokenPredictionModule


class _IdentityBlock(nn.Module):
    def forward(self, hidden_states: torch.Tensor, *, return_dict: bool = True, **kwargs: object) -> object:
        if return_dict:
            return {"hidden_states": hidden_states}
        return hidden_states


def test_mtp_public_api_forward_shape() -> None:
    mtp = MultiTokenPredictionModule(
        hidden_size=8,
        embedding=Embedding(vocab_size=16, hidden_size=8),
        transformer_block=_IdentityBlock(),
        lm_head=LMHead(hidden_size=8, vocab_size=16),
    )
    hidden_states = torch.randn(2, 5, 8)
    input_ids = torch.randint(0, 16, (2, 5))
    output = mtp(hidden_states, input_ids)
    assert output["hidden_states"].shape == (2, 4, 8)
    assert output["logits"].shape == (2, 4, 16)


def test_mtp_normalizes_both_inputs_before_combine_projection() -> None:
    embedding = Embedding(vocab_size=16, hidden_size=8)
    mtp = MultiTokenPredictionModule(
        hidden_size=8,
        embedding=embedding,
        transformer_block=_IdentityBlock(),
        lm_head=LMHead(hidden_size=8, vocab_size=16),
    )
    with torch.no_grad():
        mtp.combine_proj.weight.zero_()
        mtp.combine_proj.weight[:, :8].copy_(torch.eye(8))
        mtp.combine_proj.weight[:, 8:].copy_(torch.eye(8))

    hidden_states = torch.randn(2, 5, 8)
    input_ids = torch.randint(0, 16, (2, 5))
    output = mtp(hidden_states, input_ids)
    expected = mtp.hidden_norm(hidden_states[:, :-1]) + mtp.embedding_norm(embedding(input_ids[:, 1:]))

    assert torch.allclose(output["hidden_states"], expected)
