from __future__ import annotations

import torch

from torchforge.common.embedding import Embedding
from torchforge.common.lm_head import LMHead


def test_lm_head_public_api_forward_shape() -> None:
    lm_head = LMHead(hidden_size=8, vocab_size=13)
    hidden_states = torch.randn(2, 3, 8)
    logits = lm_head(hidden_states)
    assert logits.shape == (2, 3, 13)


def test_lm_head_tie_weights_shares_embedding_parameter() -> None:
    embedding = Embedding(vocab_size=13, hidden_size=8)
    lm_head = LMHead(hidden_size=8, vocab_size=13)
    lm_head.tie_weights(embedding)
    assert lm_head.weight is embedding.weight

