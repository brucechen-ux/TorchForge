from __future__ import annotations

import torch
from torch import nn

from torchforge.common.loss import CausalLMLoss
from torchforge.common.optim import AdamW, build_param_groups
from torchforge.common.train import TrainStep, random_token_batches


class _TinyLM(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(input_ids))


def test_random_token_batches_shapes_and_count() -> None:
    generator = torch.Generator().manual_seed(0)
    batches = list(
        random_token_batches(
            vocab_size=16, batch_size=2, seq_length=5, num_steps=3, generator=generator
        )
    )
    assert len(batches) == 3
    for input_ids, labels in batches:
        assert input_ids.shape == (2, 5)
        assert labels.shape == (2, 5)
        assert torch.equal(input_ids, labels)


def test_train_step_reduces_loss() -> None:
    torch.manual_seed(0)
    model = _TinyLM(vocab_size=16, hidden_size=8)
    optimizer = AdamW(build_param_groups(model, weight_decay=0.0), lr=1e-2)
    step = TrainStep(forward_fn=model, loss_module=CausalLMLoss(), optimizer=optimizer)

    generator = torch.Generator().manual_seed(0)
    losses = [
        step.run(input_ids, labels)["loss"]
        for input_ids, labels in random_token_batches(
            vocab_size=16, batch_size=4, seq_length=8, num_steps=50, generator=generator
        )
    ]
    assert losses[-1] < losses[0]


def test_train_step_returns_metrics() -> None:
    torch.manual_seed(0)
    model = _TinyLM(vocab_size=16, hidden_size=8)
    optimizer = AdamW(model.parameters(), lr=1e-2)
    step = TrainStep(forward_fn=model, loss_module=CausalLMLoss(), optimizer=optimizer)
    metrics = step.run(*next(random_token_batches(vocab_size=16, batch_size=2, seq_length=6, num_steps=1)))
    assert set(metrics) == {"loss", "grad_norm"}
    assert metrics["grad_norm"] >= 0.0
