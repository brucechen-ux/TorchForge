from __future__ import annotations

from typing import Iterator, Optional, Tuple

import torch


def random_token_batches(
    *,
    vocab_size: int,
    batch_size: int,
    seq_length: int,
    num_steps: int,
    generator: Optional[torch.Generator] = None,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """Yield random ``(input_ids, labels)`` batches for smoke-testing training.

    Labels equal the inputs; the loss module is responsible for the causal
    shift, so this produces a next-token-prediction target. Intended for
    assembly demos and tests, not real training data.

    Args:
        vocab_size: Upper bound (exclusive) for sampled token ids.
        batch_size: Number of sequences per batch.
        seq_length: Number of tokens per sequence.
        num_steps: Number of batches to yield.
        generator: Optional RNG for reproducibility.

    Yields:
        ``(input_ids, labels)`` tensors, each of shape ``(batch_size, seq_length)``.
    """

    _validate_positive_int("vocab_size", vocab_size)
    _validate_positive_int("batch_size", batch_size)
    _validate_positive_int("seq_length", seq_length)
    _validate_positive_int("num_steps", num_steps)
    if generator is not None and not isinstance(generator, torch.Generator):
        raise TypeError("generator must be a torch.Generator or None.")

    for _ in range(num_steps):
        input_ids = torch.randint(
            0, vocab_size, (batch_size, seq_length), generator=generator, dtype=torch.long
        )
        yield input_ids, input_ids.clone()


def _validate_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive int, got {value!r}.")


__all__ = ["random_token_batches"]
