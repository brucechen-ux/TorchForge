from __future__ import annotations

from typing import Any, Optional

import torch
from torch import nn


class MultiTokenPredictionModule(nn.Module):
    """Sequential multi-token prediction module.

    Args:
        hidden_size: Hidden-state dimension.
        embedding: Shared token embedding module.
        transformer_block: Block used for the MTP depth.
        lm_head: Shared language-model head module.
        bias: Whether the input-combine projection uses bias.

    Forward:
        ``hidden_states`` has shape ``(batch, sequence_length, hidden_size)``.
        ``input_ids`` has shape ``(batch, sequence_length)``.

    Returns:
        With ``return_dict=True``, returns ``hidden_states`` and ``logits``.
        With ``return_dict=False``, returns ``(hidden_states, logits)``.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        embedding: nn.Module,
        transformer_block: nn.Module,
        lm_head: nn.Module,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ValueError(f"hidden_size must be a positive int, got {hidden_size!r}.")
        self.hidden_size = hidden_size
        self.embedding = embedding
        self.transformer_block = transformer_block
        self.lm_head = lm_head
        self.combine_proj = nn.Linear(2 * hidden_size, hidden_size, bias=bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Any] = None,
        return_dict: bool = True,
        **kwargs: Any,
    ) -> Any:
        _validate_hidden_states(hidden_states, self.hidden_size)
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError(f"input_ids must be a torch.Tensor, got {type(input_ids).__name__}.")
        if input_ids.dim() != 2:
            raise ValueError("input_ids must have shape (batch, sequence_length).")
        if tuple(input_ids.shape) != tuple(hidden_states.shape[:2]):
            raise ValueError(
                f"input_ids must match hidden_states batch and sequence dimensions, got "
                f"{tuple(input_ids.shape)} and {tuple(hidden_states.shape[:2])}."
            )

        shifted_hidden = hidden_states[:, :-1]
        shifted_input_ids = input_ids[:, 1:]
        token_embeddings = self.embedding(shifted_input_ids)
        combined = self.combine_proj(torch.cat([shifted_hidden, token_embeddings], dim=-1))

        block_kwargs = dict(kwargs)
        if attention_mask is not None:
            block_kwargs["attention_mask"] = attention_mask[..., :-1, :-1]
        if position_ids is not None:
            block_kwargs["position_ids"] = position_ids[:, :-1]
        if position_embeddings is not None:
            block_kwargs["position_embeddings"] = _slice_position_embeddings(position_embeddings)
        block_kwargs.setdefault("input_ids", shifted_input_ids)

        block_output = self.transformer_block(combined, return_dict=True, **block_kwargs)
        if isinstance(block_output, dict):
            next_hidden = block_output["hidden_states"]
        else:
            next_hidden = block_output[0] if isinstance(block_output, tuple) else block_output
        logits = self.lm_head(next_hidden)
        if return_dict:
            return {"hidden_states": next_hidden, "logits": logits}
        return next_hidden, logits


def _slice_position_embeddings(position_embeddings: Any) -> Any:
    if isinstance(position_embeddings, tuple) and len(position_embeddings) == 2:
        return position_embeddings[0][:, :-1], position_embeddings[1][:, :-1]
    if isinstance(position_embeddings, dict):
        return {key: _slice_position_embeddings(value) for key, value in position_embeddings.items()}
    return position_embeddings


def _validate_hidden_states(hidden_states: torch.Tensor, hidden_size: int) -> None:
    if not isinstance(hidden_states, torch.Tensor):
        raise TypeError(f"hidden_states must be a torch.Tensor, got {type(hidden_states).__name__}.")
    if hidden_states.dim() != 3:
        raise ValueError("hidden_states must have shape (batch, sequence_length, hidden_size).")
    if hidden_states.shape[-1] != hidden_size:
        raise ValueError(f"hidden_states last dimension must be {hidden_size}, got {hidden_states.shape[-1]}.")
    if hidden_states.shape[1] < 2:
        raise ValueError("hidden_states sequence_length must be at least 2 for MTP.")


__all__ = ["MultiTokenPredictionModule"]
