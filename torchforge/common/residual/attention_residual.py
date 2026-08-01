from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class BlockAttentionResidualState:
    """Compact depth state retained by Block Attention Residuals."""

    embedding: torch.Tensor
    completed_blocks: tuple[torch.Tensor, ...]
    current_block_sum: torch.Tensor
    current_module_count: int
    layers_in_current_block: int


class BlockAttentionResidual(nn.Module):
    """Kimi-K3 Block Attention Residuals from report equations 8-10."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_layers: int,
        block_size: int,
        sublayers_per_layer: int = 2,
        rms_norm_eps: float = 1.0e-6,
    ) -> None:
        super().__init__()
        for name, value in {
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "block_size": block_size,
            "sublayers_per_layer": sublayers_per_layer,
        }.items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive int, got {value!r}.")
        if rms_norm_eps <= 0.0:
            raise ValueError("rms_norm_eps must be positive.")
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.block_size = block_size
        self.sublayers_per_layer = sublayers_per_layer
        self.rms_norm_eps = float(rms_norm_eps)
        self.pseudo_queries = nn.Parameter(
            torch.zeros(num_layers * sublayers_per_layer + 1, hidden_size)
        )
        nn.init.normal_(self.pseudo_queries, mean=0.0, std=hidden_size**-0.5)

    def init_state(self, embedding: torch.Tensor) -> BlockAttentionResidualState:
        self._validate_hidden("embedding", embedding)
        return BlockAttentionResidualState(
            embedding=embedding,
            completed_blocks=(),
            current_block_sum=torch.zeros_like(embedding),
            current_module_count=0,
            layers_in_current_block=0,
        )

    def _sources(self, state: BlockAttentionResidualState) -> torch.Tensor:
        sources = [state.embedding, *state.completed_blocks]
        if state.current_module_count:
            sources.append(state.current_block_sum)
        return torch.stack(sources, dim=-2)

    def _read_with_query(
        self,
        state: BlockAttentionResidualState,
        query: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sources = self._sources(state)
        normalized = sources.float() * torch.rsqrt(
            sources.float().square().mean(-1, keepdim=True) + self.rms_norm_eps
        )
        scores = torch.einsum("...sh,h->...s", normalized, query.float())
        weights = torch.softmax(scores, dim=-1, dtype=torch.float32)
        output = torch.sum(sources * weights.to(sources.dtype).unsqueeze(-1), dim=-2)
        return output, weights

    def forward(
        self,
        state: BlockAttentionResidualState,
        *,
        layer_index: int,
        sublayer_index: int,
        return_dict: bool = True,
    ) -> Any:
        self._validate_state(state)
        if not 0 <= layer_index < self.num_layers:
            raise ValueError(f"layer_index must be in [0, {self.num_layers}), got {layer_index}.")
        if not 0 <= sublayer_index < self.sublayers_per_layer:
            raise ValueError(
                f"sublayer_index must be in [0, {self.sublayers_per_layer}), got {sublayer_index}."
            )
        query_index = layer_index * self.sublayers_per_layer + sublayer_index
        hidden_states, weights = self._read_with_query(state, self.pseudo_queries[query_index])
        if not return_dict:
            return hidden_states, weights
        return {"hidden_states": hidden_states, "attention_weights": weights}

    def update(
        self,
        state: BlockAttentionResidualState,
        module_output: torch.Tensor,
        *,
        layer_complete: bool,
    ) -> BlockAttentionResidualState:
        self._validate_state(state)
        self._validate_hidden("module_output", module_output)
        if module_output.shape != state.embedding.shape:
            raise ValueError("module_output must have the same shape as the embedding state.")
        current_sum = state.current_block_sum + module_output
        module_count = state.current_module_count + 1
        layers_in_block = state.layers_in_current_block + int(layer_complete)
        completed = state.completed_blocks
        if layer_complete and layers_in_block == self.block_size:
            completed = (*completed, current_sum)
            current_sum = torch.zeros_like(current_sum)
            module_count = 0
            layers_in_block = 0
        return BlockAttentionResidualState(
            embedding=state.embedding,
            completed_blocks=completed,
            current_block_sum=current_sum,
            current_module_count=module_count,
            layers_in_current_block=layers_in_block,
        )

    def finalize(self, state: BlockAttentionResidualState, *, return_dict: bool = True) -> Any:
        self._validate_state(state)
        hidden_states, weights = self._read_with_query(state, self.pseudo_queries[-1])
        if not return_dict:
            return hidden_states, weights
        return {"hidden_states": hidden_states, "attention_weights": weights}

    def _validate_hidden(self, name: str, value: torch.Tensor) -> None:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor.")
        if value.dim() != 3 or value.shape[-1] != self.hidden_size:
            raise ValueError(
                f"{name} must have shape (batch, sequence, {self.hidden_size}), got {tuple(value.shape)}."
            )

    def _validate_state(self, state: BlockAttentionResidualState) -> None:
        if not isinstance(state, BlockAttentionResidualState):
            raise TypeError("state must be a BlockAttentionResidualState.")
        self._validate_hidden("state.embedding", state.embedding)
        self._validate_hidden("state.current_block_sum", state.current_block_sum)
        for block in state.completed_blocks:
            self._validate_hidden("state.completed_blocks entry", block)
        if not 0 <= state.layers_in_current_block < self.block_size:
            raise ValueError("state.layers_in_current_block is outside the configured block.")


__all__ = ["BlockAttentionResidual", "BlockAttentionResidualState"]
