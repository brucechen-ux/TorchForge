from __future__ import annotations

from typing import Any

import torch
from torch import nn

from torchforge.common.nn import RMSNorm


class ManifoldConstrainedHyperConnection(nn.Module):
    """Manifold-constrained hyper-connection for expanded residual streams.

    Args:
        hidden_size: Hidden-state dimension of each residual path.
        expansion_factor: Number of residual paths.
        sinkhorn_iters: Number of Sinkhorn row/column normalization iterations.
        dynamic: Whether to add input-dependent dynamic parameters.

    Forward:
        ``residual_state`` has shape ``(..., expansion_factor, hidden_size)``.
        ``update`` has shape ``(..., hidden_size)``.

    Returns:
        With ``return_dict=True``, returns ``residual_state``, ``hidden_states``,
        ``input_weights``, ``output_weights``, and ``residual_mapping``.
        With ``return_dict=False``, returns ``(residual_state, hidden_states)``.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        expansion_factor: int = 4,
        sinkhorn_iters: int = 20,
        dynamic: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ValueError(f"hidden_size must be a positive int, got {hidden_size!r}.")
        if not isinstance(expansion_factor, int) or expansion_factor <= 0:
            raise ValueError(f"expansion_factor must be a positive int, got {expansion_factor!r}.")
        if not isinstance(sinkhorn_iters, int) or sinkhorn_iters <= 0:
            raise ValueError(f"sinkhorn_iters must be a positive int, got {sinkhorn_iters!r}.")
        self.hidden_size = hidden_size
        self.expansion_factor = expansion_factor
        self.sinkhorn_iters = sinkhorn_iters
        self.dynamic = dynamic
        self.input_logits = nn.Parameter(torch.zeros(expansion_factor))
        self.output_logits = nn.Parameter(torch.zeros(expansion_factor))
        self.residual_logits = nn.Parameter(torch.zeros(expansion_factor, expansion_factor))
        if dynamic:
            dynamic_hidden_size = expansion_factor * hidden_size
            self.dynamic_norm = RMSNorm(dynamic_hidden_size)
            self.dynamic_input = nn.Linear(dynamic_hidden_size, expansion_factor, bias=False)
            self.dynamic_output = nn.Linear(dynamic_hidden_size, expansion_factor, bias=False)
            self.dynamic_residual = nn.Linear(dynamic_hidden_size, expansion_factor * expansion_factor, bias=False)
            self.dynamic_input_gate = nn.Parameter(torch.zeros(()))
            self.dynamic_output_gate = nn.Parameter(torch.zeros(()))
            self.dynamic_residual_gate = nn.Parameter(torch.zeros(()))
        else:
            self.dynamic_norm = None
            self.dynamic_input = None
            self.dynamic_output = None
            self.dynamic_residual = None
            self.register_parameter("dynamic_input_gate", None)
            self.register_parameter("dynamic_output_gate", None)
            self.register_parameter("dynamic_residual_gate", None)

    def init_state(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Expand a standard hidden-state tensor into an mHC residual state."""

        _validate_hidden_states("hidden_states", hidden_states, self.hidden_size)
        return hidden_states.unsqueeze(-2).expand(*hidden_states.shape[:-1], self.expansion_factor, self.hidden_size)

    def project_residual_mapping(self, raw_logits: torch.Tensor | None = None) -> torch.Tensor:
        """Return the Sinkhorn-projected residual mapping matrix."""

        mapping = torch.exp((self.residual_logits if raw_logits is None else raw_logits).float())
        for _ in range(self.sinkhorn_iters):
            mapping = mapping / mapping.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
            mapping = mapping / mapping.sum(dim=-2, keepdim=True).clamp_min(1.0e-12)
        target_dtype = self.residual_logits.dtype if raw_logits is None else raw_logits.dtype
        return mapping.to(target_dtype)

    def read(self, residual_state: torch.Tensor) -> torch.Tensor:
        """Collapse an expanded residual state into a hidden-state tensor."""

        _validate_residual_state(residual_state, self.expansion_factor, self.hidden_size)
        input_weights, _, _ = self._constrained_parameters(residual_state)
        if input_weights.dim() == 1:
            input_weights = input_weights.view(*([1] * (residual_state.dim() - 2)), -1)
        return torch.sum(residual_state * input_weights.unsqueeze(-1).to(residual_state.dtype), dim=-2)

    def forward(
        self,
        residual_state: torch.Tensor,
        update: torch.Tensor,
        *,
        return_dict: bool = True,
    ) -> Any:
        _validate_residual_state(residual_state, self.expansion_factor, self.hidden_size)
        _validate_hidden_states("update", update, self.hidden_size)
        if residual_state.shape[:-2] != update.shape[:-1]:
            raise ValueError(
                "residual_state and update must share leading dimensions, "
                f"got {tuple(residual_state.shape)} and {tuple(update.shape)}."
            )

        input_weights, output_weights, residual_mapping = self._constrained_parameters(residual_state)
        if residual_mapping.dim() > 2:
            mixed = torch.einsum("...eh,...ef->...fh", residual_state, residual_mapping)
        else:
            mixed = torch.einsum("...eh,ef->...fh", residual_state, residual_mapping)
        expanded_update = update.unsqueeze(-2) * output_weights.unsqueeze(-1).to(update.dtype)
        next_state = mixed + expanded_update.to(mixed.dtype)
        hidden_states = torch.sum(next_state * input_weights.unsqueeze(-1).to(next_state.dtype), dim=-2)
        if return_dict:
            return {
                "residual_state": next_state,
                "hidden_states": hidden_states,
                "input_weights": input_weights,
                "output_weights": output_weights,
                "residual_mapping": residual_mapping,
            }
        return next_state, hidden_states

    def _constrained_parameters(self, residual_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        leading_shape = residual_state.shape[:-2]
        input_logits = self.input_logits
        output_logits = self.output_logits
        residual_logits = self.residual_logits
        if self.dynamic:
            if self.dynamic_norm is None or self.dynamic_input is None or self.dynamic_output is None or self.dynamic_residual is None:
                raise RuntimeError("dynamic mHC parameters are not initialized.")
            context = residual_state.reshape(*leading_shape, self.expansion_factor * self.hidden_size)
            context = self.dynamic_norm(context)
            input_logits = input_logits + self.dynamic_input_gate * self.dynamic_input(context)
            output_logits = output_logits + self.dynamic_output_gate * self.dynamic_output(context)
            residual_logits = residual_logits + self.dynamic_residual_gate * self.dynamic_residual(context).view(
                *leading_shape,
                self.expansion_factor,
                self.expansion_factor,
            )
        input_weights = torch.sigmoid(input_logits)
        # DeepSeek-V4 Eq. 7: the output mapping C_l = 2*sigmoid(.), bounded in (0, 2).
        output_weights = 2.0 * torch.sigmoid(output_logits)
        residual_mapping = self.project_residual_mapping(residual_logits)
        return input_weights, output_weights, residual_mapping


def _validate_hidden_states(name: str, value: torch.Tensor, hidden_size: int) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}.")
    if value.dim() < 1:
        raise ValueError(f"{name} must have at least 1 dimension.")
    if value.shape[-1] != hidden_size:
        raise ValueError(f"{name} last dimension must be {hidden_size}, got {value.shape[-1]}.")


def _validate_residual_state(value: torch.Tensor, expansion_factor: int, hidden_size: int) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"residual_state must be a torch.Tensor, got {type(value).__name__}.")
    if value.dim() < 2:
        raise ValueError("residual_state must have at least 2 dimensions.")
    if value.shape[-2:] != (expansion_factor, hidden_size):
        raise ValueError(
            f"residual_state last dimensions must be {(expansion_factor, hidden_size)}, "
            f"got {tuple(value.shape[-2:])}."
        )


__all__ = ["ManifoldConstrainedHyperConnection"]
