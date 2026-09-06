"""Qwen3.8-Flash-Next 4-way Gated Residual - simplified residual connection."""

from __future__ import annotations

from typing import Any, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from torchforge.common.nn import RMSNorm


class QwenGatedResidual(nn.Module):
    """Qwen3.8-Flash-Next 4-way Gated Residual connection.
    
    Simplification of mHC (Manifold-constrained Hyper-Connection):
        - Keeps 4-way residual branches
        - Removes 4×4 branch mixing (Sinkhorn projection)
        - Uses element-wise dynamic Read gates
        - Uses per-branch scalar Write gates
    
    Mathematical principle:
        Read phase:
            g_read = sigmoid(W_read @ x_in)  # Element-wise, per-branch
            x_module_in = sum_i (g_read[:, i] * branch[i])
        
        Block computation:
            x_module_out = Block(x_module_in)
        
        Write phase:
            g_write = scalar_gate[i] for each branch i
            branch[i] = branch[i] + g_write[i] * x_module_out
    
    Key differences from mHC:
        1. No 4×4 branch-to-branch mixing matrix
        2. Element-wise Read instead of vector Read weights
        3. Scalar Write instead of vector Write weights
        4. No Sinkhorn-Knopp projection overhead
        5. Lower memory traffic, simpler gradient flow
    
    Trade-offs (from Qwen tech report):
        + Lower memory access and computation
        + Simpler training dynamics
        + FP8-friendly (gates can quantize well)
        - Less expressive cross-branch mixing
        - Relies on element-wise granularity to compensate
    
    Args:
        hidden_size: Model hidden dimension.
        num_branches: Number of residual branches (default 4).
        bottleneck_rank: Low-rank bottleneck for Read gate (default 320).
        eps: Small constant for numerical stability.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_branches: int = 4,
        bottleneck_rank: int = 320,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if num_branches <= 0:
            raise ValueError(f"num_branches must be positive, got {num_branches}")
        if bottleneck_rank <= 0:
            raise ValueError(f"bottleneck_rank must be positive, got {bottleneck_rank}")
        
        self.hidden_size = hidden_size
        self.num_branches = num_branches
        self.bottleneck_rank = bottleneck_rank
        self.eps = eps
        
        # Element-wise Read gate: low-rank projection + per-branch gates
        # This is more efficient than full hidden_size × num_branches matrix
        self.read_down_proj = nn.Linear(hidden_size, bottleneck_rank, bias=False)
        self.read_gate_proj = nn.Linear(bottleneck_rank, num_branches * hidden_size, bias=True)
        
        # Per-branch scalar Write gates (learnable parameters)
        self.write_gates = nn.Parameter(torch.ones(num_branches))
        
        # Optional: RMSNorm for stabilization (Qwen uses this)
        self.branch_norm = RMSNorm(hidden_size, eps=eps)
    
    def init_state(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Initialize 4-way residual state from standard hidden states.
        
        Args:
            hidden_states: (..., hidden_size)
        
        Returns:
            residual_state: (..., num_branches, hidden_size)
        """
        if not isinstance(hidden_states, torch.Tensor):
            raise TypeError("hidden_states must be a Tensor")
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"hidden_states last dimension must be {self.hidden_size}, "
                f"got {hidden_states.shape[-1]}"
            )
        
        # Replicate across branches
        state = hidden_states.unsqueeze(-2).expand(
            *hidden_states.shape[:-1],
            self.num_branches,
            self.hidden_size
        )
        return state.clone()
    
    def read(self, residual_state: torch.Tensor) -> torch.Tensor:
        """Read from 4-way residual state using element-wise dynamic gates.
        
        Mathematical principle:
            1. Compute element-wise, per-branch gates: (..., num_branches, hidden_size)
            2. Apply sigmoid activation for [0, 1] range
            3. Weighted sum across branches: sum_i (gate[i] * branch[i])
        
        Args:
            residual_state: (..., num_branches, hidden_size)
        
        Returns:
            hidden_states: (..., hidden_size)
        """
        _validate_residual_state(residual_state, self.num_branches, self.hidden_size)
        
        # Compute element-wise Read gates
        # First: low-rank projection to reduce parameters
        # residual_state: (..., num_branches, hidden_size)
        # We average across branches for the gate computation
        branch_avg = residual_state.mean(dim=-2)  # (..., hidden_size)
        
        # Low-rank bottleneck
        gate_hidden = self.read_down_proj(branch_avg)  # (..., bottleneck_rank)
        
        # Expand to per-branch, per-element gates
        gates = self.read_gate_proj(gate_hidden)  # (..., num_branches * hidden_size)
        gates = gates.view(*residual_state.shape[:-2], self.num_branches, self.hidden_size)
        
        # Apply sigmoid for [0, 1] gating
        gates = torch.sigmoid(gates)
        
        # Weighted sum: element-wise multiplication then sum across branches
        hidden_states = (residual_state * gates).sum(dim=-2)
        
        return hidden_states
    
    def write(
        self,
        residual_state: torch.Tensor,
        module_output: torch.Tensor,
    ) -> torch.Tensor:
        """Write module output back to 4-way residual state using scalar gates.
        
        Mathematical principle:
            For each branch i:
                branch[i] = branch[i] + write_gate[i] * module_output
        
        Args:
            residual_state: (..., num_branches, hidden_size)
            module_output: (..., hidden_size)
        
        Returns:
            next_state: (..., num_branches, hidden_size)
        """
        _validate_residual_state(residual_state, self.num_branches, self.hidden_size)
        _validate_hidden_states("module_output", module_output, self.hidden_size)
        
        if residual_state.shape[:-2] != module_output.shape[:-1]:
            raise ValueError(
                f"residual_state and module_output batch shapes must match: "
                f"{residual_state.shape[:-2]} vs {module_output.shape[:-1]}"
            )
        
        # Normalize module output for stability
        module_output = self.branch_norm(module_output)
        
        # Apply per-branch scalar Write gates
        # write_gates: (num_branches,)
        # module_output: (..., hidden_size)
        # Broadcast module_output across branches, scale by write_gates
        write_gates = torch.sigmoid(self.write_gates)  # [0, 1] range for stability
        write_gates = write_gates.view(1, self.num_branches, 1)  # (1, num_branches, 1)
        
        # Expand module_output: (..., hidden_size) -> (..., num_branches, hidden_size)
        module_output_expanded = module_output.unsqueeze(-2).expand_as(residual_state)
        
        # Write: branch[i] += write_gate[i] * module_output
        delta = write_gates * module_output_expanded
        next_state = residual_state + delta
        
        return next_state
    
    def forward(
        self,
        residual_state: torch.Tensor,
        module_output: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full Gated Residual cycle: write module output, then read for next module.
        
        This is the typical usage pattern:
            1. Write current module output to residual state
            2. Read from updated state to get input for next module
        
        Args:
            residual_state: (..., num_branches, hidden_size)
            module_output: (..., hidden_size)
        
        Returns:
            next_state: (..., num_branches, hidden_size)
            next_input: (..., hidden_size)
        """
        # Write phase
        next_state = self.write(residual_state, module_output)
        
        # Read phase
        next_input = self.read(next_state)
        
        return next_state, next_input


def _validate_hidden_states(name: str, value: torch.Tensor, hidden_size: int) -> None:
    """Validate hidden states tensor."""
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}.")
    if value.shape[-1] != hidden_size:
        raise ValueError(
            f"{name} last dimension must be {hidden_size}, got {value.shape[-1]}."
        )


def _validate_residual_state(
    value: torch.Tensor,
    num_branches: int,
    hidden_size: int,
) -> None:
    """Validate residual state tensor."""
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"residual_state must be a torch.Tensor, got {type(value).__name__}.")
    if len(value.shape) < 2:
        raise ValueError(
            f"residual_state must have at least 2 dimensions (num_branches, hidden_size), "
            f"got shape {value.shape}."
        )
    if value.shape[-2] != num_branches:
        raise ValueError(
            f"residual_state second-to-last dimension must be {num_branches}, "
            f"got {value.shape[-2]}."
        )
    if value.shape[-1] != hidden_size:
        raise ValueError(
            f"residual_state last dimension must be {hidden_size}, got {value.shape[-1]}."
        )


# Aliases
GatedResidual = QwenGatedResidual


__all__ = [
    "QwenGatedResidual",
    "GatedResidual",
]
