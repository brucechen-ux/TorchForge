"""DeepSeek-V4.1-Flash Single-Pass mHC (Manifold-constrained Hyper Connection).

Single-Pass mHC shifts input-mixing coefficients by one block to eliminate
data dependencies, enabling single-kernel fusion and halving activation memory traffic.

Mathematical formulation:
    X_{l+1} = B_l @ X_l + C_l @ F_l(A_{l-1} @ X_l)
    (A_l, B_l, C_l) = H(X_l)

Key difference from original mHC:
- Original: Uses A_l for input mixing (requires waiting for coefficient computation)
- Single-Pass: Uses A_{l-1} for input mixing (no dependency, allows fusion)

Reference: DeepSeek-V4.1-Flash Technical Report, Section 2.4.1
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Unbiased RMS normalization."""
    input_dtype = x.dtype
    x_fp32 = x.float()
    variance = x_fp32.square().mean(-1, keepdim=True)
    x_fp32 = x_fp32 * torch.rsqrt(variance + eps)
    return weight * x_fp32.to(input_dtype)


class SinglePassMHC(nn.Module):
    """Single-Pass Manifold-constrained Hyper Connection.
    
    Maintains n residual streams between adjacent Transformer blocks with
    single-pass fusion for efficient deployment.
    
    Args:
        num_branches: Number of residual streams (n)
        hidden_size: Hidden dimension (d)
        rms_norm_eps: RMS normalization epsilon
    """
    
    def __init__(
        self,
        *,
        num_branches: int,
        hidden_size: int,
        rms_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        
        if num_branches < 1:
            raise ValueError(f"num_branches must be >= 1, got {num_branches}")
        
        self.num_branches = num_branches
        self.hidden_size = hidden_size
        self.rms_norm_eps = rms_norm_eps
        
        # Coefficient predictor H(X_l)
        # Projects from n branches to coefficient predictions
        self.coeff_proj = nn.Linear(
            num_branches * hidden_size,
            num_branches + num_branches * num_branches + num_branches,
            bias=False,
        )
        
        # RMS norm for coefficient predictor input
        self.norm_weight = nn.Parameter(torch.ones(num_branches * hidden_size))
        
        # Cache for A_{l-1} (used for input mixing in next block)
        self.prev_A: Optional[torch.Tensor] = None
    
    def _predict_coefficients(
        self,
        X: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict coefficients (A, B, C) from residual streams.
        
        Args:
            X: [batch, seq_len, num_branches, hidden_size]
        
        Returns:
            A: [batch, seq_len, 1, num_branches] - Input mixing coefficients
            B: [batch, seq_len, num_branches, num_branches] - Residual transformation
            C: [batch, seq_len, num_branches, 1] - Output scaling coefficients
        """
        batch_size, seq_len, num_branches, hidden_size = X.shape
        
        # Flatten branches for projection
        X_flat = X.reshape(batch_size, seq_len, num_branches * hidden_size)
        
        # Apply RMS norm
        X_normed = _rms_norm(X_flat, self.norm_weight, self.rms_norm_eps)
        
        # Predict coefficients
        coeffs = self.coeff_proj(X_normed)
        # coeffs: [batch, seq_len, num_branches + num_branches^2 + num_branches]
        
        # Split into A, B, C
        n = self.num_branches
        A = coeffs[:, :, :n].unsqueeze(2)  # [batch, seq_len, 1, num_branches]
        B = coeffs[:, :, n:n + n * n].view(batch_size, seq_len, n, n)
        C = coeffs[:, :, n + n * n:].unsqueeze(3)  # [batch, seq_len, num_branches, 1]
        
        return A, B, C
    
    def forward(
        self,
        X_prev: torch.Tensor,
        Y_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of Single-Pass mHC.
        
        Computes: X_l = B_{l-1} @ X_{l-1} + C_{l-1} @ Y_{l-1}
        And produces: X_hat_l = A_{l-1} @ X_l for block input
        
        Args:
            X_prev: Previous residual streams [batch, seq_len, num_branches, hidden_size]
            Y_prev: Previous block output [batch, seq_len, hidden_size]
        
        Returns:
            X_curr: Current residual streams [batch, seq_len, num_branches, hidden_size]
            X_hat_curr: Mixed input for current block [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, num_branches, hidden_size = X_prev.shape
        
        # Predict current coefficients (A_l, B_l, C_l)
        A_curr, B_curr, C_curr = self._predict_coefficients(X_prev)
        
        # Residual update: X_l = B_{l-1} @ X_{l-1} + C_{l-1} @ Y_{l-1}
        # B: [batch, seq_len, n, n], X_prev: [batch, seq_len, n, d]
        X_curr = torch.matmul(B_curr, X_prev)  # [batch, seq_len, n, d]
        
        # Add scaled block output
        # C: [batch, seq_len, n, 1], Y_prev: [batch, seq_len, d]
        Y_expanded = Y_prev.unsqueeze(2)  # [batch, seq_len, 1, d]
        X_curr = X_curr + C_curr * Y_expanded  # Broadcasting
        
        # Input mixing: X_hat_l = A_{l-1} @ X_l
        # Use previous A if available, otherwise use current A (first layer)
        A_for_mixing = self.prev_A if self.prev_A is not None else A_curr
        
        # A_for_mixing: [batch, seq_len, 1, n], X_curr: [batch, seq_len, n, d]
        X_hat_curr = torch.matmul(A_for_mixing, X_curr).squeeze(2)
        # [batch, seq_len, d]
        
        # Cache current A for next block
        self.prev_A = A_curr.detach()
        
        return X_curr, X_hat_curr
    
    def init_state(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Initialize residual streams by replicating hidden states.
        
        Args:
            hidden_states: [batch, seq_len, hidden_size]
        
        Returns:
            X_init: [batch, seq_len, num_branches, hidden_size]
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Replicate to all branches
        X_init = hidden_states.unsqueeze(2).expand(
            batch_size, seq_len, self.num_branches, hidden_size
        ).contiguous()
        
        return X_init
    
    def read_state(
        self,
        X: torch.Tensor,
    ) -> torch.Tensor:
        """Read from residual streams (simple average for initialization).
        
        Args:
            X: [batch, seq_len, num_branches, hidden_size]
        
        Returns:
            output: [batch, seq_len, hidden_size]
        """
        return X.mean(dim=2)
    
    def reset_cache(self) -> None:
        """Reset cached A coefficients (call at sequence boundaries)."""
        self.prev_A = None


class SinglePassMHCBlock(nn.Module):
    """Complete Single-Pass mHC block wrapper.
    
    Wraps a Transformer block (attention + FFN) with Single-Pass mHC residual connection.
    
    Args:
        mhc: SinglePassMHC instance
        block: Transformer block module (should accept hidden_states and return output)
    """
    
    def __init__(
        self,
        *,
        mhc: SinglePassMHC,
        block: nn.Module,
    ) -> None:
        super().__init__()
        
        self.mhc = mhc
        self.block = block
    
    def forward(
        self,
        X_prev: torch.Tensor,
        Y_prev: torch.Tensor,
        **block_kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through mHC and block.
        
        Args:
            X_prev: Previous residual streams [batch, seq_len, num_branches, hidden_size]
            Y_prev: Previous block output [batch, seq_len, hidden_size]
            **block_kwargs: Additional arguments for the block (e.g., attention_mask)
        
        Returns:
            X_curr: Current residual streams [batch, seq_len, num_branches, hidden_size]
            Y_curr: Current block output [batch, seq_len, hidden_size]
        """
        # Update residual streams and get mixed input
        X_curr, X_hat = self.mhc(X_prev, Y_prev)
        
        # Apply Transformer block
        Y_curr = self.block(X_hat, **block_kwargs)
        
        return X_curr, Y_curr


__all__ = [
    "SinglePassMHC",
    "SinglePassMHCBlock",
]
