"""DeepSeek-V4.1 Single-Pass mHC, report section 2.4.1, equation (6).

X_next = B(X) @ X + C(X) * F(A_prev @ X).
A(X) is passed explicitly to the NEXT sublayer, with its gradient intact.
This is an eager reference, not the fused Mega-mHC deployment kernel.
"""

from __future__ import annotations

import math
from typing import Any, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .hyper_connection import _validate_hidden_states, _validate_residual_state


class SinglePassMHC(nn.Module):
    """Predict constrained coefficients and apply the delayed-input-map update.

    Residual states have shape (..., num_branches, hidden_size). A has shape
    (..., 1, num_branches), B (..., num_branches, num_branches), and C
    (..., num_branches, 1). B is indexed [destination, source].

    There is no module-owned activation cache. Initialize A_prev once at the
    start of a model forward, then pass each sublayer's A to the next sublayer.
    Attention and FFN have separate mHC parameters and participate in the same
    chain. The final A contracts the final residual state before the head norm.

    hc_eps=0 follows the report's Sigmoid and column-then-row Sinkhorn equations.
    hc_eps>0 uses the released V4.1 kernel's epsilon-stabilized, row-then-column
    variant, including sigmoid(A_logits)+hc_eps. Both use FP32 accumulation.
    """

    def __init__(
        self,
        *,
        num_branches: int,
        hidden_size: int,
        rms_norm_eps: float = 1e-20,
        sinkhorn_iters: int = 20,
        alpha_init: float = 1e-2,
        hc_eps: float = 0.0,
    ) -> None:
        super().__init__()
        for name, value in (("num_branches", num_branches), ("hidden_size", hidden_size),
                            ("sinkhorn_iters", sinkhorn_iters)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive int.")
        for name, value in (("rms_norm_eps", rms_norm_eps), ("alpha_init", alpha_init), ("hc_eps", hc_eps)):
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        if rms_norm_eps == 0:
            raise ValueError("rms_norm_eps must be positive.")
        self.num_branches = num_branches
        self.hidden_size = hidden_size
        self.rms_norm_eps = rms_norm_eps
        self.sinkhorn_iters = sinkhorn_iters
        self.hc_eps = hc_eps
        # Packed order matches released hc_fn/hc_base: pre(A), post(C), residual(B).
        self.coeff_proj = nn.Linear(num_branches * hidden_size, (2 + num_branches) * num_branches, bias=False)
        self.norm_weight = nn.Parameter(torch.ones(num_branches * hidden_size))
        self.coeff_bias = nn.Parameter(torch.zeros((2 + num_branches) * num_branches))
        self.coeff_scale = nn.Parameter(torch.full((3,), float(alpha_init)))

    def _validate_state(self, X: torch.Tensor) -> None:
        _validate_residual_state(X, self.num_branches, self.hidden_size)
        if not X.is_floating_point():
            raise ValueError("residual_state must be floating point.")
        if X.device != self.coeff_proj.weight.device:
            raise ValueError("residual_state and mHC parameters must be on the same device.")

    def _validate_map(self, X: torch.Tensor, value: torch.Tensor, shape: tuple, name: str) -> None:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor.")
        if tuple(value.shape) != (*X.shape[:-2], *shape):
            raise ValueError(f"{name} must have shape {(*X.shape[:-2], *shape)}.")
        if value.device != X.device or not value.is_floating_point():
            raise ValueError(f"{name} must be floating point on the residual state's device.")

    def _sinkhorn(self, logits: torch.Tensor) -> torch.Tensor:
        if self.hc_eps:
            # Exact operation order of the released hc_split_sinkhorn reference.
            mapping = logits.softmax(dim=-1) + self.hc_eps
            mapping = mapping / (mapping.sum(dim=-2, keepdim=True) + self.hc_eps)
            for _ in range(self.sinkhorn_iters - 1):
                mapping = mapping / (mapping.sum(dim=-1, keepdim=True) + self.hc_eps)
                mapping = mapping / (mapping.sum(dim=-2, keepdim=True) + self.hc_eps)
            return mapping
        # Log-domain form of exp -> column normalization -> row normalization;
        # prevents overflowing exp or zeroing an entire row for large logits.
        log_mapping = logits
        for _ in range(self.sinkhorn_iters):
            log_mapping = log_mapping - torch.logsumexp(log_mapping, dim=-2, keepdim=True)
            log_mapping = log_mapping - torch.logsumexp(log_mapping, dim=-1, keepdim=True)
        return log_mapping.exp()

    def _predict_coefficients(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """H_l(X_l), with packed dynamic/static parameters and FP32 coefficients."""
        self._validate_state(X)
        with torch.autocast(device_type=X.device.type, enabled=False):
            flat = X.flatten(-2).float()
            inverse_rms = torch.rsqrt(flat.square().mean(dim=-1, keepdim=True) + self.rms_norm_eps)
            # Fold the RMS weight into the projection and divide AFTER projection.
            # This equals RMSNorm(vec(X)) @ W, and retains both parameter gradients.
            weight = self.coeff_proj.weight.float() * self.norm_weight.float().unsqueeze(0)
            projected = F.linear(flat, weight) * inverse_rms
            n = self.num_branches
            pre = projected[..., :n] * self.coeff_scale[0].float() + self.coeff_bias[:n].float()
            post = projected[..., n:2*n] * self.coeff_scale[1].float() + self.coeff_bias[n:2*n].float()
            residual = projected[..., 2*n:] * self.coeff_scale[2].float() + self.coeff_bias[2*n:].float()
            A = (pre.sigmoid() + self.hc_eps).unsqueeze(-2)
            B = self._sinkhorn(residual.reshape(*X.shape[:-2], n, n))
            C = (2 * post.sigmoid()).unsqueeze(-1)
        return A, B, C

    def init_state(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Replicate embeddings across residual streams without resetting other calls."""
        _validate_hidden_states("hidden_states", hidden_states, self.hidden_size)
        if not hidden_states.is_floating_point():
            raise ValueError("hidden_states must be floating point.")
        return hidden_states.unsqueeze(-2).expand(*hidden_states.shape[:-1], self.num_branches, self.hidden_size).clone()

    def initial_input_weights(self, X: torch.Tensor) -> torch.Tensor:
        """One-hot A_{-1}, matching official make_identity_pre_mix initialization."""
        self._validate_state(X)
        A = X.new_zeros((*X.shape[:-2], 1, self.num_branches), dtype=torch.float32)
        A[..., 0, 0] = 1
        return A

    def input_mix(self, X: torch.Tensor, A_prev: torch.Tensor) -> torch.Tensor:
        """A_{l-1} X_l; never predict or silently substitute current A_l."""
        self._validate_state(X)
        self._validate_map(X, A_prev, (1, self.num_branches), "A_prev")
        with torch.autocast(device_type=X.device.type, enabled=False):
            return torch.matmul(A_prev.float(), X.float()).squeeze(-2).to(X.dtype)

    def read_state(self, X: torch.Tensor, *, A: torch.Tensor) -> torch.Tensor:
        """Final learned contraction using the LAST sublayer's A, not a mean."""
        return self.input_mix(X, A)

    def update_state(self, X: torch.Tensor, block_output: torch.Tensor,
                     B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """B_l X_l + C_l F_l(...), using current-state coefficients."""
        self._validate_state(X)
        self._validate_map(X, block_output, (self.hidden_size,), "block_output")
        self._validate_map(X, B, (self.num_branches, self.num_branches), "B")
        self._validate_map(X, C, (self.num_branches, 1), "C")
        with torch.autocast(device_type=X.device.type, enabled=False):
            output = torch.matmul(B.float(), X.float()) + C.float() * block_output.float().unsqueeze(-2)
        return output.to(X.dtype)

    def forward(self, X: torch.Tensor, block_output: torch.Tensor, *,
                A_prev: torch.Tensor, return_coefficients: bool = False):
        """Low-level update; block_output must already equal F_l(input_mix(X,A_prev)).

        Returns (X_next, X_hat). With return_coefficients=True also returns a
        dictionary containing A_l/B_l/C_l; pass its A to the NEXT mHC instance.
        Use SinglePassMHCBlock to run the entire sublayer with one call.
        """
        X_hat = self.input_mix(X, A_prev)
        A, B, C = self._predict_coefficients(X)
        next_state = self.update_state(X, block_output, B, C)
        if return_coefficients:
            return next_state, X_hat, {"A": A, "B": B, "C": C}
        return next_state, X_hat


class SinglePassMHCBlock(nn.Module):
    """Wrap ONE attention or FFN sublayer, including its input pre-norm.

    block must map (..., hidden_size) to the same shape, and must not add its
    own residual. Supply pre-norm inside block (e.g. nn.Sequential(norm, ffn)).
    Returns (X_next, A_next), making the layer-to-layer dependency explicit.
    """

    def __init__(self, *, mhc: SinglePassMHC, block: nn.Module) -> None:
        super().__init__()
        self.mhc = mhc
        self.block = block

    def forward(self, X: torch.Tensor, *, A_prev: torch.Tensor,
                return_coefficients: bool = False, **block_kwargs: Any):
        X_hat = self.mhc.input_mix(X, A_prev)
        A, B, C = self.mhc._predict_coefficients(X)
        block_output = self.block(X_hat, **block_kwargs)
        X_next = self.mhc.update_state(X, block_output, B, C)
        if return_coefficients:
            return X_next, A, {"A": A, "B": B, "C": C,
                               "hidden_states": X_hat, "block_output": block_output}
        return X_next, A


__all__ = ["SinglePassMHC", "SinglePassMHCBlock"]
