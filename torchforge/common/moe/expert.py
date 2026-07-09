from __future__ import annotations

from torchforge.common.mlp import GatedMLP


class ExpertMLP(GatedMLP):
    """Feed-forward expert MLP.

    A thin specialization of :class:`~torchforge.common.mlp.GatedMLP` that names
    the DeepSeek-style gated feed-forward network used as a routed MoE expert.

    Args:
        hidden_size: Size of the input and output hidden-state dimension.
        intermediate_size: Size of the intermediate feed-forward dimension.
        activation: Activation function, one of ``"silu"``, ``"gelu"``, or ``"relu"``.
        gated: Whether to use a gated MLP path.
        bias: Whether projection layers use bias.

    Forward:
        ``hidden_states`` has shape ``(..., hidden_size)``.

    Returns:
        Tensor with the same shape as ``hidden_states``.
    """


__all__ = ["ExpertMLP"]
