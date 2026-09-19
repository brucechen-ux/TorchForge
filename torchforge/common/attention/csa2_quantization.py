"""Portable dequantized reference for V4.1 cache quantization (report §2.4.4).

These functions emulate the values seen by attention, not packed cache storage
or low-precision GEMMs. Straight-through gradients support QAT experiments.
"""

from __future__ import annotations

import torch


def _e4m3(x: torch.Tensor) -> torch.Tensor:
    """Round to finite E4M3, including subnormals, with round-to-nearest-even."""
    magnitude = x.abs().clamp(max=448.0)
    exponent = torch.floor(torch.log2(magnitude.clamp_min(2.0**-6)))
    step = torch.pow(2.0, exponent - 3)
    return torch.copysign((magnitude / step).round() * step, x)


def _e2m1(x: torch.Tensor) -> torch.Tensor:
    # Positive E2M1 codes 0..7. At a midpoint choose the even code.
    levels = x.new_tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    midpoints = (levels[:-1] + levels[1:]) * 0.5
    magnitude = x.abs().clamp(max=6.0)
    lower = torch.bucketize(magnitude.contiguous(), midpoints)
    tie = (lower < 7) & (magnitude == midpoints[lower.clamp_max(6)])
    code = lower + (tie & (lower.remainder(2) == 1)).long()
    return torch.copysign(levels[code], x)


def fake_quantize_fp4(
    x: torch.Tensor, *, block_size: int, scale_format: str,
) -> torch.Tensor:
    """E2M1 with E4M3/16 (main KV) or power-of-two E8M0/32 (index Q/K)."""
    if block_size <= 0 or x.shape[-1] % block_size:
        raise ValueError("FP4 channel dimension must be divisible by block_size.")
    if scale_format not in ("e4m3", "e8m0"):
        raise ValueError("scale_format must be 'e4m3' or 'e8m0'.")
    if x.numel() == 0:
        return x
    with torch.no_grad():
        blocks = x.float().unflatten(-1, (-1, block_size))
        amax = blocks.abs().amax(-1, keepdim=True)
        if scale_format == "e4m3":
            scale = _e4m3(amax.clamp_min(6 * 2.0**-9) / 6)
        else:
            scale = torch.pow(2.0, torch.ceil(torch.log2(amax.clamp_min(6 * 2.0**-126) / 6)))
        quantized = (_e2m1(blocks / scale) * scale).flatten(-2).to(x.dtype)
    return x + (quantized - x).detach()


def fake_quantize_swa(x: torch.Tensor) -> torch.Tensor:
    """FP8 E4M3 SWA values, with one power-of-two scale per 32 channels."""
    if x.shape[-1] % 32:
        raise ValueError("FP8 SWA channel dimension must be divisible by 32.")
    if x.numel() == 0:
        return x
    with torch.no_grad():
        blocks = x.float().unflatten(-1, (-1, 32))
        amax = blocks.abs().amax(-1, keepdim=True).clamp_min(1e-4)
        scale = torch.pow(2.0, torch.ceil(torch.log2(amax / 448)))
        quantized = (_e4m3(blocks / scale) * scale).flatten(-2).to(x.dtype)
    return x + (quantized - x).detach()
