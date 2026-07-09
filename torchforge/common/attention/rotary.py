from __future__ import annotations

from typing import Tuple

import torch


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def rotate_half_interleaved(x: torch.Tensor) -> torch.Tensor:
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


def apply_rotary_standard(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    query = (query * cos) + (rotate_half(query) * sin)
    key = (key * cos) + (rotate_half(key) * sin)
    return query, key


def apply_rotary_interleaved(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if cos.shape[-1] == query.shape[-1]:
        cos = cos[..., : cos.shape[-1] // 2]
        sin = sin[..., : sin.shape[-1] // 2]
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q1, q2 = query[..., 0::2], query[..., 1::2]
    k1, k2 = key[..., 0::2], key[..., 1::2]
    query = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
    key = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)
    return query, key


def apply_partial_trailing_rotary_interleaved(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.repeat_interleave(2, dim=-1).unsqueeze(1)
    sin = sin.repeat_interleave(2, dim=-1).unsqueeze(1)
    rotary_dim = cos.shape[-1]
    q_pass, q_rot = query[..., :-rotary_dim], query[..., -rotary_dim:]
    k_pass, k_rot = key[..., :-rotary_dim], key[..., -rotary_dim:]
    q_rot = (q_rot.float() * cos) + (rotate_half_interleaved(q_rot).float() * sin)
    k_rot = (k_rot.float() * cos) + (rotate_half_interleaved(k_rot).float() * sin)
    return (
        torch.cat([q_pass, q_rot.to(query.dtype)], dim=-1),
        torch.cat([k_pass, k_rot.to(key.dtype)], dim=-1),
    )


def apply_rotary_interleaved_single(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos.repeat_interleave(2, dim=-1).unsqueeze(1)
    sin = sin.repeat_interleave(2, dim=-1).unsqueeze(1)
    rope_dim = cos.shape[-1]
    nope, rope = x[..., :-rope_dim], x[..., -rope_dim:]
    rotated = (rope.float() * cos) + (rotate_half_interleaved(rope).float() * sin)
    return torch.cat([nope, rotated.to(x.dtype)], dim=-1)


def rotary_embeddings(
    x: torch.Tensor,
    *,
    position_ids: torch.Tensor,
    head_dim: int,
    partial_rotary_factor: float,
    theta: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    dim = int(head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=x.device) / dim))
    inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
    position_ids_expanded = position_ids[:, None, :].float()
    freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
    return freqs.cos().to(dtype=x.dtype), freqs.sin().to(dtype=x.dtype)


__all__ = [
    "apply_partial_trailing_rotary_interleaved",
    "apply_rotary_interleaved",
    "apply_rotary_interleaved_single",
    "apply_rotary_standard",
    "rotate_half",
    "rotate_half_interleaved",
    "rotary_embeddings",
]
