from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Sequence

import numpy as np
import torch
from sympy import isprime, nextprime
from tokenizers import Regex, Tokenizer, normalizers
from torch import nn


def build_compressed_token_map(tokenizer: Tokenizer) -> tuple[list[int], int]:
    """使用 Engram 的 tokenizer normalization 合并等价 token。"""
    if not isinstance(tokenizer, Tokenizer):
        raise TypeError("tokenizer must be a tokenizers.Tokenizer instance.")
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), "\ue000"),
            normalizers.Strip(),
            normalizers.Replace("\ue000", " "),
        ]
    )
    key_to_id: dict[str, int] = {}
    token_map = []
    for token_id in range(tokenizer.get_vocab_size(with_added_tokens=True)):
        text = tokenizer.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            key = tokenizer.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        if key not in key_to_id:
            key_to_id[key] = len(key_to_id)
        token_map.append(key_to_id[key])
    return token_map, len(key_to_id)


@dataclass(frozen=True)
class EngramLayout:
    layer_ids: tuple[int, ...]
    max_ngram_size: int
    n_heads: int
    head_dim: int
    primes: tuple[tuple[tuple[int, ...], ...], ...]
    num_embeddings: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.layer_ids or len(set(self.layer_ids)) != len(self.layer_ids):
            raise ValueError("layer_ids must contain distinct layer indices.")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in self.layer_ids):
            raise ValueError("layer_ids must contain nonnegative integers.")
        for name, value, minimum in (("max_ngram_size", self.max_ngram_size, 2),
                                     ("n_heads", self.n_heads, 1), ("head_dim", self.head_dim, 1)):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        if len(self.primes) != len(self.layer_ids) or len(self.num_embeddings) != len(self.layer_ids):
            raise ValueError("primes and num_embeddings must match layer_ids.")
        seen: set[int] = set()
        for layer, rows in zip(self.primes, self.num_embeddings):
            if len(layer) != self.max_ngram_size - 1 or any(len(group) != self.n_heads for group in layer):
                raise ValueError("primes must provide one modulus per layer, n-gram order and hash head.")
            sizes = [prime for group in layer for prime in group]
            if any(isinstance(prime, bool) or not isinstance(prime, int) or not isprime(prime) for prime in sizes):
                raise ValueError("Every hash modulus must be a prime integer.")
            if len(set(sizes)) != len(sizes) or seen.intersection(sizes):
                raise ValueError("Hash moduli must be distinct across all layers and heads.")
            seen.update(sizes)
            if isinstance(rows, bool) or not isinstance(rows, int) or rows < sum(sizes):
                raise ValueError("num_embeddings must cover all hash bucket ranges.")

    @classmethod
    def create(
        cls,
        layer_ids: Sequence[int],
        table_size: int,
        max_ngram_size: int = 4,
        n_heads: int = 8,
        head_dim: int = 256,
    ) -> EngramLayout:
        layer_ids = tuple(layer_ids)
        if not layer_ids or len(set(layer_ids)) != len(layer_ids) or min(layer_ids) < 0:
            raise ValueError("layer_ids must contain distinct nonnegative layer indices")
        for name, value, minimum in (("table_size", table_size, 2), ("max_ngram_size", max_ngram_size, 2),
                                     ("n_heads", n_heads, 1), ("head_dim", head_dim, 1)):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        primes = []
        seen: set[int] = set()
        for _ in layer_ids:
            layer_primes = []
            for _ in range(max_ngram_size - 1):
                current = table_size - 1
                head_primes = []
                for _ in range(n_heads):
                    current = int(nextprime(current))
                    while current in seen:
                        current = int(nextprime(current))
                    seen.add(current)
                    head_primes.append(current)
                layer_primes.append(tuple(head_primes))
            primes.append(tuple(layer_primes))
        num_embeddings = tuple(sum(sum(group) for group in layer) for layer in primes)
        return cls(layer_ids, max_ngram_size, n_heads, head_dim, tuple(primes), num_embeddings)


class EngramHash(nn.Module):
    """2..N-gram 多头寻址；history 显式保存最多 N-1 个 compressed token ID。"""
    def __init__(
        self,
        layout: EngramLayout,
        vocab_size: int,
        token_map: Optional[Sequence[int]] = None,
        pad_id: int = 2,
        expected_compressed_vocab_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        if vocab_size <= 0 or not 0 <= pad_id < vocab_size:
            raise ValueError("vocab_size and pad_id are invalid")
        mapping = torch.arange(vocab_size) if token_map is None else torch.as_tensor(token_map)
        if mapping.dtype not in (torch.int32, torch.int64) or mapping.shape != (vocab_size,):
            raise ValueError("token_map must contain one integer per vocabulary entry")
        unique = torch.unique(mapping)
        if not torch.equal(unique, torch.arange(unique.numel(), device=mapping.device)):
            raise ValueError("token_map must use contiguous nonnegative compressed ids")
        self.layout = layout
        self.vocab_size = vocab_size
        self.compressed_vocab_size = unique.numel()
        if expected_compressed_vocab_size is not None and expected_compressed_vocab_size != self.compressed_vocab_size:
            raise ValueError("The compressed vocabulary size does not match the hash configuration.")
        self.pad_id = int(mapping[pad_id])
        bound = max(1, (np.iinfo(np.int64).max // self.compressed_vocab_size) // 2)
        multipliers = []
        for layer_id in layout.layer_ids:
            generator = np.random.default_rng(10007 * layer_id)
            values = generator.integers(0, bound, size=layout.max_ngram_size, dtype=np.int64)
            multipliers.append(torch.from_numpy(values * 2 + 1))
        offsets = []
        for layer in layout.primes:
            sizes = [prime for group in layer for prime in group]
            offsets.append(torch.tensor([0, *sizes[:-1]]).cumsum(0))
        self.register_buffer("token_map", mapping.to(torch.int64).clone())
        self.register_buffer("multipliers", torch.stack(multipliers))
        self.register_buffer("primes", torch.tensor(layout.primes))
        self.register_buffer("offsets", torch.stack(offsets))

    def forward(
        self,
        input_ids: torch.Tensor,
        history: Optional[torch.Tensor] = None,
        token_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if input_ids.ndim != 2 or input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("input_ids must be an integer tensor [batch, sequence]")
        if input_ids.numel() == 0 or bool(((input_ids < 0) | (input_ids >= self.vocab_size)).any()):
            raise ValueError("input_ids must be nonempty and within vocab_size")
        if input_ids.device != self.token_map.device:
            raise ValueError("input_ids and hash buffers must share a device.")
        compressed = self.token_map[input_ids.long()]
        if token_mask is not None:
            if token_mask.shape != input_ids.shape or token_mask.dtype != torch.bool or token_mask.device != input_ids.device:
                raise ValueError("token_mask must be boolean with input_ids shape")
            compressed = compressed.masked_fill(~token_mask, -1)
        keep = self.layout.max_ngram_size - 1
        if history is None:
            history = compressed[:, :0]
        if history.ndim != 2 or history.shape[0] != input_ids.shape[0] or history.shape[1] > keep:
            raise ValueError("history must contain at most max_ngram_size - 1 compressed tokens per sample")
        if history.dtype != torch.int64 or history.device != input_ids.device:
            raise ValueError("history must be int64 and share input_ids device")
        if bool(((history < -1) | (history >= self.compressed_vocab_size)).any()):
            raise ValueError("history contains invalid compressed token ids")
        combined = torch.cat((history, compressed), dim=1)
        positions = torch.arange(input_ids.shape[1], device=input_ids.device) + history.shape[1]
        blocked = torch.zeros_like(compressed, dtype=torch.bool)
        tokens = []
        for shift in range(self.layout.max_ngram_size):
            source = combined[:, (positions - shift).clamp_min(0)]
            blocked = blocked | (positions < shift) | (source == -1)
            tokens.append(torch.where(blocked, self.pad_id, source))
        products = torch.stack(tokens, dim=-1).unsqueeze(2) * self.multipliers
        rolling = products[..., 0]
        hashes = []
        for order in range(1, self.layout.max_ngram_size):
            rolling = torch.bitwise_xor(rolling, products[..., order])
            hashes.append(rolling.unsqueeze(-1) % self.primes[:, order - 1])
        return torch.cat(hashes, dim=-1) + self.offsets, combined[:, -keep:].clone()


class Engram(nn.Module):
    """稀疏记忆查表、分支独立 key、共享 value 与 context-aware gate。"""
    def __init__(
        self,
        hidden_size: int,
        hc_mult: int,
        layout: EngramLayout,
        layer_id: int,
        norm_eps: float = 1e-20,
    ) -> None:
        super().__init__()
        if min(hidden_size, hc_mult) <= 0 or not math.isfinite(norm_eps) or norm_eps <= 0:
            raise ValueError("hidden_size, hc_mult and norm_eps must be positive")
        self.hidden_size = hidden_size
        self.hc_mult = hc_mult
        self.layer_hash_index = layout.layer_ids.index(layer_id)
        self.n_hash_cols = (layout.max_ngram_size - 1) * layout.n_heads
        self.embed = nn.Embedding(layout.num_embeddings[self.layer_hash_index], layout.head_dim)
        self.embed.optimizer_role = "engram_embedding"
        self.wkv = nn.Linear(self.n_hash_cols * layout.head_dim, hidden_size * (hc_mult + 1), bias=False)
        self.q_weight = nn.Parameter(torch.ones(hc_mult, hidden_size))
        self.k_weight = nn.Parameter(torch.ones(hc_mult, hidden_size))
        self.norm_eps = norm_eps
        self.optimizer_norm_parameter_names = ("q_weight", "k_weight")
        nn.init.normal_(self.embed.weight, std=0.02)

    def forward(
        self,
        hidden_states: torch.Tensor,
        hash_ids: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if hidden_states.ndim != 4 or hidden_states.shape[-2:] != (self.hc_mult, self.hidden_size):
            raise ValueError("hidden_states must have shape [batch, sequence, hc_mult, hidden_size]")
        if hash_ids.shape != (*hidden_states.shape[:2], self.n_hash_cols):
            raise ValueError("hash_ids must provide one column per n-gram hash head")
        if hash_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("hash_ids must contain integers")
        if hidden_states.numel() == 0 or not hidden_states.is_floating_point():
            raise ValueError("hidden_states must be nonempty and floating point.")
        if hash_ids.device != hidden_states.device or hidden_states.device != self.embed.weight.device:
            raise ValueError("Engram inputs and parameters must share a device.")
        if ((hash_ids < 0) | (hash_ids >= self.embed.num_embeddings)).any():
            raise ValueError("hash_ids are outside this layer's memory table.")
        if token_mask is not None and (token_mask.shape != hidden_states.shape[:2] or token_mask.dtype != torch.bool
                                       or token_mask.device != hidden_states.device):
            raise ValueError("token_mask must be boolean [batch, sequence]")
        kv = self.wkv(self.embed(hash_ids).flatten(-2))
        key, value = kv.split([self.hc_mult * self.hidden_size, self.hidden_size], dim=-1)
        key = key.float().unflatten(-1, (self.hc_mult, self.hidden_size))
        hidden = hidden_states.float()
        rstd = torch.rsqrt(hidden.square().mean(-1) + self.norm_eps)
        rstd = rstd * torch.rsqrt(key.square().mean(-1) + self.norm_eps)
        dot = (hidden * self.q_weight.float() * self.k_weight.float() * key).sum(-1)
        dot = dot * rstd * self.hidden_size**-0.5
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
        if token_mask is not None:
            gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
        return (hidden + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(hidden_states.dtype)


__all__ = ["Engram", "EngramHash", "EngramLayout", "build_compressed_token_map"]
