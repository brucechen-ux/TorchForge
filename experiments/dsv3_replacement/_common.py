from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from torchforge.common.attention import MLA
from torchforge.common.moe import MoE as TorchForgeMoE
from torchforge.common.nn import FeedForward, RMSNorm


@dataclass(frozen=True)
class TinyDSV3BlockConfig:
    hidden_size: int = 16
    num_attention_heads: int = 2
    num_key_value_heads: int = 2
    q_lora_rank: int = 4
    kv_lora_rank: int = 4
    qk_nope_head_dim: int = 4
    qk_rope_head_dim: int = 4
    v_head_dim: int = 8
    intermediate_size: int = 32
    rms_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    attention_bias: bool = False

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 2026
    steps: int = 100
    batch_size: int = 4
    seq_length: int = 8
    learning_rate: float = 1e-3
    output: str = "losses.json"
    device: str = "cpu"


@dataclass(frozen=True)
class ComponentConfig:
    attention: str = "pytorch"
    norm: str = "pytorch"
    ffn: str = "pytorch"
    kv: str = "pytorch"


def parse_train_args() -> tuple[TrainConfig, ComponentConfig, str]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attention", choices=["pytorch", "torchforge"], default="pytorch")
    parser.add_argument("--norm", choices=["pytorch", "torchforge"], default="pytorch")
    parser.add_argument("--ffn", choices=["pytorch", "torchforge", "moe"], default="pytorch")
    parser.add_argument("--kv", choices=["pytorch", "torchforge"], default="pytorch")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-length", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    components = ComponentConfig(attention=args.attention, norm=args.norm, ffn=args.ffn, kv=args.kv)
    default_output = f"experiments/dsv3_replacement/{variant_name(components)}_losses.json"
    train_config = TrainConfig(
        seed=args.seed,
        steps=args.steps,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        learning_rate=args.learning_rate,
        output=args.output or default_output,
        device=args.device,
    )
    return train_config, components, variant_name(components)


def variant_name(components: ComponentConfig) -> str:
    return (
        f"attention_{components.attention}"
        f"__ffn_{components.ffn}"
        f"__norm_{components.norm}"
        f"__kv_{components.kv}"
    )


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    input_dtype = x.dtype
    x_fp32 = x.to(torch.float32)
    x_fp32 = x_fp32 * torch.rsqrt(x_fp32.square().mean(-1, keepdim=True) + eps)
    return weight * x_fp32.to(input_dtype)


def _apply_rotary_interleaved(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if cos.shape[-1] == query.shape[-1]:
        cos = cos[..., : cos.shape[-1] // 2]
        sin = sin[..., : sin.shape[-1] // 2]
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q1, q2 = query[..., 0::2], query[..., 1::2]
    k1, k2 = key[..., 0::2], key[..., 1::2]
    query = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
    key = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)
    return query, key


class TinyRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return _rms_norm(hidden_states, self.weight, self.eps)


class TinyFFN(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class TorchForgeMoEFFN(nn.Module):
    def __init__(self, config: TinyDSV3BlockConfig) -> None:
        super().__init__()
        self.moe = TorchForgeMoE(
            hidden_size=config.hidden_size,
            num_experts=4,
            top_k=2,
            expert_intermediate_size=config.intermediate_size,
            router_score_function="softmax",
            normalize_topk=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.moe(hidden_states)["hidden_states"]


class ReferenceDSV3Attention(nn.Module):
    def __init__(self, config: TinyDSV3BlockConfig) -> None:
        super().__init__()
        self.config = config
        self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=config.attention_bias)
        self.q_a_norm_weight = nn.Parameter(torch.ones(config.q_lora_rank))
        self.q_b_proj = nn.Linear(
            config.q_lora_rank,
            config.num_attention_heads * config.qk_head_dim,
            bias=False,
        )
        self.kv_a_proj_with_mqa = nn.Linear(
            config.hidden_size,
            config.kv_lora_rank + config.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        self.kv_a_norm_weight = nn.Parameter(torch.ones(config.kv_lora_rank))
        self.kv_b_proj = nn.Linear(
            config.kv_lora_rank,
            config.num_attention_heads * (config.qk_nope_head_dim + config.v_head_dim),
            bias=False,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * config.v_head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.scaling = config.qk_head_dim**-0.5

    def forward(self, hidden_states: torch.Tensor, position_embeddings: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        config = self.config
        batch_size, seq_length = hidden_states.shape[:2]
        cos, sin = position_embeddings
        q_latent = _rms_norm(self.q_a_proj(hidden_states), self.q_a_norm_weight, config.rms_norm_eps)
        query = self.q_b_proj(q_latent)
        query = query.view(batch_size, seq_length, config.num_attention_heads, config.qk_head_dim).transpose(1, 2)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_latent, k_rot = torch.split(compressed_kv, [config.kv_lora_rank, config.qk_rope_head_dim], dim=-1)
        k_latent = _rms_norm(k_latent, self.kv_a_norm_weight, config.rms_norm_eps)
        key_value = self.kv_b_proj(k_latent)
        key_value = key_value.view(
            batch_size,
            seq_length,
            config.num_attention_heads,
            config.qk_nope_head_dim + config.v_head_dim,
        ).transpose(1, 2)
        k_pass, value = torch.split(key_value, [config.qk_nope_head_dim, config.v_head_dim], dim=-1)
        k_rot = k_rot.view(batch_size, 1, seq_length, config.qk_rope_head_dim).expand(*k_pass.shape[:-1], -1)
        key = torch.cat((k_pass, k_rot), dim=-1)

        q_pass, q_rot = torch.split(query, [config.qk_nope_head_dim, config.qk_rope_head_dim], dim=-1)
        k_pass, k_rot = torch.split(key, [config.qk_nope_head_dim, config.qk_rope_head_dim], dim=-1)
        q_rot, k_rot = _apply_rotary_interleaved(q_rot, k_rot, cos, sin)
        query = torch.cat((q_pass, q_rot), dim=-1)
        key = torch.cat((k_pass, k_rot), dim=-1)

        attention_weights = torch.matmul(query, key.transpose(2, 3)) * self.scaling
        attention_weights = F.softmax(attention_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attention_output = torch.matmul(attention_weights, value).transpose(1, 2).contiguous()
        return self.o_proj(attention_output.reshape(batch_size, seq_length, -1))


class DSV3ReplacementBlock(nn.Module):
    def __init__(self, config: TinyDSV3BlockConfig, components: ComponentConfig) -> None:
        super().__init__()
        if components.kv != "pytorch":
            # Reserved switch: DSV3 tiny block KV is still embedded inside the attention implementation.
            # Future experiments can split this without changing train.py or compare.py.
            pass
        self.components = components
        self.input_norm = _build_norm(config, components.norm)
        self.attention = _build_attention(config, components.attention)
        self.post_attention_norm = _build_norm(config, components.norm)
        self.ffn = _build_ffn(config, components.ffn)

    def forward(self, hidden_states: torch.Tensor, position_embeddings: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        hidden_states = hidden_states + self.attention(self.input_norm(hidden_states), position_embeddings)
        return hidden_states + self.ffn(self.post_attention_norm(hidden_states))


class TorchForgeDSV3Attention(nn.Module):
    def __init__(self, config: TinyDSV3BlockConfig) -> None:
        super().__init__()
        self.attention = MLA(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            attention_dropout=config.attention_dropout,
            attention_bias=config.attention_bias,
            rms_norm_eps=config.rms_norm_eps,
            query_projection_type="low_rank",
            query_pre_norm="rmsnorm",
            kv_projection_type="latent_kv_with_rope",
            kv_latent_norm="rmsnorm",
            rotary_layout="interleaved",
            rotary_application="explicit_split",
            position_source="tuple",
            attention_scaling="qk_head_dim",
            output_projection_type="linear",
        )

    def forward(self, hidden_states: torch.Tensor, position_embeddings: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return self.attention(hidden_states, position_embeddings=position_embeddings)["hidden_states"]


def build_block(block_config: TinyDSV3BlockConfig, components: ComponentConfig) -> nn.Module:
    return DSV3ReplacementBlock(block_config, components)


def _build_norm(config: TinyDSV3BlockConfig, implementation: str) -> nn.Module:
    if implementation == "pytorch":
        return TinyRMSNorm(config.hidden_size, config.rms_norm_eps)
    if implementation == "torchforge":
        return RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    raise ValueError(f"Unsupported norm implementation: {implementation!r}.")


def _build_attention(config: TinyDSV3BlockConfig, implementation: str) -> nn.Module:
    if implementation == "pytorch":
        return ReferenceDSV3Attention(config)
    if implementation == "torchforge":
        return TorchForgeDSV3Attention(config)
    raise ValueError(f"Unsupported attention implementation: {implementation!r}.")


def _build_ffn(config: TinyDSV3BlockConfig, implementation: str) -> nn.Module:
    if implementation == "pytorch":
        return TinyFFN(config.hidden_size, config.intermediate_size)
    if implementation == "torchforge":
        return FeedForward(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            activation="swiglu",
            dropout=0.0,
            bias=False,
        )
    if implementation == "moe":
        return TorchForgeMoEFFN(config)
    raise ValueError(f"Unsupported ffn implementation: {implementation!r}.")


def make_toy_batch(
    *,
    step: int,
    train_config: TrainConfig,
    block_config: TinyDSV3BlockConfig,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(train_config.seed + step)
    hidden_states = torch.randn(
        train_config.batch_size,
        train_config.seq_length,
        block_config.hidden_size,
        generator=generator,
        device=device,
    )
    target = torch.tanh(hidden_states.roll(shifts=-1, dims=1))
    cos = torch.randn(
        train_config.batch_size,
        train_config.seq_length,
        block_config.qk_rope_head_dim,
        generator=generator,
        device=device,
    )
    sin = torch.randn(
        train_config.batch_size,
        train_config.seq_length,
        block_config.qk_rope_head_dim,
        generator=generator,
        device=device,
    )
    return hidden_states, target, cos, sin


def train_model(
    *,
    train_config: TrainConfig,
    components: ComponentConfig,
    variant: str,
) -> dict[str, object]:
    torch.manual_seed(train_config.seed)
    device = torch.device(train_config.device)
    block_config = TinyDSV3BlockConfig()
    model = build_block(block_config, components).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate)
    losses: list[dict[str, float]] = []
    use_cuda_memory = device.type == "cuda" and torch.cuda.is_available()

    for step in range(train_config.steps):
        hidden_states, target, cos, sin = make_toy_batch(
            step=step,
            train_config=train_config,
            block_config=block_config,
            device=device,
        )
        if use_cuda_memory:
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        optimizer.zero_grad(set_to_none=True)
        step_start = time.perf_counter()
        forward_start = time.perf_counter()
        output = model(hidden_states, (cos, sin))
        loss = F.mse_loss(output, target)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        forward_time_ms = (time.perf_counter() - forward_start) * 1000.0
        backward_start = time.perf_counter()
        loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        backward_time_ms = (time.perf_counter() - backward_start) * 1000.0
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_time_ms = (time.perf_counter() - step_start) * 1000.0
        peak_memory_mb = (
            float(torch.cuda.max_memory_allocated(device) / (1024 * 1024)) if use_cuda_memory else 0.0
        )
        losses.append(
            {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "forward_time_ms": forward_time_ms,
                "backward_time_ms": backward_time_ms,
                "step_time_ms": step_time_ms,
                "peak_memory_mb": peak_memory_mb,
            }
        )

    return {
        "variant": variant,
        "components": asdict(components),
        "train_config": asdict(train_config),
        "block_config": asdict(block_config),
        "losses": losses,
    }


def write_json(result: dict[str, object], path: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")


def load_losses(path: str) -> list[float]:
    data = json.loads(Path(path).read_text())
    return [float(item["loss"]) for item in data["losses"]]


def load_result(path: str) -> dict[str, object]:
    return json.loads(Path(path).read_text())
