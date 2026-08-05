from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from torchforge.common.attention import GatedMLA, KimiDeltaAttention
from torchforge.common.embedding import Embedding
from torchforge.common.lm_head import LMHead
from torchforge.common.moe import StableLatentMoE
from torchforge.common.mtp import MultiTokenPredictionModule
from torchforge.common.nn import RMSNorm, SiTUGLU
from torchforge.common.residual import BlockAttentionResidual


def num_decoder_layers(config: dict[str, Any]) -> int:
    return 4 * int(config["model"]["num_hybrid_blocks"]) + 1


def attention_kind_for_layer(config: dict[str, Any], layer_index: int) -> str:
    total = num_decoder_layers(config)
    if not 0 <= layer_index < total:
        raise ValueError(f"layer_index must be in [0, {total}), got {layer_index}.")
    if layer_index == total - 1:
        return "mla"
    return "kda" if layer_index % 4 < 3 else "mla"


class _DenseSiTUGLU(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model_cfg, moe_cfg = config["model"], config["moe"]
        hidden_size = int(model_cfg["hidden_size"])
        intermediate_size = int(model_cfg["dense_intermediate_size"])
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.activation = SiTUGLU(
            beta_gate=float(moe_cfg["beta_gate"]),
            beta_up=float(moe_cfg["beta_up"]),
        )
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.activation((self.gate_proj(hidden_states), self.up_proj(hidden_states))))


def _build_kda(config: dict[str, Any]) -> KimiDeltaAttention:
    model_cfg, kda_cfg = config["model"], config["kda"]
    return KimiDeltaAttention(
        hidden_size=int(model_cfg["hidden_size"]),
        num_heads=int(model_cfg["num_attention_heads"]),
        head_dim=int(model_cfg["head_dim"]),
        value_head_dim=int(model_cfg["head_dim"]),
        short_conv_kernel_size=int(kda_cfg["short_conv_kernel_size"]),
        decay_rank=int(kda_cfg["decay_rank"]),
        g_min=float(kda_cfg["g_min"]),
        chunk_size=int(kda_cfg["chunk_size"]),
        tile_size=int(kda_cfg["tile_size"]),
        rms_norm_eps=float(model_cfg["rms_norm_eps"]),
        backend=str(kda_cfg["backend"]),
        bias=False,
    )


def _build_mla(config: dict[str, Any]) -> GatedMLA:
    model_cfg, mla_cfg = config["model"], config["mla"]
    return GatedMLA(
        hidden_size=int(model_cfg["hidden_size"]),
        num_heads=int(model_cfg["num_attention_heads"]),
        q_lora_rank=int(mla_cfg["q_lora_rank"]),
        kv_lora_rank=int(mla_cfg["kv_lora_rank"]),
        head_dim=int(model_cfg["head_dim"]),
        value_head_dim=int(model_cfg["head_dim"]),
        rms_norm_eps=float(model_cfg["rms_norm_eps"]),
        attention_dropout=float(mla_cfg["attention_dropout"]),
        attention_backend=str(mla_cfg["attention_backend"]),
        bias=False,
    )


def _build_moe(config: dict[str, Any]) -> StableLatentMoE:
    model_cfg, moe_cfg = config["model"], config["moe"]
    return StableLatentMoE(
        hidden_size=int(model_cfg["hidden_size"]),
        latent_size=int(moe_cfg["latent_size"]),
        num_experts=int(moe_cfg["num_routed_experts"]),
        top_k=int(moe_cfg["num_experts_per_token"]),
        expert_intermediate_size=int(moe_cfg["expert_intermediate_size"]),
        num_shared_experts=int(moe_cfg["num_shared_experts"]),
        shared_intermediate_size=int(moe_cfg["shared_intermediate_size"]),
        beta_gate=float(moe_cfg["beta_gate"]),
        beta_up=float(moe_cfg["beta_up"]),
        histogram_bins=int(moe_cfg["histogram_bins"]),
        rms_norm_eps=float(model_cfg["rms_norm_eps"]),
        bias=False,
    )


class SmallK3DecoderLayer(nn.Module):
    def __init__(self, config: dict[str, Any], layer_index: int) -> None:
        super().__init__()
        model_cfg = config["model"]
        hidden_size = int(model_cfg["hidden_size"])
        eps = float(model_cfg["rms_norm_eps"])
        self.layer_index = layer_index
        self.attention_kind = attention_kind_for_layer(config, layer_index)
        self.attention_norm = RMSNorm(hidden_size, eps=eps)
        self.attention = _build_kda(config) if self.attention_kind == "kda" else _build_mla(config)
        self.ffn_norm = RMSNorm(hidden_size, eps=eps)
        self.ffn = _DenseSiTUGLU(config) if layer_index == 0 else _build_moe(config)

    def attention_update(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.attention(self.attention_norm(hidden_states))["hidden_states"]

    def ffn_update(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = self.ffn(self.ffn_norm(hidden_states))
        return output["hidden_states"] if isinstance(output, dict) else output


class _MTPDecoderBlock(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model_cfg = config["model"]
        hidden_size = int(model_cfg["hidden_size"])
        eps = float(model_cfg["rms_norm_eps"])
        self.attention_norm = RMSNorm(hidden_size, eps=eps)
        self.attention = _build_mla(config)
        self.ffn_norm = RMSNorm(hidden_size, eps=eps)
        self.ffn = _build_moe(config)

    def forward(self, hidden_states: torch.Tensor, *, return_dict: bool = True, **_: Any) -> Any:
        hidden_states = hidden_states + self.attention(self.attention_norm(hidden_states))["hidden_states"]
        hidden_states = hidden_states + self.ffn(self.ffn_norm(hidden_states))["hidden_states"]
        return {"hidden_states": hidden_states} if return_dict else hidden_states


class SmallK3Model(nn.Module):
    """Small Kimi-K3 reference model for 8-GPU DDP training."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        model_cfg = config["model"]
        hidden_size = int(model_cfg["hidden_size"])
        self.embed_tokens = Embedding(
            vocab_size=int(model_cfg["vocab_size"]),
            hidden_size=hidden_size,
        )
        self.layers = nn.ModuleList(
            SmallK3DecoderLayer(config, layer_index)
            for layer_index in range(num_decoder_layers(config))
        )
        self.attention_residual = BlockAttentionResidual(
            hidden_size=hidden_size,
            num_layers=len(self.layers),
            block_size=int(model_cfg["attnres_block_size"]),
            sublayers_per_layer=2,
            rms_norm_eps=float(model_cfg["rms_norm_eps"]),
        )
        self.final_norm = RMSNorm(hidden_size, eps=float(model_cfg["rms_norm_eps"]))
        self.lm_head = LMHead(
            hidden_size=hidden_size,
            vocab_size=int(model_cfg["vocab_size"]),
            bias=False,
        )
        if bool(model_cfg["tie_word_embeddings"]):
            self.lm_head.tie_weights(self.embed_tokens)
        if not bool(config["mtp"]["enabled"]) or int(config["mtp"]["depth"]) != 1:
            raise ValueError("SmallK3Model requires one enabled MTP layer.")
        self.mtp = MultiTokenPredictionModule(
            hidden_size=hidden_size,
            embedding=self.embed_tokens,
            transformer_block=_MTPDecoderBlock(config),
            lm_head=self.lm_head,
            bias=False,
            rms_norm_eps=float(model_cfg["rms_norm_eps"]),
        )

    def _maybe_checkpoint(self, function: Any, hidden_states: torch.Tensor) -> torch.Tensor:
        enabled = bool(self.config["train"].get("activation_checkpointing", False)) and self.training
        if not enabled:
            return function(hidden_states)
        return checkpoint(function, hidden_states, use_reentrant=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must have shape (batch, sequence).")
        embedding = self.embed_tokens(input_ids)
        residual_state = self.attention_residual.init_state(embedding)
        expert_loads = []
        for layer_index, layer in enumerate(self.layers):
            attention_input = self.attention_residual(
                residual_state,
                layer_index=layer_index,
                sublayer_index=0,
            )["hidden_states"]
            attention_output = self._maybe_checkpoint(layer.attention_update, attention_input)
            residual_state = self.attention_residual.update(
                residual_state,
                attention_output,
                layer_complete=False,
            )

            ffn_input = self.attention_residual(
                residual_state,
                layer_index=layer_index,
                sublayer_index=1,
            )["hidden_states"]
            ffn_output = self._maybe_checkpoint(layer.ffn_update, ffn_input)
            residual_state = self.attention_residual.update(
                residual_state,
                ffn_output,
                layer_complete=True,
            )
            if isinstance(layer.ffn, StableLatentMoE):
                expert_loads.append(layer.ffn.router.margin_histogram.sum().detach())

        hidden_states = self.final_norm(
            self.attention_residual.finalize(residual_state)["hidden_states"]
        )
        logits = self.lm_head(hidden_states)
        zero = logits.new_zeros(())
        lm_loss = mtp_loss = zero
        mtp_logits = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels must have the same shape as input_ids.")
            lm_loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=-100,
            )
            mtp_output = self.mtp(hidden_states, input_ids)
            mtp_logits = mtp_output["logits"]
            mtp_loss = F.cross_entropy(
                mtp_logits.reshape(-1, mtp_logits.shape[-1]),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        loss = lm_loss + float(self.config["mtp"]["loss_weight"]) * mtp_loss
        return {
            "logits": logits,
            "mtp_logits": mtp_logits,
            "hidden_states": hidden_states,
            "loss": loss,
            "lm_loss": lm_loss,
            "mtp_loss": mtp_loss,
            "recorded_router_samples": torch.stack(expert_loads).sum() if expert_loads else zero,
        }

    @torch.no_grad()
    def update_router_biases(self, *, distributed: bool = True) -> list[torch.Tensor]:
        updated = []
        seen: set[int] = set()
        for module in self.modules():
            if isinstance(module, StableLatentMoE) and id(module) not in seen:
                seen.add(id(module))
                updated.append(module.update_router_bias(distributed=distributed))
        return updated


def architecture_summary(model: SmallK3Model) -> dict[str, int]:
    attention_kinds = [layer.attention_kind for layer in model.layers]
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    inactive_expert_parameters = 0
    for module in model.modules():
        if isinstance(module, StableLatentMoE):
            expert_parameters = sum(
                parameter.numel() for parameter in module.experts.parameters()
            )
            per_expert = expert_parameters // module.num_experts
            inactive_expert_parameters += per_expert * (module.num_experts - module.top_k)
    return {
        "decoder_layers": len(model.layers),
        "kda_layers": attention_kinds.count("kda"),
        "mla_layers": attention_kinds.count("mla"),
        "dense_layers": sum(isinstance(layer.ffn, _DenseSiTUGLU) for layer in model.layers),
        "moe_layers": sum(isinstance(layer.ffn, StableLatentMoE) for layer in model.layers),
        "mtp_layers": 1,
        "total_parameters": total_parameters,
        "active_parameters": total_parameters - inactive_expert_parameters,
    }


__all__ = [
    "SmallK3DecoderLayer",
    "SmallK3Model",
    "architecture_summary",
    "attention_kind_for_layer",
    "num_decoder_layers",
]
