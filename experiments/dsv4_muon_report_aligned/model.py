from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from torchforge.common.attention import CSACompressor, HCACompressor, MLA, SlidingWindowCausalMask
from torchforge.common.embedding import Embedding, RotaryEmbedding
from torchforge.common.lm_head import LMHead
from torchforge.common.moe import HashRouter, SharedExpertMLP
from torchforge.common.mtp import MultiTokenPredictionModule
from torchforge.common.nn import RMSNorm
from torchforge.common.position import PositionIds


class ReportAlignedTopKRouter(nn.Module):
    """V4 top-k router with unconditional selected-score normalization."""

    def __init__(self, hidden_size: int, num_experts: int, top_k: int, route_scale: float, correction_bias: bool) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.route_scale = route_scale
        self.proj = nn.Linear(hidden_size, num_experts, bias=False)
        if correction_bias:
            self.register_buffer("e_score_correction_bias", torch.zeros(num_experts), persistent=True)
        else:
            self.register_buffer("e_score_correction_bias", None)

    def forward(self, hidden_states: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = self.proj(hidden_states.float())
        scores = torch.sqrt(F.softplus(logits))
        selection_scores = scores
        if self.e_score_correction_bias is not None:
            selection_scores = selection_scores + self.e_score_correction_bias
        selected = torch.topk(selection_scores, self.top_k, dim=-1, sorted=False).indices
        weights = scores.gather(-1, selected)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1.0e-20)
        return {
            "routing_weights": (weights * self.route_scale).to(hidden_states.dtype),
            "selected_experts": selected,
            "router_scores": scores,
            "router_logits": logits,
        }


class ReportAlignedHashRouter(nn.Module):
    """Fixed token-to-expert selection with learned selected-expert scoring."""

    def __init__(self, hidden_size: int, num_experts: int, top_k: int, route_scale: float) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.route_scale = route_scale
        self.proj = nn.Linear(hidden_size, num_experts, bias=False)
        self.hash_router = HashRouter(num_experts=num_experts, top_k=top_k, seed=0)

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = self.proj(hidden_states.float())
        scores = torch.sqrt(F.softplus(logits))
        selected = self.hash_router(input_ids.reshape(-1), return_dict=True)["selected_experts"]
        weights = scores.gather(-1, selected)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1.0e-20)
        return {
            "routing_weights": (weights * self.route_scale).to(hidden_states.dtype),
            "selected_experts": selected,
            "router_scores": scores,
            "router_logits": logits,
        }


class ReportAlignedPackedExperts(nn.Module):
    """Packed 3-D expert parameters; axis zero is the logical-matrix axis."""

    def __init__(self, hidden_size: int, intermediate_size: int, num_experts: int, clamp_limit: float) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.intermediate_size = intermediate_size
        self.clamp_limit = clamp_limit
        self.gate_up_proj = nn.Parameter(torch.empty(num_experts, 2 * intermediate_size, hidden_size))
        self.down_proj = nn.Parameter(torch.empty(num_experts, hidden_size, intermediate_size))
        nn.init.kaiming_uniform_(self.gate_up_proj, a=5**0.5)
        nn.init.kaiming_uniform_(self.down_proj, a=5**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = torch.zeros_like(hidden_states)
        expert_load = torch.zeros(self.num_experts, device=hidden_states.device, dtype=torch.float32)
        with torch.no_grad():
            mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
            active = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_tensor in active:
            expert_id = expert_tensor[0]
            route_pos, token_pos = torch.where(mask[expert_id])
            gate_up = F.linear(hidden_states[token_pos], self.gate_up_proj[expert_id])
            gate, up = gate_up.chunk(2, dim=-1)
            gate = gate.clamp(max=self.clamp_limit)
            up = up.clamp(min=-self.clamp_limit, max=self.clamp_limit)
            current = F.linear(F.silu(gate) * up, self.down_proj[expert_id])
            current = current * routing_weights[token_pos, route_pos, None]
            output.index_add_(0, token_pos, current.to(output.dtype))
            expert_load[expert_id] = float(token_pos.numel())
        return output, expert_load


class ReportAlignedMoE(nn.Module):
    def __init__(self, config: dict[str, Any], *, hash_routing: bool) -> None:
        super().__init__()
        model_cfg, moe_cfg = config["model"], config["moe"]
        hidden_size = int(model_cfg["hidden_size"])
        num_experts = int(moe_cfg["num_routed_experts"])
        top_k = int(moe_cfg["num_experts_per_token"])
        route_scale = float(moe_cfg["route_scale"])
        self.num_experts = num_experts
        self.aux_loss_weight = float(moe_cfg["aux_loss_weight"])
        self.hash_routing = hash_routing
        if hash_routing:
            self.router = ReportAlignedHashRouter(hidden_size, num_experts, top_k, route_scale)
        else:
            self.router = ReportAlignedTopKRouter(
                hidden_size,
                num_experts,
                top_k,
                route_scale,
                correction_bias=bool(moe_cfg["use_correction_bias"]),
            )
        self.experts = ReportAlignedPackedExperts(
            hidden_size,
            int(moe_cfg["expert_intermediate_size"]),
            num_experts,
            float(moe_cfg["swiglu_limit"]),
        )
        self.shared_experts = SharedExpertMLP(
            hidden_size=hidden_size,
            intermediate_size=int(moe_cfg["expert_intermediate_size"]) * int(moe_cfg["num_shared_experts"]),
            activation="silu",
            gated=True,
            bias=False,
            clamp_limit=float(moe_cfg["swiglu_limit"]),
        )

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        original_shape = hidden_states.shape
        flat = hidden_states.reshape(-1, original_shape[-1])
        routed = self.router(flat, input_ids) if self.hash_routing else self.router(flat)
        routed_output, expert_load = self.experts(
            flat,
            routed["selected_experts"],
            routed["routing_weights"].to(flat.dtype),
        )
        output = routed_output + self.shared_experts(flat).to(flat.dtype)
        load = expert_load / expert_load.sum().clamp_min(1.0)
        importance = routed["router_scores"]
        importance = importance / importance.sum(dim=-1, keepdim=True).clamp_min(1.0e-9)
        aux_loss = flat.new_zeros(())
        if self.aux_loss_weight:
            aux_loss = self.aux_loss_weight * self.num_experts * (importance.mean(0) * load.detach()).sum()
        stats = {
            "router_entropy": (-importance.clamp_min(1.0e-9) * importance.clamp_min(1.0e-9).log()).sum(-1).mean(),
            "expert_load_variance": load.var(unbiased=False),
        }
        return output.reshape(original_shape), aux_loss, stats


class _HCAAdapter(nn.Module):
    def __init__(self, compressor: HCACompressor) -> None:
        super().__init__()
        self.compressor = compressor

    def forward(self, hidden_states: torch.Tensor, q_residual: torch.Tensor, position_ids: torch.Tensor):
        del q_residual
        return self.compressor(hidden_states, position_ids=position_ids)


class _CSAAdapter(nn.Module):
    def __init__(self, compressor: CSACompressor) -> None:
        super().__init__()
        self.compressor = compressor

    def forward(self, hidden_states: torch.Tensor, q_residual: torch.Tensor, position_ids: torch.Tensor):
        return self.compressor(hidden_states, q_residual=q_residual, position_ids=position_ids)


def attention_kind_for_layer(num_main_layers: int, layer_idx: int) -> str:
    if layer_idx >= num_main_layers:
        return "sliding"
    if layer_idx < 2:
        return "hca"
    return "csa" if (layer_idx - 2) % 2 == 0 else "hca"


def _build_attention(config: dict[str, Any], layer_idx: int) -> MLA:
    model_cfg, attn_cfg = config["model"], config["v4_attention"]
    hidden_size = int(model_cfg["hidden_size"])
    head_dim = int(attn_cfg["head_dim"])
    rope_dim = int(attn_cfg["qk_rope_head_dim"])
    factor = rope_dim / head_dim
    kind = attention_kind_for_layer(int(model_cfg["num_layers"]), layer_idx)
    compressor: nn.Module | None = None
    if kind == "hca":
        compressor = _HCAAdapter(
            HCACompressor(
                hidden_size=hidden_size,
                head_dim=head_dim,
                compress_rate=int(attn_cfg["compress_rates"]["heavily_compressed_attention"]),
                partial_rotary_factor=factor,
                rope_theta=float(attn_cfg["compress_rope_theta"]),
                rms_norm_eps=float(model_cfg["rms_norm_eps"]),
            )
        )
    elif kind == "csa":
        compressor = _CSAAdapter(
            CSACompressor(
                hidden_size=hidden_size,
                q_lora_rank=int(attn_cfg["q_lora_rank"]),
                head_dim=head_dim,
                index_num_heads=int(attn_cfg["index_n_heads"]),
                index_head_dim=int(attn_cfg["index_head_dim"]),
                index_top_k=int(attn_cfg["index_topk"]),
                compress_rate=int(attn_cfg["compress_rates"]["compressed_sparse_attention"]),
                partial_rotary_factor=factor,
                rope_theta=float(attn_cfg["compress_rope_theta"]),
                rms_norm_eps=float(model_cfg["rms_norm_eps"]),
            )
        )
    return MLA(
        hidden_size=hidden_size,
        num_attention_heads=int(model_cfg["num_attention_heads"]),
        num_key_value_heads=1,
        q_lora_rank=int(attn_cfg["q_lora_rank"]),
        kv_lora_rank=None,
        qk_nope_head_dim=head_dim - rope_dim,
        qk_rope_head_dim=rope_dim,
        v_head_dim=head_dim,
        kv_compressor=compressor,
        attention_dropout=0.0,
        rms_norm_eps=float(model_cfg["rms_norm_eps"]),
        query_projection_type="low_rank",
        query_pre_norm="rmsnorm",
        query_post_norm="unweighted_rmsnorm",
        query_store_residual=True,
        kv_projection_type="direct_kv",
        kv_final_norm="rmsnorm",
        kv_value_mode="shared_with_key",
        rotary_layout="interleaved",
        rotary_application="partial_trailing",
        position_source="tuple",
        store_latest_position=True,
        kv_value_policy="value_equals_key_after_position",
        attention_bias_policy="append_block_bias",
        pad_attention_bias_to_kv_length=True,
        repeat_kv=True,
        attention_sinks=True,
        attention_scaling="v_head_dim",
        attention_implementation=str(attn_cfg.get("attention_implementation", "sdpa")),
        output_projection_type="grouped_low_rank",
        pre_output_transform="inverse_rope",
        o_groups=int(attn_cfg["o_groups"]),
        o_lora_rank=int(attn_cfg["o_lora_rank"]),
    )


class ReportAlignedDecoderLayer(nn.Module):
    def __init__(self, config: dict[str, Any], layer_idx: int) -> None:
        super().__init__()
        model_cfg = config["model"]
        self.layer_idx = layer_idx
        self.attention_kind = attention_kind_for_layer(int(model_cfg["num_layers"]), layer_idx)
        self.attn_norm = RMSNorm(int(model_cfg["hidden_size"]), eps=float(model_cfg["rms_norm_eps"]))
        self.self_attn = _build_attention(config, layer_idx)
        self.ffn_norm = RMSNorm(int(model_cfg["hidden_size"]), eps=float(model_cfg["rms_norm_eps"]))
        self.ffn = ReportAlignedMoE(config, hash_routing=layer_idx < int(config["moe"]["num_hash_layers"]))

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        attention_output = self.self_attn(
            self.attn_norm(hidden_states),
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )["hidden_states"]
        hidden_states = hidden_states + attention_output
        ffn_output, aux_loss, stats = self.ffn(self.ffn_norm(hidden_states), input_ids)
        return hidden_states + ffn_output, aux_loss, stats


class ReportAlignedMTPBlock(nn.Module):
    def __init__(self, layer: ReportAlignedDecoderLayer) -> None:
        super().__init__()
        self.layer = layer
        self.last_aux_loss: torch.Tensor | None = None
        self.last_stats: dict[str, torch.Tensor] = {}

    def forward(self, hidden_states: torch.Tensor, *, return_dict: bool = True, **kwargs: Any) -> Any:
        output, aux_loss, stats = self.layer(hidden_states, **kwargs)
        self.last_aux_loss = aux_loss
        self.last_stats = stats
        return {"hidden_states": output} if return_dict else output


class ReportAlignedDeepSeekV4(nn.Module):
    """TorchForge assembly matching the supplied reduced V4-like report package."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        model_cfg, attn_cfg = config["model"], config["v4_attention"]
        hidden_size = int(model_cfg["hidden_size"])
        self.embed_tokens = Embedding(vocab_size=int(model_cfg["vocab_size"]), hidden_size=hidden_size)
        self.position_ids = PositionIds()
        rotary_args = {
            "head_dim": int(attn_cfg["head_dim"]),
            "partial_rotary_factor": int(attn_cfg["qk_rope_head_dim"]) / int(attn_cfg["head_dim"]),
            "max_position_embeddings": int(model_cfg["seq_len"]),
        }
        self.main_rotary_emb = RotaryEmbedding(rope_theta=float(attn_cfg["rope_theta"]), **rotary_args)
        self.compress_rotary_emb = RotaryEmbedding(rope_theta=float(attn_cfg["compress_rope_theta"]), **rotary_args)
        self.sliding_mask = SlidingWindowCausalMask(window_size=int(attn_cfg["sliding_window"]))
        self.layers = nn.ModuleList(
            ReportAlignedDecoderLayer(config, layer_idx) for layer_idx in range(int(model_cfg["num_layers"]))
        )
        self.final_norm = RMSNorm(hidden_size, eps=float(model_cfg["rms_norm_eps"]))
        self.lm_head = LMHead(hidden_size=hidden_size, vocab_size=int(model_cfg["vocab_size"]), bias=False)
        if bool(model_cfg["tie_word_embeddings"]):
            self.lm_head.tie_weights(self.embed_tokens)
        mtp_depth = int(config["mtp"]["mtp_depth"]) if config["mtp"]["enabled"] else 0
        if mtp_depth != 1:
            raise ValueError("The report-aligned experiment requires mtp_depth=1.")
        mtp_block = ReportAlignedMTPBlock(ReportAlignedDecoderLayer(config, int(model_cfg["num_layers"])))
        self.mtp = MultiTokenPredictionModule(
            hidden_size=hidden_size,
            embedding=self.embed_tokens,
            transformer_block=mtp_block,
            lm_head=self.lm_head,
            bias=False,
            rms_norm_eps=float(model_cfg["rms_norm_eps"]),
        )

    def _position_embeddings(self, position_ids: torch.Tensor, dtype: torch.dtype) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        return {
            "main": tuple(item.to(dtype) for item in self.main_rotary_emb(position_ids)),
            "compress": tuple(item.to(dtype) for item in self.compress_rotary_emb(position_ids)),
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        hidden_states = self.embed_tokens(input_ids)
        position_ids = self.position_ids(input_ids)
        positions = self._position_embeddings(position_ids, hidden_states.dtype)
        causal_mask = self.sliding_mask(input_ids, dtype=hidden_states.dtype)
        if attention_mask is not None:
            causal_mask = causal_mask.masked_fill(~attention_mask[:, None, None, :].bool(), float("-inf"))
        aux_losses: list[torch.Tensor] = []
        router_entropies: list[torch.Tensor] = []
        load_variances: list[torch.Tensor] = []
        use_checkpoint = bool(self.config["train"].get("activation_checkpointing", False)) and self.training
        for layer in self.layers:
            position_embeddings = positions["main" if layer.attention_kind == "sliding" else "compress"]
            if use_checkpoint:
                hidden_states, aux_loss, stats = checkpoint(
                    lambda value, current=layer, pe=position_embeddings: current(
                        value,
                        input_ids=input_ids,
                        position_ids=position_ids,
                        position_embeddings=pe,
                        attention_mask=causal_mask,
                    ),
                    hidden_states,
                    use_reentrant=False,
                )
            else:
                hidden_states, aux_loss, stats = layer(
                    hidden_states,
                    input_ids=input_ids,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                    attention_mask=causal_mask,
                )
            aux_losses.append(aux_loss)
            router_entropies.append(stats["router_entropy"])
            load_variances.append(stats["expert_load_variance"])
        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)

        zero = hidden_states.new_zeros(())
        lm_loss = mtp_loss = zero
        if labels is not None:
            lm_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)
            mtp_outputs = self.mtp(
                hidden_states,
                input_ids,
                attention_mask=causal_mask,
                position_ids=position_ids,
                position_embeddings=positions["main"],
            )
            mtp_logits = mtp_outputs["logits"]
            mtp_loss = F.cross_entropy(
                mtp_logits.reshape(-1, mtp_logits.shape[-1]),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
            mtp_block = self.mtp.transformer_block
            aux_losses.append(mtp_block.last_aux_loss if mtp_block.last_aux_loss is not None else zero)
            if mtp_block.last_stats:
                router_entropies.append(mtp_block.last_stats["router_entropy"])
                load_variances.append(mtp_block.last_stats["expert_load_variance"])
        aux_loss = torch.stack([value.float() for value in aux_losses]).mean().to(hidden_states.dtype)
        total_loss = lm_loss + float(self.config["mtp"]["mtp_loss_weight"]) * mtp_loss + aux_loss
        return {
            "logits": logits,
            "hidden_states": hidden_states,
            "loss": total_loss,
            "lm_loss": lm_loss,
            "mtp_loss": mtp_loss,
            "aux_loss": aux_loss,
            "router_entropy": torch.stack([value.float() for value in router_entropies]).mean(),
            "expert_load_variance": torch.stack([value.float() for value in load_variances]).mean(),
        }


@dataclass
class WeightMappingReport:
    copied: list[tuple[str, str]]
    ignored_reference: list[str]
    missing_local_parameters: list[str]


def _map_attention_suffix(suffix: str) -> str:
    direct = {
        "q_a_proj.weight": "query_projection.q_a_proj.weight",
        "q_a_norm.weight": "query_projection.q_a_norm_weight",
        "q_b_proj.weight": "query_projection.q_b_proj.weight",
        "kv_proj.weight": "kv_projection.kv_proj.weight",
        "kv_norm.weight": "kv_projection.kv_norm_weight",
        "o_a_proj.weight": "output_projection.o_a_proj.weight",
        "o_b_proj.weight": "output_projection.o_b_proj.weight",
        "sinks": "attention_backend.sinks",
    }
    if suffix in direct:
        return direct[suffix]
    if suffix.startswith("compressor."):
        compressor_suffix = suffix[len("compressor.") :].replace("kv_norm.weight", "kv_norm_weight")
        return "kv_augment.compressor.compressor." + compressor_suffix
    raise KeyError(suffix)


def _map_layer_name(reference_name: str, local_prefix: str) -> str:
    if reference_name.startswith("attn_norm.") or reference_name.startswith("ffn_norm."):
        return local_prefix + reference_name
    if reference_name.startswith("attn."):
        return local_prefix + "self_attn." + _map_attention_suffix(reference_name[len("attn.") :])
    if reference_name.startswith("ffn.gate."):
        suffix = reference_name[len("ffn.gate.") :]
        if suffix == "weight":
            return local_prefix + "ffn.router.proj.weight"
        if suffix == "e_score_correction_bias":
            return local_prefix + "ffn.router.e_score_correction_bias"
        raise KeyError(reference_name)
    if reference_name.startswith("ffn.experts."):
        return local_prefix + reference_name
    if reference_name.startswith("ffn.shared_experts."):
        suffix = reference_name[len("ffn.shared_experts.") :]
        return local_prefix + "ffn.shared_experts.expert." + suffix
    raise KeyError(reference_name)


@torch.no_grad()
def load_reference_weights(model: ReportAlignedDeepSeekV4, reference_state: dict[str, torch.Tensor]) -> WeightMappingReport:
    """Map the supplied reference package's weights into the TorchForge assembly."""

    local_state = model.state_dict()
    copied: list[tuple[str, str]] = []
    ignored: list[str] = []
    mapped_parameter_ids: set[int] = set()
    local_parameters = dict(model.named_parameters())
    for reference_name, tensor in reference_state.items():
        target: str | None = None
        if reference_name == "embed_tokens.weight":
            target = "embed_tokens.embedding.weight"
        elif reference_name == "final_norm.weight":
            target = reference_name
        elif reference_name == "lm_head.weight":
            target = "lm_head.proj.weight"
        else:
            layer_match = re.match(r"layers\.(\d+)\.(.+)", reference_name)
            mtp_match = re.match(r"mtp_modules\.0\.(.+)", reference_name)
            if layer_match:
                target = _map_layer_name(layer_match.group(2), f"layers.{layer_match.group(1)}.")
            elif mtp_match:
                mtp_name = mtp_match.group(1)
                if mtp_name == "enorm.weight":
                    target = "mtp.embedding_norm.weight"
                elif mtp_name == "hnorm.weight":
                    target = "mtp.hidden_norm.weight"
                elif mtp_name == "eh_proj.weight":
                    target = "mtp.combine_proj.weight"
                    half = tensor.shape[1] // 2
                    tensor = torch.cat([tensor[:, half:], tensor[:, :half]], dim=1)
                elif mtp_name.startswith("block."):
                    target = _map_layer_name(mtp_name[len("block.") :], "mtp.transformer_block.layer.")
        if target is None or target not in local_state:
            ignored.append(reference_name)
            continue
        if local_state[target].shape != tensor.shape:
            raise ValueError(
                f"Weight mapping shape mismatch: {reference_name} {tuple(tensor.shape)} -> "
                f"{target} {tuple(local_state[target].shape)}"
            )
        local_state[target].copy_(tensor.to(device=local_state[target].device, dtype=local_state[target].dtype))
        if target in local_parameters:
            mapped_parameter_ids.add(id(local_parameters[target]))
        copied.append((reference_name, target))
    missing = [name for name, parameter in model.named_parameters() if id(parameter) not in mapped_parameter_ids]
    return WeightMappingReport(copied=copied, ignored_reference=ignored, missing_local_parameters=missing)
