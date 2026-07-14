from __future__ import annotations

import argparse
from contextlib import nullcontext
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .config import load_config
from .data import build_dataloaders
from .fingerprint_data import load_dataset_fingerprint
from .model import ReportAlignedDeepSeekV4
from .optim import HybridOptimizer, Optimizer, WarmupCosineScheduler, build_optimizer, write_parameter_group_csv
from .prepare_initialization import INITIALIZATION_FORMAT


LOG_FIELDS = [
    "step",
    "cumulative_tokens",
    "lr",
    "lr_next",
    "total_loss",
    "lm_loss",
    "mtp_loss",
    "aux_loss",
    "grad_norm",
    "grad_norm_after_clip",
    "muon_update_rms",
    "validation_loss",
    "reference_total_loss",
    "reference_lm_loss",
    "reference_mtp_loss",
    "reference_aux_loss",
    "reference_grad_norm",
    "reference_muon_update_rms",
    "absolute_difference",
    "relative_difference",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the fixed 397M TorchForge DSV4-inspired comparison model.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir")
    parser.add_argument("--resume")
    parser.add_argument("--initial-weights")
    parser.add_argument("--dataset-fingerprint")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--skip-final-checkpoint", action="store_true")
    parser.add_argument("--local-rank", type=int, default=int(os.environ.get("LOCAL_RANK", "0")))
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def distributed_context(local_rank: int) -> tuple[int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return rank, world_size, device


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def optimizer_update_rms(optimizer: Optimizer) -> float:
    return optimizer.muon_update_rms if isinstance(optimizer, HybridOptimizer) else 0.0


def grad_norm(parameters: Any) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().float().square().sum().item())
    return math.sqrt(total)


class LoaderCursor:
    """Deterministic sampler cursor that is persisted in checkpoints."""

    def __init__(self, loader: Any, *, epoch: int = 0, offset: int = 0) -> None:
        self.loader = loader
        self.sampler = loader.sampler
        self.epoch = int(epoch)
        self.offset = int(offset)
        self.iterator: Any = None
        self._reset_iterator(skip=self.offset)

    def _reset_iterator(self, *, skip: int = 0) -> None:
        self.sampler.set_epoch(self.epoch)
        self.iterator = iter(self.loader)
        for _ in range(skip):
            try:
                next(self.iterator)
            except StopIteration as exc:
                raise ValueError("Checkpoint data offset exceeds the sampler epoch length.") from exc

    def next(self, *, restart_sampler_epoch: int | None = None) -> dict[str, torch.Tensor]:
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.epoch = self.epoch + 1 if restart_sampler_epoch is None else int(restart_sampler_epoch)
            self.offset = 0
            self._reset_iterator()
            batch = next(self.iterator)
        self.offset += 1
        return batch

    def state_dict(self) -> dict[str, int]:
        return {"epoch": self.epoch, "offset": self.offset}


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: Optimizer,
    scheduler: WarmupCosineScheduler,
    step: int,
    cumulative_tokens: int,
    data_state: dict[str, int],
    config: dict[str, Any],
) -> None:
    state: dict[str, Any] = {
        "format": "torchforge_dsv4_muon_report_aligned_v1",
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": int(step),
        "cumulative_tokens": int(cumulative_tokens),
        "data_state": dict(data_state),
        "config": config,
        "rng_cpu": torch.get_rng_state(),
        "rng_python": random.getstate(),
    }
    if torch.cuda.is_available():
        state["rng_cuda"] = torch.cuda.get_rng_state_all()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: Optimizer,
    scheduler: WarmupCosineScheduler,
    device: torch.device,
) -> dict[str, Any]:
    state = torch.load(Path(path), map_location=device)
    if state.get("format") != "torchforge_dsv4_muon_report_aligned_v1":
        raise ValueError("Unsupported checkpoint format.")
    unwrap_model(model).load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    torch.set_rng_state(state["rng_cpu"].cpu())
    random.setstate(state["rng_python"])
    if torch.cuda.is_available() and "rng_cuda" in state:
        torch.cuda.set_rng_state_all(state["rng_cuda"])
    return {
        "step": int(state["step"]),
        "cumulative_tokens": int(state["cumulative_tokens"]),
        "data_state": dict(state["data_state"]),
    }


def load_initial_weights(
    path: str | Path,
    *,
    model: torch.nn.Module,
    config: dict[str, Any],
) -> dict[str, Any]:
    artifact = torch.load(Path(path), map_location="cpu")
    if artifact.get("format") != INITIALIZATION_FORMAT:
        raise ValueError("Unsupported comparison initialization format.")
    if int(artifact.get("seed", -1)) != int(config["seed"]):
        raise ValueError("Comparison initialization seed and training seed do not match.")
    expected = {key: config[key] for key in ("model", "v4_attention", "moe", "mtp")}
    if artifact.get("config_signature") != expected:
        raise ValueError("Comparison initialization and training model configuration do not match.")
    model.load_state_dict(artifact["model"])
    torch.set_rng_state(artifact["rng_cpu_after_comparison_model_init"].cpu())
    return {
        "format": artifact["format"],
        "seed": int(artifact["seed"]),
        "initialization_id": artifact.get("initialization_id"),
        "comparison_source": artifact.get("comparison_source"),
        "mapping": dict(artifact["mapping"]),
    }


def _data_metadata(config: dict[str, Any], *, dataset_fingerprint: str | None) -> dict[str, Any]:
    data = config["data"]
    data_dir = Path(data["data_dir"]).resolve()
    result: dict[str, Any] = {"data_dir": str(data_dir), "dtype": data["dtype"]}
    for key in ("train_file", "valid_file", "manifest_file"):
        path = data_dir / str(data[key])
        result[key] = str(path)
        result[f"{key}_size"] = path.stat().st_size if path.exists() else None
        if key == "manifest_file" and path.exists():
            result["manifest_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    if dataset_fingerprint:
        fingerprint = load_dataset_fingerprint(
            data_dir,
            manifest_file=str(data.get("manifest_file", "manifest.json")),
            fingerprint_path=dataset_fingerprint,
        )
        if fingerprint is not None:
            result.update(fingerprint)
    return result


def write_run_metadata(
    path: str | Path,
    *,
    config: dict[str, Any],
    world_size: int,
    tokens_per_step: int,
    resume: str | None,
    initial_weights: str | None,
    initialization: dict[str, Any] | None,
    dataset_fingerprint: str | None,
) -> None:
    payload = {
        "format": "torchforge_dsv4_comparison_run_v1",
        "world_size": int(world_size),
        "tokens_per_step": int(tokens_per_step),
        "seed": int(config["seed"]),
        "train_loss_scope": "rank0_microbatch_mean",
        "sampler_restart_epoch": "global_optimizer_step",
        "validation_autocast_bf16": bool(config["train"].get("validation_bf16", False)),
        "resume": str(Path(resume).resolve()) if resume else None,
        "initial_weights": str(Path(initial_weights).resolve()) if initial_weights else None,
        "initialization": initialization,
        "data": _data_metadata(config, dataset_fingerprint=dataset_fingerprint),
        "config": config,
    }
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: Any, *, device: torch.device, max_batches: int, bf16: bool) -> float:
    model.eval()
    total = torch.zeros(2, device=device, dtype=torch.float64)
    enabled = bf16 and (device.type == "cpu" or torch.cuda.is_bf16_supported())
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled):
            output = model(input_ids=input_ids, labels=labels)
        total[0] += output["loss"].detach().double()
        total[1] += 1.0
    if dist.is_initialized():
        dist.all_reduce(total)
    model.train()
    return float((total[0] / total[1].clamp_min(1.0)).item())


def append_logs(jsonl_path: Path, csv_path: Path, row: dict[str, Any]) -> None:
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
    exists = csv_path.exists() and csv_path.stat().st_size > 0
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> int:
    args = parse_args()
    if args.resume and args.initial_weights:
        raise ValueError("--resume and --initial-weights are mutually exclusive.")
    config = load_config(args.config)
    if args.data_dir:
        config["data"]["data_dir"] = args.data_dir
    if args.max_steps is not None:
        config["train"]["max_steps"] = args.max_steps
        config["train"].pop("target_tokens", None)
    if args.output_dir:
        config["train"]["output_dir"] = args.output_dir
    seed_everything(int(config["seed"]))
    rank, world_size, device = distributed_context(args.local_rank)
    train_config = config["train"]
    tokens_per_step = (
        int(train_config["micro_batch_size"])
        * int(train_config["gradient_accumulation_steps"])
        * int(train_config["seq_len"])
        * world_size
    )
    if "target_tokens" in train_config:
        train_config["max_steps"] = math.ceil(int(train_config["target_tokens"]) / tokens_per_step)
    output_dir = Path(train_config["output_dir"])
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, valid_loader = build_dataloaders(config, rank=rank, world_size=world_size)
    model = ReportAlignedDeepSeekV4(config)
    initialization = None
    if args.initial_weights:
        initialization = load_initial_weights(args.initial_weights, model=model, config=config)
    model = model.to(device)
    if world_size > 1:
        model = DDP(
            model,
            device_ids=[args.local_rank] if device.type == "cuda" else None,
            find_unused_parameters=bool(train_config.get("ddp_find_unused_parameters", True)),
        )
    optimizer = build_optimizer(model, train_config)
    scheduler = WarmupCosineScheduler(
        optimizer,
        base_lr=float(train_config["learning_rate"]),
        min_lr=float(train_config["min_lr"]),
        warmup_steps=int(train_config["warmup_steps"]),
        total_steps=int(train_config["max_steps"]),
    )
    if rank == 0:
        write_parameter_group_csv(
            unwrap_model(model),
            output_dir / "optimizer_parameter_groups.csv",
            float(train_config["weight_decay"]),
        )
        write_run_metadata(
            output_dir / "run_metadata.json",
            config=config,
            world_size=world_size,
            tokens_per_step=tokens_per_step,
            resume=args.resume,
            initial_weights=args.initial_weights,
            initialization=initialization,
            dataset_fingerprint=args.dataset_fingerprint,
        )

    start_step = 0
    cumulative_tokens = 0
    data_state = {"epoch": 0, "offset": 0}
    if args.resume:
        resume = load_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )
        start_step = resume["step"]
        cumulative_tokens = resume["cumulative_tokens"]
        data_state = resume["data_state"]
    cursor = LoaderCursor(train_loader, **data_state)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    jsonl_path = output_dir / "loss_log.jsonl"
    csv_path = output_dir / "loss_log.csv"
    grad_accum = int(train_config["gradient_accumulation_steps"])
    autocast_enabled = bool(train_config["bf16"]) and (device.type == "cpu" or torch.cuda.is_bf16_supported())

    for step_index in range(start_step, int(train_config["max_steps"])):
        loss_sums = {"loss": 0.0, "lm_loss": 0.0, "mtp_loss": 0.0, "aux_loss": 0.0}
        for micro_step in range(grad_accum):
            batch = cursor.next(restart_sampler_epoch=step_index)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            last_micro_step = micro_step == grad_accum - 1
            sync_context = (
                model.no_sync()
                if isinstance(model, DDP) and grad_accum > 1 and not last_micro_step
                else nullcontext()
            )
            with sync_context:
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=autocast_enabled,
                ):
                    output = model(input_ids=input_ids, labels=labels)
                    scaled_loss = output["loss"] / grad_accum
                scaled_loss.backward()
            for key in loss_sums:
                loss_sums[key] += float(output[key].detach().float().item())

        lr_used = float(optimizer.param_groups[0]["lr"])
        grad_norm_before = float(
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=float(train_config.get("gradient_clipping", 1.0)),
            ).item()
        )
        grad_norm_after = grad_norm(model.parameters())
        optimizer.step()
        muon_rms = optimizer_update_rms(optimizer)
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        lr_next = float(optimizer.param_groups[0]["lr"])
        cumulative_tokens += tokens_per_step
        completed_step = step_index + 1
        validation_loss: float | None = None
        if completed_step % int(train_config["valid_steps"]) == 0 or completed_step == int(train_config["max_steps"]):
            validation_loss = evaluate(
                model,
                valid_loader,
                device=device,
                max_batches=int(train_config["valid_max_batches"]),
                bf16=bool(train_config.get("validation_bf16", False)),
            )
        row = {
            "step": completed_step,
            "cumulative_tokens": cumulative_tokens,
            "lr": lr_used,
            "lr_next": lr_next,
            "total_loss": loss_sums["loss"] / grad_accum,
            "lm_loss": loss_sums["lm_loss"] / grad_accum,
            "mtp_loss": loss_sums["mtp_loss"] / grad_accum,
            "aux_loss": loss_sums["aux_loss"] / grad_accum,
            "grad_norm": grad_norm_before,
            "grad_norm_after_clip": grad_norm_after,
            "muon_update_rms": muon_rms,
            "validation_loss": validation_loss,
            "reference_total_loss": None,
            "reference_lm_loss": None,
            "reference_mtp_loss": None,
            "reference_aux_loss": None,
            "reference_grad_norm": None,
            "reference_muon_update_rms": None,
            "absolute_difference": None,
            "relative_difference": None,
        }
        if rank == 0:
            append_logs(jsonl_path, csv_path, row)
            if completed_step % int(train_config["log_steps"]) == 0:
                print(json.dumps(row, sort_keys=True))
            if completed_step % int(train_config["save_steps"]) == 0:
                save_checkpoint(
                    output_dir / f"step_{completed_step:06d}.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=completed_step,
                    cumulative_tokens=cumulative_tokens,
                    data_state=cursor.state_dict(),
                    config=config,
                )

    if rank == 0 and not args.skip_final_checkpoint:
        save_checkpoint(
            output_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=int(train_config["max_steps"]),
            cumulative_tokens=cumulative_tokens,
            data_state=cursor.state_dict(),
            config=config,
        )
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
