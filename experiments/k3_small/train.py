from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import random
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .config import load_config, validate_config
from .data import build_dataloaders
from .model import SmallK3Model, architecture_summary
from .optim import K3Optimizer, WarmupCosineScheduler, build_optimizer


CHECKPOINT_FORMAT = "torchforge_k3_small_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the TorchForge small Kimi-K3 experiment.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--resume")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--validation-seq-len", type=int)
    parser.add_argument("--skip-final-checkpoint", action="store_true")
    parser.add_argument("--local-rank", type=int, default=int(os.environ.get("LOCAL_RANK", "0")))
    return parser.parse_args()


def distributed_context(local_rank: int) -> tuple[int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return rank, world_size, torch.device("cuda", local_rank)
    return rank, world_size, torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def unwrap_model(model: torch.nn.Module) -> SmallK3Model:
    unwrapped = model.module if isinstance(model, DDP) else model
    if not isinstance(unwrapped, SmallK3Model):
        raise TypeError("Expected SmallK3Model or DDP[SmallK3Model].")
    return unwrapped


class LoaderCursor:
    def __init__(self, loader: Any, *, epoch: int = 0, offset: int = 0) -> None:
        self.loader = loader
        self.sampler = loader.sampler
        self.epoch = int(epoch)
        self.offset = int(offset)
        self.iterator: Any = None
        self._reset(skip=self.offset)

    def _reset(self, *, skip: int = 0) -> None:
        self.sampler.set_epoch(self.epoch)
        self.iterator = iter(self.loader)
        for _ in range(skip):
            try:
                next(self.iterator)
            except StopIteration as exc:
                raise ValueError("Checkpoint data offset exceeds the sampler epoch.") from exc

    def next(self) -> dict[str, torch.Tensor]:
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.epoch += 1
            self.offset = 0
            self._reset()
            batch = next(self.iterator)
        self.offset += 1
        return batch

    def state_dict(self) -> dict[str, int]:
        return {"epoch": self.epoch, "offset": self.offset}


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: K3Optimizer,
    scheduler: WarmupCosineScheduler,
    step: int,
    cumulative_tokens: int,
    data_state: dict[str, int],
    config: dict[str, Any],
) -> None:
    state: dict[str, Any] = {
        "format": CHECKPOINT_FORMAT,
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
    optimizer: K3Optimizer,
    scheduler: WarmupCosineScheduler,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    state = torch.load(Path(path), map_location=device, weights_only=False)
    if state.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("Unsupported small-K3 checkpoint format.")
    if state.get("config") != config:
        raise ValueError("Checkpoint config does not match the active config.")
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


def _instantiate_model(config: dict[str, Any], device: torch.device) -> SmallK3Model:
    dtype = torch.bfloat16 if bool(config["train"]["bf16"]) and device.type == "cuda" else torch.float32
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device(device):
            model = SmallK3Model(config)
    finally:
        torch.set_default_dtype(previous_dtype)
    return model


def _grad_norm(parameters: Any) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().float().square().sum().item())
    return math.sqrt(total)


@torch.no_grad()
def validate(model: torch.nn.Module, loader: Any, device: torch.device, max_batches: int = 8) -> float:
    model.eval()
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    count = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        loss_sum += model(input_ids, labels=labels)["loss"].double()
        count += 1
        if count >= max_batches:
            break
    totals = torch.tensor([loss_sum.item(), count], device=device, dtype=torch.float64)
    if dist.is_initialized():
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    model.train()
    return float(totals[0].item() / max(totals[1].item(), 1.0))


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.data_dir:
        config["data"]["data_dir"] = args.data_dir
    if args.output_dir:
        config["train"]["output_dir"] = args.output_dir
    if args.max_steps is not None:
        config["train"]["max_steps"] = args.max_steps
    if args.seq_len is not None:
        config["train"]["seq_len"] = args.seq_len
    if args.validation_seq_len is not None:
        config["train"]["validation_seq_len"] = args.validation_seq_len
    validate_config(config)
    rank, world_size, device = distributed_context(args.local_rank)
    seed_everything(int(config["seed"]) + rank)
    model = _instantiate_model(config, device)
    optimizer = build_optimizer(model, config["train"])
    scheduler = WarmupCosineScheduler(
        optimizer,
        base_lr=float(config["train"]["learning_rate"]),
        min_lr=float(config["train"]["min_lr"]),
        warmup_steps=int(config["train"]["warmup_steps"]),
        total_steps=int(config["train"]["max_steps"]),
    )
    if world_size > 1:
        model = DDP(
            model,
            device_ids=[args.local_rank] if device.type == "cuda" else None,
            find_unused_parameters=False,
            broadcast_buffers=False,
        )
    train_loader, valid_loader = build_dataloaders(config, rank=rank, world_size=world_size)
    start_step = 0
    cumulative_tokens = 0
    data_state = {"epoch": 0, "offset": 0}
    if args.resume:
        resume = load_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            device=device,
        )
        start_step = resume["step"]
        cumulative_tokens = resume["cumulative_tokens"]
        data_state = resume["data_state"]
    cursor = LoaderCursor(train_loader, **data_state)
    output_dir = Path(config["train"]["output_dir"])
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(json.dumps(architecture_summary(unwrap_model(model)), sort_keys=True))
    model.train()
    optimizer.zero_grad()
    grad_accum = int(config["train"]["gradient_accumulation_steps"])
    max_steps = int(config["train"]["max_steps"])
    for step in range(start_step, max_steps):
        loss_sum = 0.0
        lm_loss_sum = 0.0
        mtp_loss_sum = 0.0
        for micro_step in range(grad_accum):
            batch = cursor.next()
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            sync_context = (
                model.no_sync()
                if isinstance(model, DDP) and micro_step < grad_accum - 1
                else nullcontext()
            )
            with sync_context:
                outputs = model(input_ids, labels=labels)
                (outputs["loss"] / grad_accum).backward()
            loss_sum += float(outputs["loss"].detach())
            lm_loss_sum += float(outputs["lm_loss"].detach())
            mtp_loss_sum += float(outputs["mtp_loss"].detach())
        parameters = [parameter for parameter in model.parameters() if parameter.grad is not None]
        grad_before = _grad_norm(parameters)
        finite_step = torch.tensor(
            int(
                math.isfinite(loss_sum)
                and math.isfinite(lm_loss_sum)
                and math.isfinite(mtp_loss_sum)
                and math.isfinite(grad_before)
            ),
            device=device,
            dtype=torch.int32,
        )
        if dist.is_initialized():
            dist.all_reduce(finite_step, op=dist.ReduceOp.MIN)
        if not bool(finite_step.item()):
            optimizer.zero_grad()
            raise FloatingPointError(f"Non-finite loss or gradient detected at step {step + 1}.")
        torch.nn.utils.clip_grad_norm_(parameters, float(config["train"]["gradient_clipping"]))
        optimizer.step()
        unwrap_model(model).update_router_biases(distributed=True)
        scheduler.step()
        optimizer.zero_grad()
        cumulative_tokens += (
            world_size
            * int(config["train"]["micro_batch_size"])
            * int(config["train"]["seq_len"])
            * grad_accum
        )
        completed_step = step + 1
        if rank == 0 and completed_step % int(config["train"]["log_steps"]) == 0:
            peak_memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
            print(
                json.dumps(
                    {
                        "step": completed_step,
                        "tokens": cumulative_tokens,
                        "loss": loss_sum / grad_accum,
                        "lm_loss": lm_loss_sum / grad_accum,
                        "mtp_loss": mtp_loss_sum / grad_accum,
                        "grad_norm": grad_before,
                        "lr": optimizer.param_groups[0]["lr"],
                        "peak_memory_bytes": peak_memory,
                    },
                    sort_keys=True,
                )
            )
        if completed_step % int(config["train"]["valid_steps"]) == 0:
            validation_loss = validate(model, valid_loader, device)
            if rank == 0:
                print(json.dumps({"step": completed_step, "validation_loss": validation_loss}))
        if completed_step % int(config["train"]["save_steps"]) == 0:
            if dist.is_initialized():
                dist.barrier()
            if rank == 0:
                save_checkpoint(
                    output_dir / f"step_{completed_step:08d}.pt",
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
            output_dir / "final.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=max_steps,
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


__all__ = [
    "CHECKPOINT_FORMAT",
    "LoaderCursor",
    "load_checkpoint",
    "save_checkpoint",
]
