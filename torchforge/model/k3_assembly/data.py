from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


def _seed_worker(worker_id: int) -> None:
    del worker_id
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    torch.manual_seed(seed)


class MemmapTokenDataset(Dataset[dict[str, torch.Tensor]]):
    """Contiguous next-token windows from TorchForge's uint32 memmap format."""

    def __init__(self, config: dict[str, Any], split: str, seq_len: int) -> None:
        if split not in {"train", "valid"}:
            raise ValueError("split must be 'train' or 'valid'.")
        self.seq_len = int(seq_len)
        data_dir = Path(config["data_dir"])
        with (data_dir / str(config.get("manifest_file", "manifest.json"))).open(
            "r", encoding="utf-8"
        ) as handle:
            manifest = json.load(handle)
        if str(manifest.get("dtype", config.get("dtype"))) != "uint32":
            raise ValueError("Small K3 requires uint32 memmap tokens.")
        file_key = f"{split}_file"
        file_name = str(manifest[file_key])
        if config.get(file_key) is not None and str(config[file_key]) != file_name:
            raise ValueError(f"Configured {file_key} does not match the data manifest.")
        path = data_dir / file_name
        file_elements = path.stat().st_size // 4
        count = manifest.get(f"{split}_tokens_written", manifest.get(f"{split}_tokens", file_elements))
        self.token_count = int(count)
        if self.token_count > file_elements:
            raise ValueError(f"Manifest token count exceeds file size for {path}.")
        if self.token_count < self.seq_len + 1:
            raise ValueError(f"{path} does not contain one complete sequence.")
        if int(manifest["vocab_size"]) != int(config["vocab_size"]):
            raise ValueError("Manifest and config vocabulary sizes differ.")
        self.tokens = torch.from_file(
            str(path), shared=False, size=self.token_count, dtype=torch.int32
        )
        self.num_sequences = (self.token_count - 1) // self.seq_len

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if not 0 <= index < self.num_sequences:
            raise IndexError(index)
        start = index * self.seq_len
        window = self.tokens[start : start + self.seq_len + 1].long()
        return {"input_ids": window[:-1].clone(), "labels": window[1:].clone()}


def build_dataloaders(
    config: dict[str, Any],
    *,
    rank: int,
    world_size: int,
) -> tuple[DataLoader[dict[str, torch.Tensor]], DataLoader[dict[str, torch.Tensor]]]:
    data_cfg, train_cfg = config["data"], config["train"]
    train_seq_len = int(train_cfg["seq_len"])
    validation_seq_len = int(train_cfg.get("validation_seq_len", train_seq_len))
    train_dataset = MemmapTokenDataset(data_cfg, "train", train_seq_len)
    valid_dataset = MemmapTokenDataset(data_cfg, "valid", validation_seq_len)
    seed = int(config["seed"])
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=seed,
        drop_last=True,
    )
    valid_sampler = DistributedSampler(
        valid_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )
    workers = int(data_cfg.get("num_workers", 0))
    options: dict[str, Any] = {
        "num_workers": workers,
        "pin_memory": bool(data_cfg.get("pin_memory", True)),
        "worker_init_fn": _seed_worker,
    }
    if workers:
        options.update(
            persistent_workers=bool(data_cfg.get("persistent_workers", True)),
            prefetch_factor=int(data_cfg.get("prefetch_factor", 2)),
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_cfg["micro_batch_size"]),
        sampler=train_sampler,
        drop_last=True,
        generator=torch.Generator().manual_seed(seed + 2 * rank),
        **options,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=int(train_cfg["micro_batch_size"]),
        sampler=valid_sampler,
        drop_last=False,
        generator=torch.Generator().manual_seed(seed + 2 * rank + 1),
        **options,
    )
    return train_loader, valid_loader


__all__ = ["MemmapTokenDataset", "build_dataloaders"]
