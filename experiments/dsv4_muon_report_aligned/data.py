from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


class MemmapTokenDataset(Dataset[dict[str, torch.Tensor]]):
    """Read contiguous shifted token windows from a read-only uint32 token file."""

    def __init__(self, config: dict[str, Any], split: str, seq_len: int) -> None:
        if split not in {"train", "valid"}:
            raise ValueError("split must be 'train' or 'valid'.")
        self.seq_len = int(seq_len)
        data_dir = Path(config["data_dir"])
        manifest_path = data_dir / str(config.get("manifest_file", "manifest.json"))
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if str(manifest.get("dtype", config.get("dtype"))) != "uint32":
            raise ValueError("The fixed 397M comparison memmap format must be uint32.")
        file_key = f"{split}_file"
        count_key = f"{split}_tokens_written"
        file_name = str(manifest[file_key])
        configured_file = config.get(file_key)
        if configured_file is not None and str(configured_file) != file_name:
            raise ValueError(
                f"data.{file_key}={configured_file!r} does not match manifest {file_key}={file_name!r}."
            )
        path = data_dir / file_name
        file_elements = path.stat().st_size // 4
        self.token_count = int(manifest.get(count_key, file_elements))
        if self.token_count > file_elements:
            raise ValueError(f"Manifest token count exceeds file size for {path}.")
        if self.token_count < self.seq_len + 1:
            raise ValueError(f"{path} does not contain one complete sequence.")
        manifest_vocab = int(manifest["vocab_size"])
        if manifest_vocab != int(config["vocab_size"]):
            raise ValueError(f"Manifest vocab_size={manifest_vocab} != config vocab_size={config['vocab_size']}.")
        # Token ids are below 2^31 for this experiment, so a signed int32 view is
        # byte-identical to uint32 and avoids requiring NumPy for memory mapping.
        self.tokens = torch.from_file(str(path), shared=False, size=self.token_count, dtype=torch.int32)
        self.num_sequences = (self.token_count - 1) // self.seq_len

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0 or index >= self.num_sequences:
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
    data_config, train_config = config["data"], config["train"]
    train_dataset = MemmapTokenDataset(data_config, "train", int(train_config["seq_len"]))
    valid_dataset = MemmapTokenDataset(data_config, "valid", int(train_config["seq_len"]))
    seed = int(config["seed"])
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=seed,
        drop_last=False,
    )
    valid_sampler = DistributedSampler(
        valid_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        seed=seed,
        drop_last=False,
    )
    worker_count = int(data_config.get("num_workers", 0))
    loader_options: dict[str, Any] = {
        "num_workers": worker_count,
        "pin_memory": bool(data_config.get("pin_memory", True)),
    }
    if worker_count:
        loader_options.update(
            persistent_workers=bool(data_config.get("persistent_workers", True)),
            prefetch_factor=int(data_config.get("prefetch_factor", 2)),
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_config["micro_batch_size"]),
        sampler=train_sampler,
        drop_last=True,
        **loader_options,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=int(train_config["micro_batch_size"]),
        sampler=valid_sampler,
        drop_last=False,
        **loader_options,
    )
    return train_loader, valid_loader
