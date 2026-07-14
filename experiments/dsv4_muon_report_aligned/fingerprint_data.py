from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


FINGERPRINT_FORMAT = "torchforge_memmap_dataset_fingerprint_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute full SHA-256 fingerprints for memmap token files.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--manifest-file", default="manifest.json")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def sha256_file(path: str | Path, *, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_dataset(data_dir: str | Path, *, manifest_file: str = "manifest.json") -> dict[str, Any]:
    data_dir = Path(data_dir).resolve()
    manifest_path = data_dir / manifest_file
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dtype") != "uint32":
        raise ValueError(f"Expected uint32 manifest dtype, got {manifest.get('dtype')!r}.")

    result: dict[str, Any] = {
        "format": FINGERPRINT_FORMAT,
        "manifest_file": manifest_file,
        "manifest_sha256": sha256_file(manifest_path),
        "dtype": "uint32",
        "vocab_size": int(manifest["vocab_size"]),
    }
    for split in ("train", "valid"):
        file_name = str(manifest[f"{split}_file"])
        path = data_dir / file_name
        file_size = path.stat().st_size
        tokens_written = int(manifest[f"{split}_tokens_written"])
        if tokens_written * 4 > file_size:
            raise ValueError(f"{split} token count exceeds file size for {path}.")
        result[f"{split}_file"] = file_name
        result[f"{split}_file_size"] = file_size
        result[f"{split}_file_mtime_ns"] = path.stat().st_mtime_ns
        result[f"{split}_tokens_written"] = tokens_written
        result[f"{split}_sha256"] = sha256_file(path)
    identity = {
        key: value
        for key, value in result.items()
        if key not in {"format", "train_file_mtime_ns", "valid_file_mtime_ns"}
    }
    result["dataset_id"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return result


def load_dataset_fingerprint(
    data_dir: str | Path,
    *,
    manifest_file: str = "manifest.json",
    fingerprint_path: str | Path,
) -> dict[str, Any] | None:
    data_dir = Path(data_dir).resolve()
    fingerprint_path = Path(fingerprint_path).resolve()
    if not fingerprint_path.exists():
        return None
    payload = json.loads(fingerprint_path.read_text(encoding="utf-8"))
    if payload.get("format") != FINGERPRINT_FORMAT:
        raise ValueError(f"Unsupported dataset fingerprint format in {fingerprint_path}.")
    manifest_path = data_dir / manifest_file
    if payload.get("manifest_sha256") != sha256_file(manifest_path):
        raise ValueError(f"Dataset fingerprint manifest hash is stale: {fingerprint_path}.")
    identity = {
        key: value
        for key, value in payload.items()
        if key not in {"format", "dataset_id", "train_file_mtime_ns", "valid_file_mtime_ns"}
    }
    expected_dataset_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if payload.get("dataset_id") != expected_dataset_id:
        raise ValueError(f"Dataset fingerprint identity is invalid: {fingerprint_path}.")
    for split in ("train", "valid"):
        file_name = str(payload[f"{split}_file"])
        path = data_dir / file_name
        if int(payload[f"{split}_file_size"]) != path.stat().st_size:
            raise ValueError(f"Dataset fingerprint file size is stale for {path}.")
        if int(payload[f"{split}_file_mtime_ns"]) != path.stat().st_mtime_ns:
            raise ValueError(f"Dataset fingerprint modification time is stale for {path}.")
    return {
        "dataset_fingerprint_file": str(fingerprint_path),
        "dataset_fingerprint_sha256": sha256_file(fingerprint_path),
        "dataset_id": payload["dataset_id"],
        "train_sha256": payload["train_sha256"],
        "valid_sha256": payload["valid_sha256"],
    }


def main() -> int:
    args = parse_args()
    result = fingerprint_dataset(args.data_dir, manifest_file=args.manifest_file)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), **result}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
