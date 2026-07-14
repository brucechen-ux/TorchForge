from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

from experiments.dsv4_muon_report_aligned.fingerprint_data import (
    fingerprint_dataset,
    load_dataset_fingerprint,
)


def test_full_dataset_fingerprint_round_trip(tmp_path: Path) -> None:
    train = struct.pack("<5I", 1, 2, 3, 4, 5)
    valid = struct.pack("<3I", 6, 7, 8)
    (tmp_path / "train.bin").write_bytes(train)
    (tmp_path / "valid.bin").write_bytes(valid)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "dtype": "uint32",
                "vocab_size": 64,
                "train_file": "train.bin",
                "valid_file": "valid.bin",
                "train_tokens_written": 5,
                "valid_tokens_written": 3,
            }
        ),
        encoding="utf-8",
    )

    payload = fingerprint_dataset(tmp_path)
    sidecar = tmp_path / "fingerprint-output.json"
    sidecar.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    loaded = load_dataset_fingerprint(tmp_path, fingerprint_path=sidecar)

    assert payload["train_sha256"] == hashlib.sha256(train).hexdigest()
    assert payload["valid_sha256"] == hashlib.sha256(valid).hexdigest()
    assert loaded is not None
    assert loaded["dataset_id"] == payload["dataset_id"]
    assert loaded["train_sha256"] == payload["train_sha256"]
    assert loaded["valid_sha256"] == payload["valid_sha256"]
