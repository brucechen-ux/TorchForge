from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from .config import load_config
from .model import ReportAlignedDeepSeekV4, load_reference_weights
from .parity import import_reference, reference_config


INITIALIZATION_FORMAT = "torchforge_dsv4_comparison_initialization_v1"
COMPARISON_SOURCE_FILES = (
    "modeling_v3.py",
    "v4_attention.py",
    "moe.py",
    "mtp.py",
    "muon.py",
    "data.py",
    "train.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare mapped TorchForge weights for a shared cross-project initialization."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--comparison-root", "--reference-root", dest="comparison_root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def prepare_initialization(
    config: dict,
    comparison_model_type: type[torch.nn.Module],
    *,
    device: torch.device,
) -> dict:
    seed = int(config["seed"])
    torch.manual_seed(seed)
    comparison_model = comparison_model_type(reference_config(config)).to(device)
    comparison_rng_cpu = torch.get_rng_state().cpu()

    torch.manual_seed(seed)
    torchforge_model = ReportAlignedDeepSeekV4(config).to(device)
    mapping = load_reference_weights(torchforge_model, comparison_model.state_dict())
    if mapping.missing_local_parameters:
        raise ValueError(
            "Cannot prepare shared initialization; unmapped TorchForge parameters: "
            + ", ".join(mapping.missing_local_parameters)
        )

    return {
        "format": INITIALIZATION_FORMAT,
        "seed": seed,
        "model": {name: tensor.detach().cpu() for name, tensor in torchforge_model.state_dict().items()},
        "rng_cpu_after_comparison_model_init": comparison_rng_cpu,
        "mapping": {
            "copied": mapping.copied,
            "ignored_comparison": mapping.ignored_reference,
            "missing_torchforge_parameters": mapping.missing_local_parameters,
        },
        "config_signature": {
            key: config[key] for key in ("model", "v4_attention", "moe", "mtp")
        },
    }


def comparison_source_fingerprint(root: str | Path) -> dict[str, object]:
    root = Path(root).resolve()
    files = {}
    for name in COMPARISON_SOURCE_FILES:
        path = root / "src" / name
        if not path.exists():
            raise FileNotFoundError(f"Comparison source file not found: {path}")
        files[f"src/{name}"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"root": str(root), "sha256": files}


def initialization_id(artifact: dict) -> str:
    identity = {
        "format": artifact["format"],
        "seed": artifact["seed"],
        "config_signature": artifact["config_signature"],
        "mapping": artifact["mapping"],
        "comparison_source_sha256": artifact["comparison_source"]["sha256"],
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    comparison_model_type, _ = import_reference(args.comparison_root)
    artifact = prepare_initialization(config, comparison_model_type, device=torch.device(args.device))
    artifact["comparison_source"] = comparison_source_fingerprint(args.comparison_root)
    artifact["initialization_id"] = initialization_id(artifact)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output)
    provenance_path = output.with_name(output.name + ".provenance.json")
    provenance_path.write_text(
        json.dumps(
            {key: value for key, value in artifact.items() if key != "model" and key != "rng_cpu_after_comparison_model_init"},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "provenance": str(provenance_path.resolve()),
                "format": artifact["format"],
                "seed": artifact["seed"],
                "initialization_id": artifact["initialization_id"],
                "copied": len(artifact["mapping"]["copied"]),
                "ignored_comparison": artifact["mapping"]["ignored_comparison"],
                "missing_torchforge_parameters": artifact["mapping"]["missing_torchforge_parameters"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
