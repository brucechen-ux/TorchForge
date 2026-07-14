# DeepSeek-V4-like Muon report-aligned experiment

This experiment assembles the supplied approximately 397M comparison shape from
`torchforge.common` components. It targets the behavior disclosed in the audit
package; it is not an official or complete DeepSeek-V4 implementation.
The configured module shapes contain 397,359,035 unique trainable parameters.

## Reused components

| Required behavior | TorchForge component | Experiment-specific work |
| --- | --- | --- |
| Token embedding, RMSNorm, LM head | `common.embedding`, `common.nn`, `common.lm_head` | Reference weight-name mapping |
| V4 query/KV normalization, partial RoPE, sinks, grouped output | `common.attention.MLA` | Opt-in SDPA backend and layer-specific RoPE selection |
| CSA/HCA compression and index scoring | `CSACompressor`, `HCACompressor` | Thin call-signature adapters |
| Hash and learned top-1 routing | `HashRouter` and common routing conventions | Reference top-1 normalization and learned hash scores |
| Shared SwiGLU expert | `SharedExpertMLP` | None |
| MTP fusion topology | `MultiTokenPredictionModule` | Ordinary-residual MTP block and reference projection mapping |
| Muon and AdamW | `common.optim.Muon`, `common.optim.AdamW` | One scheduler/checkpoint facade |

The existing `experiments/dsv4_assembly` uses mHC residuals and unpacked experts,
so it is intentionally not used as the report-aligned model body.

## Checks

Run the focused suite on a machine with a working PyTorch environment:

```bash
python -m pytest tests/test_optim_public_api.py tests/test_dsv4_muon_report_aligned.py -q
```

Run deterministic FP32/BF16 forward, gradient, single-step, and short multi-step
comparison against the read-only reference package:

```bash
python -m experiments.dsv4_muon_report_aligned.parity \
  --reference-root D:/infra-project/deepseek_v4_muon_report_aligned_package_20260713 \
  --device cuda --steps 3
```

This overwrites `loss_alignment_report.csv`, updates the measured section of
`loss_alignment_report.md`, and writes `parity_summary.json`, including actual
max absolute/relative errors and the first module exceeding the fixed FP32/BF16
thresholds.

## Smoke and full commands

Two-rank native-DDP smoke with real memmap data:

```bash
torchrun --standalone --nproc_per_node=2 \
  -m experiments.dsv4_muon_report_aligned.train \
  --config experiments/dsv4_muon_report_aligned/configs/B_muon_hybrid.yaml \
  --data-dir /path/to/openbmb_UltraFineWeb_5b_random_tokens \
  --max-steps 2 \
  --output-dir experiments/dsv4_muon_report_aligned/outputs/ddp2_smoke
```

The full aligned runs use the same command with `--nproc_per_node=8` and without
`--max-steps`. Run A, then B, then C. Do not add DeepSpeed or ZeRO to these Muon
runs without separately proving that each rank sees complete logical matrices.

```bash
torchrun --standalone --nproc_per_node=8 -m experiments.dsv4_muon_report_aligned.train --config experiments/dsv4_muon_report_aligned/configs/A_adamw.yaml --data-dir /path/to/data
torchrun --standalone --nproc_per_node=8 -m experiments.dsv4_muon_report_aligned.train --config experiments/dsv4_muon_report_aligned/configs/B_muon_hybrid.yaml --data-dir /path/to/data
torchrun --standalone --nproc_per_node=8 -m experiments.dsv4_muon_report_aligned.train --config experiments/dsv4_muon_report_aligned/configs/C_muon_standard.yaml --data-dir /path/to/data
```

At world size 8, each optimizer step consumes 262,144 tokens. `target_tokens=5e9`
therefore resolves to 19,074 steps (5,000,134,656 tokens).

Only after the parity and first A/B/C mechanism checks pass should short LR sweep
configs be derived at 0.5x, 0.75x, 1x, 1.5x, and 2x.
