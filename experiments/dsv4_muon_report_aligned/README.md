# 397M DeepSeek-V4-inspired cross-project loss comparison

This experiment assembles an approximately 397M comparison model from
`torchforge.common` components. The DeepSeek-V4 technical report is the authority
for disclosed V4 mechanisms. The adjacent package is a peer implementation whose
numerical differences are measured; it is not an oracle. This is not an official
or complete DeepSeek-V4 implementation.
The configured module shapes contain 397,359,035 unique trainable parameters.

The directory, Python symbols containing `ReportAligned`, `--reference-root`, and
CSV fields prefixed by `reference_` are retained for compatibility. In this
experiment, `reference_` means "comparison project", not "correct value".

## Provenance of the 397M protocol

| Item | Source and status |
| --- | --- |
| CSA/HCA, sqrt-softplus routing scores, hash routing, MTP, SwiGLU clamping | Mechanisms disclosed by the technical report |
| Muon Hybrid 8+2, momentum 0.95, Nesterov, logical-matrix scaling to RMS 0.18 | Explicitly disclosed by the technical report |
| 16 layers, hidden size 704, 11 heads, 16 experts, top-1, ordinary residuals | 397M comparison protocol; not a published V4 model configuration |
| Fixed MTP weight 0.1, zero auxiliary-loss weight, router in AdamW | 397M comparison protocol; differs from the full report setup |
| AdamW baseline and Standard-10 Newton-Schulz | Experimental controls; not claimed as V4 training settings |

The full report uses mHC, six routed experts per token, sequence balance loss
`1e-4`, correction-bias update speed `0.001`, and an MTP weight of `0.3` before
switching to `0.1` near learning-rate decay. Those are documented differences,
not silently attributed to the 397M protocol.

## Reused components

| Required behavior | TorchForge component | Experiment-specific work |
| --- | --- | --- |
| Token embedding, RMSNorm, LM head | `common.embedding`, `common.nn`, `common.lm_head` | Cross-project weight-name mapping for controlled diagnostics |
| V4 query/KV normalization, partial RoPE, sinks, grouped output | `common.attention.MLA` | Opt-in SDPA backend and layer-specific RoPE selection |
| CSA/HCA compression and index scoring | `CSACompressor`, `HCACompressor` | Thin call-signature adapters |
| Hash and learned top-1 routing | `HashRouter` and common routing conventions | 397M protocol routing behavior |
| Shared SwiGLU expert | `SharedExpertMLP` | None |
| MTP fusion topology | `MultiTokenPredictionModule` | Ordinary-residual MTP block and cross-project projection mapping |
| Muon and AdamW | `common.optim.Muon`, `common.optim.AdamW` | One scheduler/checkpoint facade |

The existing `experiments/dsv4_assembly` uses mHC residuals and unpacked experts,
so it is intentionally not used as the fixed 397M comparison model body.

## Checks

Run the focused suite on a machine with a working PyTorch environment:

```bash
python -m pytest tests/test_optim_public_api.py tests/test_dsv4_muon_report_aligned.py tests/test_dsv4_loss_curve_comparison.py tests/test_dsv4_data_fingerprint.py tests/test_dsv4_loss_curve_plot.py -q
```

Run a controlled FP32/BF16 forward, gradient, single-step, and short multi-step
comparison against the read-only peer package. This maps the same initial weights
and uses the same tokens and training settings to locate differences; non-zero
errors are measurements, not test failures:

```bash
python -m experiments.dsv4_muon_report_aligned.parity \
  --reference-root D:/infra-project/deepseek_v4_muon_report_aligned_package_20260713 \
  --device cuda --steps 3
```

This overwrites `loss_alignment_report.csv`, updates the measured section of
`loss_alignment_report.md`, and writes `parity_summary.json`. Reporting thresholds
only locate the first visible difference and are not acceptance tolerances.

For independently produced training curves, compare the two native logs by exact
cumulative-token position:

```bash
python -m experiments.dsv4_muon_report_aligned.compare_curves \
  --torchforge-log /path/to/torchforge/output/loss_log.jsonl \
  --comparison-log /path/to/comparison/train_metrics_RUN.jsonl \
  --torchforge-meta /path/to/torchforge/output/run_metadata.json \
  --comparison-meta /path/to/comparison/train_meta_RUN.json \
  --comparison-data-dir /path/to/data \
  --comparison-dataset-fingerprint experiments/dsv4_muon_report_aligned/outputs/dataset_fingerprint.json \
  --output-dir experiments/dsv4_muon_report_aligned/outputs/comparisons/B \
  --comparison-lr-is-next-step \
  --require-identical-token-grid
```

The output CSV contains both values plus absolute and relative differences for
LR, total/LM/MTP/aux loss, gradient norm, Muon update RMS, and validation loss.
Missing source metrics remain blank. Prefer the comparison project's JSONL over
its CSV because its CSV `loss` column is LM loss and omits loss components.
`--comparison-lr-is-next-step` is required for the supplied comparison runner:
it logs LR after `scheduler.step()`, whereas TorchForge logs the LR actually used
for the completed update.

After A/B/C comparison CSVs are generated, render the paired loss curves and a
separate absolute-difference panel without installing plotting dependencies:

```bash
python -m experiments.dsv4_muon_report_aligned.plot_curves \
  --series A=experiments/dsv4_muon_report_aligned/outputs/comparisons/A/loss_curve_comparison.csv \
  --series B=experiments/dsv4_muon_report_aligned/outputs/comparisons/B/loss_curve_comparison.csv \
  --series C=experiments/dsv4_muon_report_aligned/outputs/comparisons/C/loss_curve_comparison.csv \
  --metric total_loss \
  --smooth-window 50 \
  --output experiments/dsv4_muon_report_aligned/outputs/comparisons/total_loss.svg
```

Solid lines are TorchForge, dashed lines are the peer project, and the lower
panel is `abs(TorchForge - peer)` at the same cumulative-token positions. Use
`--metric lm_loss` for the LM-only comparison.

For the complete optimizer-analysis figure, separate each A/B/C cross-project
loss and absolute-difference panel, then add project-specific A/B/C overlays and
signed optimizer differences (`A-B` means `loss(A) - loss(B)`):

```bash
python -m experiments.dsv4_muon_report_aligned.plot_curves \
  --series A=experiments/dsv4_muon_report_aligned/outputs/comparisons/A/loss_curve_comparison.csv \
  --series B=experiments/dsv4_muon_report_aligned/outputs/comparisons/B/loss_curve_comparison.csv \
  --series C=experiments/dsv4_muon_report_aligned/outputs/comparisons/C/loss_curve_comparison.csv \
  --metric total_loss \
  --smooth-window 5 \
  --optimizer-analysis \
  --output experiments/dsv4_muon_report_aligned/outputs/comparisons/total_loss_full_analysis.svg
```

Do not rely on equal seeds alone for native curves: module construction order
can produce different initial tensors. Prepare a mapped TorchForge initialization
from the peer implementation once, then reuse it for A/B/C:

First compute the full token-file fingerprint once. This reads but does not modify
the memmap files; it writes a sidecar required by strict run-metadata checks:

```bash
python -m experiments.dsv4_muon_report_aligned.fingerprint_data \
  --data-dir /path/to/data \
  --output experiments/dsv4_muon_report_aligned/outputs/dataset_fingerprint.json
```

```bash
python -m experiments.dsv4_muon_report_aligned.prepare_initialization \
  --config experiments/dsv4_muon_report_aligned/configs/B_muon_hybrid.yaml \
  --comparison-root /path/to/TorchForge-reference \
  --output experiments/dsv4_muon_report_aligned/outputs/shared_initial_weights.pt
```

Start the peer run from its normal seed-2026 initialization and pass the generated
artifact only to the TorchForge run:

```bash
torchrun --standalone --nproc_per_node=8 \
  -m experiments.dsv4_muon_report_aligned.train \
  --config experiments/dsv4_muon_report_aligned/configs/B_muon_hybrid.yaml \
  --data-dir /path/to/data \
  --dataset-fingerprint experiments/dsv4_muon_report_aligned/outputs/dataset_fingerprint.json \
  --initial-weights experiments/dsv4_muon_report_aligned/outputs/shared_initial_weights.pt
```

Each TorchForge output directory contains `run_metadata.json` with the resolved
config, world size, tokens per step, data paths, manifest hash, validation dtype,
and initialization provenance. Compare this with the peer run metadata before
interpreting a curve. The current native runners record rank-0 training loss;
validation is globally reduced. The 397M configs use FP32 validation so the
validation dtype matches the supplied peer runner.

The initialization metadata includes SHA-256 fingerprints of the peer model,
attention, MoE, MTP, Muon, data, and train source files. Preparation also writes
`shared_initial_weights.pt.provenance.json`; preserve it with both run records.
Loading the artifact rejects a seed or model-config mismatch. Native peer logs do not expose Muon update RMS; that
peer column is intentionally blank in formal curve CSVs. Use the controlled
single-step comparison for per-logical-matrix RMS evidence, and never substitute
the peer's semantically different `update_norm` metric.

The train sampler also uses the peer runner's global-step epoch value when the
loader wraps, so a data-boundary crossing does not silently change batch order.
The peer runner does not restore its sampler/RNG cursor on resume; therefore a
formal paired curve should run uninterrupted, or be treated as comparable only
after an explicit post-resume batch-order trace check.

## Smoke and full commands

Keep the peer checkout read-only by disabling bytecode writes and redirecting
all checkpoints, TensorBoard files, and logs into TorchForge. Example paths for
the Linux workspace used during testing:

```bash
export TF=/gemini/code/TorchForge
export PEER=/gemini/code/TorchForge-reference
export DATA=/path/to/openbmb_UltraFineWeb_5b_random_tokens
export RUN_ROOT=$TF/experiments/dsv4_muon_report_aligned/outputs
export DATA_FP=$RUN_ROOT/dataset_fingerprint.json
export SHARED_INIT=$RUN_ROOT/shared_initial_weights.pt
```

Run the peer B configuration for a two-rank, 20-step smoke. The max-step
override intentionally changes the scheduler horizon and is only a mechanism
smoke, not a representative training curve:

```bash
cd "$TF"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PEER" \
torchrun --standalone --nproc_per_node=2 -m src.train \
  --config "$PEER/configs/rerun_v4_muon_hybrid.yaml" \
  --data_dir "$DATA" \
  --override_max_steps 20 \
  --override_output_dir "$RUN_ROOT/smoke/peer_B/checkpoints" \
  --override_tensorboard_dir "$RUN_ROOT/smoke/peer_B/tensorboard" \
  --log_dir "$RUN_ROOT/smoke/peer_B/logs" \
  --skip_final_checkpoint
```

Run TorchForge B with the same world size, data, step count, and mapped initial
weights:

```bash
cd "$TF"
torchrun --standalone --nproc_per_node=2 \
  -m experiments.dsv4_muon_report_aligned.train \
  --config experiments/dsv4_muon_report_aligned/configs/B_muon_hybrid.yaml \
  --data-dir "$DATA" \
  --dataset-fingerprint "$DATA_FP" \
  --initial-weights "$SHARED_INIT" \
  --max-steps 20 \
  --output-dir "$RUN_ROOT/smoke/torchforge_B" \
  --skip-final-checkpoint
```

The peer log names include a timestamp. Resolve the single files from the clean
smoke directory, compare them, and render the SVG:

```bash
PEER_LOG=$(ls "$RUN_ROOT/smoke/peer_B/logs"/train_metrics_*.jsonl)
PEER_META=$(ls "$RUN_ROOT/smoke/peer_B/logs"/train_meta_*.json)

python -m experiments.dsv4_muon_report_aligned.compare_curves \
  --torchforge-log "$RUN_ROOT/smoke/torchforge_B/loss_log.jsonl" \
  --comparison-log "$PEER_LOG" \
  --torchforge-meta "$RUN_ROOT/smoke/torchforge_B/run_metadata.json" \
  --comparison-meta "$PEER_META" \
  --comparison-data-dir "$DATA" \
  --comparison-dataset-fingerprint "$DATA_FP" \
  --output-dir "$RUN_ROOT/smoke/comparison_B" \
  --comparison-lr-is-next-step \
  --require-identical-token-grid

python -m experiments.dsv4_muon_report_aligned.plot_curves \
  --series B="$RUN_ROOT/smoke/comparison_B/loss_curve_comparison.csv" \
  --metric total_loss \
  --smooth-window 5 \
  --output "$RUN_ROOT/smoke/comparison_B/total_loss.svg"
```

Formal peer/TorchForge config pairs are:

| Run | Peer config | TorchForge config |
| --- | --- | --- |
| A | `rerun_adamw.yaml` | `A_adamw.yaml` |
| B | `rerun_v4_muon_hybrid.yaml` | `B_muon_hybrid.yaml` |
| C | `rerun_muon_standard_ns.yaml` | `C_muon_standard.yaml` |

The paired 397M runs use the same command with `--nproc_per_node=8` and without
either max-step override. Run A, then B, then C, with separate output/log
directories. Do not add DeepSpeed or ZeRO to these Muon runs without separately
proving that each rank sees complete logical matrices.

```bash
torchrun --standalone --nproc_per_node=8 -m experiments.dsv4_muon_report_aligned.train --config experiments/dsv4_muon_report_aligned/configs/A_adamw.yaml --data-dir /path/to/data --dataset-fingerprint experiments/dsv4_muon_report_aligned/outputs/dataset_fingerprint.json --initial-weights experiments/dsv4_muon_report_aligned/outputs/shared_initial_weights.pt
torchrun --standalone --nproc_per_node=8 -m experiments.dsv4_muon_report_aligned.train --config experiments/dsv4_muon_report_aligned/configs/B_muon_hybrid.yaml --data-dir /path/to/data --dataset-fingerprint experiments/dsv4_muon_report_aligned/outputs/dataset_fingerprint.json --initial-weights experiments/dsv4_muon_report_aligned/outputs/shared_initial_weights.pt
torchrun --standalone --nproc_per_node=8 -m experiments.dsv4_muon_report_aligned.train --config experiments/dsv4_muon_report_aligned/configs/C_muon_standard.yaml --data-dir /path/to/data --dataset-fingerprint experiments/dsv4_muon_report_aligned/outputs/dataset_fingerprint.json --initial-weights experiments/dsv4_muon_report_aligned/outputs/shared_initial_weights.pt
```

At world size 8, each optimizer step consumes 262,144 tokens. `target_tokens=5e9`
therefore resolves to 19,074 steps (5,000,134,656 tokens).

Only after the controlled differences are categorized and the paired A/B/C run
metadata match should short LR sweep configs be derived at 0.5x, 0.75x, 1x,
1.5x, and 2x.
