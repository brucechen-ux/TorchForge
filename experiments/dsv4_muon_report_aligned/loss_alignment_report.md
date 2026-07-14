# Cross-project loss difference report

## Scope

This is an approximately 397M DeepSeek-V4-inspired TorchForge comparison
assembly. The technical report is authoritative for disclosed V4 mechanisms;
the adjacent read-only package is a peer implementation, not an oracle.
This is not an official or complete DeepSeek-V4 implementation.
The configured shapes imply 397,359,035 unique trainable parameters, matching
the comparison package's parameter inventory.

## Component mapping and differences

| 397M comparison item | TorchForge implementation | Provenance/status |
| --- | --- | --- |
| `embed_tokens`, RMSNorm, `lm_head` | Common Embedding/RMSNorm/LMHead | Explicit cross-project state-dict mapping |
| V4 Q/KV projections and norms | Configured common MLA | Shape and operation mapped |
| HCA/CSA compressor/indexer | Common HCA/CSA components | Shape and operation mapped |
| First three Hash-MoE layers | Common HashRouter plus learned score adapter | Top-1 weight normalization mapped |
| Remaining learned top-1 routers | Experiment V4 router using common conventions | 397M protocol; router kept in AdamW |
| Packed routed experts | Experiment packed 3-D module | Axis zero treated as independent logical matrices |
| Shared experts | Common SharedExpertMLP | Direct projection mapping |
| Ordinary residual blocks | Experiment decoder layer | 397M protocol deviation from report mHC |
| MTP depth 1 | Common MTP module plus ordinary-residual block | Fusion projection column halves mapped |
| Hybrid/Standard Muon | Corrected common Muon | Report Hybrid 8+2 plus experimental Standard-10 control |

The first CUDA comparison observed a CSA selection difference: TorchForge uses
stable sorting while the comparison package calls `torch.topk`. ReLU creates tied
zero scores, so the implementations can select different equal-scoring compressed
KV entries. The report defines a mathematical Top-k set but does not specify tie
ordering; therefore this remains a measured implementation difference.

## Mechanism checks

- Hybrid coefficients: eight `(3.4445, -4.7750, 2.0315)` followed by two `(2, -1.5, 0.5)`.
- Standard coefficients: ten `(2, -1.5, 0.5)` iterations (experimental control).
- Momentum: `M=0.95*M+G`; Nesterov input: `N=0.95*M+G`.
- Each 2-D matrix, including every axis-zero slice of packed 3-D experts, runs NS independently.
- Each logical update is scaled by `sqrt(max(rows, cols))*0.18`.
- Embedding, LM head, norms, routers, vectors, and scalars use AdamW in the fixed 397M protocol.
- Muon and auxiliary AdamW use the same numeric LR and scheduler; auxiliary AdamW uses `eps=1e-20`.
- AdamW baseline uses `eps=1e-8` as an experimental control.
- Weight decay is applied once as decoupled decay.

## Numerical status

<!-- PARITY_RESULTS_START -->
Status: **CONTROLLED CUDA SHORT COMPARISON MEASURED**.

All 110 mapped tensors were copied with no missing local parameters. Layers 0
and 1 matched exactly. The first difference was `layers.2.attn`, where FP32 max
absolute error was `0.2598247230`; FP32 logits max absolute error was
`0.1549579203`, and total-loss absolute error was `0.0006480217`. BF16 total-loss
absolute error was `0.0023994446`. The final measured gradient difference began
at `embed_tokens.weight`, with max absolute error `0.0182360932`.

These measurements record, rather than invalidate, the stable-sort versus
`torch.topk` difference. The two implementations shared mapped initial weights,
tokens, and training settings for this diagnostic. The historical A/B/C losses
in the comparison audit are invalid and are not curve-comparison targets.
<!-- PARITY_RESULTS_END -->

The repository CSV remains marked `FORMAL_CURVE_NOT_RUN`. Once both native logs
exist, `compare_curves.py` writes the complete exact-token-joined machine CSV;
short diagnostic rows are not fabricated from the console summary.

The unmodified comparison runner does not emit an initialization/source checksum
or Muon update RMS. Preserve the generated initialization provenance sidecar and
run the controlled shared-weight diagnostic from the same checkout immediately
before native runs. Formal curve CSV cells for comparison Muon RMS remain blank;
its `update_norm` has different semantics and is not substituted.

## Missing formal-run inputs

The comparison package and this workspace contain no training token files,
checkpoint, or corrected formal A/B/C logs. A real 5B-token curve cannot be
compared until the following are supplied:

- `train.bin` and `valid.bin` uint32 token files;
- `manifest.json` with file names, written token counts, dtype, and vocab size;
- `dataset_fingerprint.json` with full train/validation token-file SHA-256 values;
- a working BF16 CUDA/NCCL PyTorch environment;
- approval to launch the 8-GPU runs after controlled comparison and smoke checks.

The full commands are documented in `README.md`; they have not been launched.
