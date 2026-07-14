# Loss alignment report

## Scope

This is a TorchForge assembly aligned to the supplied reduced, approximately
397M DeepSeek-V4-like audit package. It is not an official or complete
DeepSeek-V4 implementation. The adjacent package was treated as read-only.
The configured shapes imply 397,359,035 unique trainable parameters, matching
the reference audit's parameter inventory.

## Component mapping and differences

| Reference package | TorchForge implementation | Alignment status |
| --- | --- | --- |
| `embed_tokens`, RMSNorm, `lm_head` | Common Embedding/RMSNorm/LMHead | Explicit state-dict mapping |
| V4 Q/KV projections and norms | Configured common MLA | Shape and operation mapped |
| HCA/CSA compressor/indexer | Common HCA/CSA components | Shape and operation mapped |
| First three Hash-MoE layers | Common HashRouter plus learned score adapter | Top-1 weight normalization mapped |
| Remaining learned top-1 routers | Experiment V4 router using common conventions | Router kept in AdamW |
| Packed routed experts | Experiment packed 3-D module | Axis zero treated as independent logical matrices |
| Shared experts | Common SharedExpertMLP | Direct projection mapping |
| Ordinary residual blocks | Experiment decoder layer | Existing mHC assembly intentionally not reused |
| MTP depth 1 | Common MTP module plus ordinary-residual block | Fusion projection column halves mapped |
| Hybrid/Standard Muon | Corrected common Muon | Hybrid 8+2 and Standard 10 implemented |

Known potential tie-only difference: the common CSA indexer uses stable sorting,
while the reference calls `torch.topk`. With exactly tied index scores this can
choose a different equal-scoring entry. The parity output reports this as the
first differing module if it occurs.

## Mechanism checks

- Hybrid coefficients: eight `(3.4445, -4.7750, 2.0315)` followed by two `(2, -1.5, 0.5)`.
- Standard coefficients: ten `(2, -1.5, 0.5)` iterations.
- Momentum: `M=0.95*M+G`; Nesterov input: `N=0.95*M+G`.
- Each 2-D matrix, including every axis-zero slice of packed 3-D experts, runs NS independently.
- Each logical update is scaled by `sqrt(max(rows, cols))*0.18`.
- Embedding, LM head, norms, routers, vectors, and scalars use AdamW.
- Muon and auxiliary AdamW use the same numeric LR and scheduler; auxiliary AdamW uses `eps=1e-20`.
- AdamW baseline uses `eps=1e-8`.
- Weight decay is applied once as decoupled decay.

## Numerical status

<!-- PARITY_RESULTS_START -->
Status: **NOT RUN on this workstation**.

The workstation's installed PyTorch cannot load `c10_cuda.dll`, and the user
requested test code only. No FP32, BF16, gradient, parameter-delta, short-loss,
or 2-rank result is claimed here. `parity.py` records all requested actual errors
and overwrites the placeholder CSV on a working machine.

Consequently, the first actual residual deviation and short-run loss difference
are not yet measured. They must not be inferred from the invalid historical A/B/C
losses in the reference audit.
<!-- PARITY_RESULTS_END -->

## Missing formal-run inputs

The reference package and this workspace contain no training token files,
checkpoint, or corrected formal A/B/C logs. A real 5B-token curve cannot be
accepted until the following are supplied:

- `train.bin` and `valid.bin` uint32 token files;
- `manifest.json` with file names, written token counts, dtype, and vocab size;
- a working BF16 CUDA/NCCL PyTorch environment;
- approval to launch the 8-GPU runs after parity and smoke checks pass.

The full commands are documented in `README.md`; they have not been launched.
