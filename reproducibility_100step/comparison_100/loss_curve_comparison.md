# Cross-project loss curve comparison

The comparison project is a peer implementation, not a numerical oracle. Rows are joined only at exact cumulative-token positions; no interpolation is used.

- TorchForge log: `/gemini/code/TorchForge/experiments/dsv4_muon_report_aligned/outputs/repro_test/run_5/loss_log.jsonl`
- Comparison log: `/gemini/code/TorchForge/experiments/dsv4_muon_report_aligned/outputs/repro_test/run_6/loss_log.jsonl`
- Aligned points: `100`
- TorchForge-only points: `0`
- Comparison-only points: `0`
- Matched run-metadata fields: `40`
- Mismatched run-metadata fields: `0`
- Unavailable run-metadata fields: `2`
- Critical unavailable fields: `0`

| metric | compared points | mean abs diff | max abs diff | max relative diff |
| --- | ---: | ---: | ---: | ---: |
| lr | 100 | 0 | 0 | 0 |
| total_loss | 100 | 0 | 0 | 0 |
| lm_loss | 100 | 0 | 0 | 0 |
| mtp_loss | 100 | 0 | 0 | 0 |
| aux_loss | 100 | 0 | 0 | 0 |
| grad_norm | 100 | 0 | 0 | 0 |
| muon_update_rms | 100 | 0 | 0 | 0 |
| validation_loss | 1 | 0 | 0 | 0 |

The complete per-token-position values and differences are in `loss_curve_comparison.csv`. Blank cells mean that the source log did not expose that metric.
