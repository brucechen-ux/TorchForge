# Small Kimi-K3 Experiment

This experiment assembles a text-only, principle-aligned Kimi-K3 model from
`torchforge.common` components. The fixed configuration has 25 decoder layers:
six repetitions of three KDA layers plus one Gated MLA layer, followed by a
final Gated MLA layer. The first FFN is dense and the remaining 24 use Stable
LatentMoE. It contains about 4.75B total parameters and 1.14B parameters active
for each token, including the one-layer MTP branch.

## Scope

- 18 KDA and 7 NoPE Gated MLA layers.
- Block Attention Residuals with four decoder layers per block.
- 64 routed latent experts, Top-4 routing, and two shared experts.
- Histogram Quantile Balancing updated once per optimizer step.
- Per-head Muon through head-packed 3-D Q/K/V parameters.
- One MTP layer.
- BF16, activation checkpointing, and eight-process DDP.

Vision, RL, expert/context parallelism, quantization, serving caches, and
production kernels are intentionally outside this experiment.

## KDA Backend

The public KDA component always includes a token-recurrent PyTorch reference
backend. The full training config selects `backend="fla"`; the environment must
provide a Flash Linear Attention build containing `fla.ops.kda.chunk_kda`. A
missing or incompatible FLA build raises an explicit error instead of silently
falling back to the slow reference scan.

## Data

The loader reuses TorchForge's existing uint32 memmap format and validates the
configured vocabulary against `manifest.json`. The fixed config expects a
49,152-token vocabulary and `train.bin`/`valid.bin` files under the configured
data directory.

## Training

```bash
torchrun --standalone --nproc_per_node=8 \
  -m experiments.k3_small.train \
  --config experiments/k3_small/configs/k3_4_8b.json \
  --data-dir /path/to/tokenized/data
```

Resume from a checkpoint with `--resume /path/to/step_XXXXXXXX.pt`. Checkpoints
include the model, Muon and AdamW states, scheduler, QB biases, RNG state, and
sampler cursor. The validation loader uses `validation_seq_len` (32K in the
fixed config).

For the required 32K forward/backward capability check, add
`--seq-len 32768 --max-steps 1 --skip-final-checkpoint` on the H100 host. This
override is an acceptance check, not a second automatic training phase.
