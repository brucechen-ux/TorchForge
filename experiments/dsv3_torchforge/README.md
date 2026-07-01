# DSV3 Reference Replacement Experiment

This experiment replaces components inside a small single-card DeepSeek-V3-style causal language model.

The pure PyTorch reference model lives in `experiments/dsv3_reference`. It includes embedding, RMSNorm, MLA, dense FFN or local MoE, decoder layers, final norm, and LM head.

Run from the repository root:

```bash
python experiments/dsv3_torchforge/train.py \
  --attention pytorch \
  --ffn pytorch \
  --norm pytorch \
  --kv pytorch \
  --layers 2 \
  --steps 100 \
  --output experiments/dsv3_torchforge/reference_losses.json

python experiments/dsv3_torchforge/train.py \
  --attention torchforge \
  --ffn pytorch \
  --norm pytorch \
  --kv pytorch \
  --layers 2 \
  --steps 100 \
  --output experiments/dsv3_torchforge/torchforge_mla_losses.json

python experiments/dsv3_torchforge/compare.py \
  experiments/dsv3_torchforge/reference_losses.json \
  experiments/dsv3_torchforge/torchforge_mla_losses.json \
  --baseline experiments/dsv3_torchforge/reference_losses.json
```

To run a strict single-step diagnostic with reference weights copied into the replacement model:

```bash
python experiments/dsv3_torchforge/diagnose.py \
  --attention torchforge \
  --ffn pytorch \
  --norm pytorch \
  --kv pytorch \
  --copy-reference-weights \
  --device cuda \
  --output experiments/dsv3_torchforge/mla_diagnostics.json
```

Training can also embed diagnostics in the result JSON:

```bash
python experiments/dsv3_torchforge/train.py \
  --attention torchforge \
  --ffn pytorch \
  --norm pytorch \
  --kv pytorch \
  --copy-reference-weights \
  --diagnostics \
  --device cuda \
  --steps 100 \
  --output experiments/dsv3_torchforge/torchforge_mla_losses.json
```

Supported component switches:

- `--attention pytorch|torchforge`
- `--ffn pytorch|torchforge|moe`
- `--norm pytorch|torchforge`
- `--kv pytorch|torchforge`

`--kv torchforge` is reserved for future independent KV replacement. In this first reference model, KV projection is still part of the selected attention implementation.

Diagnostics record activation summaries, parameter diffs, gradient diffs, and weight-copy status as JSON.

To validate the diagnostics system itself:

```bash
python experiments/dsv3_torchforge/self_test_diagnostics.py \
  --device cuda \
  --output experiments/dsv3_torchforge/diagnostics_self_test.json
```

To generate diagnostics for each replacement stage:

```bash
python experiments/dsv3_torchforge/run_incremental_diagnostics.py \
  --device cuda \
  --output-dir experiments/dsv3_torchforge/diagnostics
```

To summarize multiple diagnostics files:

```bash
python experiments/dsv3_torchforge/summarize_diagnostics.py \
  experiments/dsv3_torchforge/diagnostics/mla_diagnostics.json \
  experiments/dsv3_torchforge/diagnostics/rmsnorm_diagnostics.json \
  experiments/dsv3_torchforge/diagnostics/feedforward_diagnostics.json \
  experiments/dsv3_torchforge/diagnostics/moe_diagnostics.json
```
