# DSV3 Incremental Replacement Training Validation

This experiment validates training behavior while replacing one component at a time inside the same tiny DSV3 block.

Run from the repository root:

```bash
python experiments/dsv3_replacement/train.py \
  --attention pytorch \
  --ffn pytorch \
  --norm pytorch \
  --kv pytorch \
  --steps 100 \
  --output experiments/dsv3_replacement/reference_losses.json

python experiments/dsv3_replacement/train.py \
  --attention torchforge \
  --ffn pytorch \
  --norm pytorch \
  --kv pytorch \
  --steps 100 \
  --output experiments/dsv3_replacement/torchforge_mla_losses.json

python experiments/dsv3_replacement/compare.py \
  experiments/dsv3_replacement/reference_losses.json \
  experiments/dsv3_replacement/torchforge_mla_losses.json \
  --baseline experiments/dsv3_replacement/reference_losses.json
```

The training script uses component switches:

```bash
python experiments/dsv3_replacement/train.py --attention pytorch
python experiments/dsv3_replacement/train.py --attention torchforge
python experiments/dsv3_replacement/train.py --ffn torchforge
python experiments/dsv3_replacement/train.py --ffn moe
python experiments/dsv3_replacement/train.py --norm torchforge
```

Supported switches:

- `--attention pytorch|torchforge`
- `--ffn pytorch|torchforge|moe`
- `--norm pytorch|torchforge`
- `--kv pytorch|torchforge`

The first three switches select concrete implementations inside one shared block class. The KV switch is reserved
for future independent KV replacement; in this tiny DSV3 block the KV path is still embedded inside the selected
attention implementation.

Each result records loss, forward time, backward time, total step time, and CUDA peak memory when CUDA is used.
