# TorchForge

TorchForge is a foundation components library for transformer research. It provides directly instantiable PyTorch modules for attention, KV compression, MoE, and common neural network layers without introducing a training framework or model zoo.

## Install

From a local checkout:

```bash
pip install -e .
```

For development and tests:

```bash
pip install -e ".[dev]"
pytest
```

## Public API

TorchForge components are imported from family namespaces:

```python
from torchforge.common.attention import MLA, GQA, MQA, MHA
from torchforge.common.kv import HCACompressor, CSACompressor, CompressedKVIndexer
from torchforge.common.moe import TopKRouter, ExpertMLP, MoE
from torchforge.common.nn import RMSNorm, UnweightedRMSNorm, SwiGLU, GEGLU, FeedForward, MLP
```

Each public component is an `nn.Module` that can be instantiated and called directly.

## Repository Layout

```text
torchforge/
  common/
    attention/
    kv/
    moe/
    nn/
  patches/
experiments/
tests/
docs/
```

- `torchforge/common`: reusable foundation components.
- `torchforge/patches`: small model adapters that map model configs to common components.
- `experiments`: validation scripts, including incremental component replacement.
- `tests`: public API and behavior tests.
- `docs`: project documentation.

## Experiments

The DSV3 reference replacement experiment validates component swaps inside a small single-card
DeepSeek-V3-style causal language model:

```bash
python experiments/dsv3_torchforge/train.py --attention pytorch --ffn pytorch --norm pytorch --kv pytorch
python experiments/dsv3_torchforge/train.py --attention torchforge --ffn pytorch --norm pytorch --kv pytorch
python experiments/dsv3_torchforge/compare.py \
  experiments/dsv3_torchforge/attention_pytorch__ffn_pytorch__norm_pytorch__kv_pytorch_losses.json \
  experiments/dsv3_torchforge/attention_torchforge__ffn_pytorch__norm_pytorch__kv_pytorch_losses.json
```

## Design Principles

- Foundation components, not a training framework.
- Common components over model-specific implementations.
- Public APIs use `from torchforge.common.<family> import Component`.
- Components inherit directly from `torch.nn.Module`.
- No Core, Plugin, Factory, Registry, Builder, Manager, or Pipeline abstractions in common components.
