# TorchForge

TorchForge is a foundation components library for DeepSeek-style transformer research. It provides directly instantiable PyTorch modules for attention, KV compression, MoE, neural network layers, embeddings, masks, residual utilities, and assembly examples without introducing a trainer, runtime, inference engine, distributed framework, or model zoo.

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
from torchforge.common.moe import TopKRouter, HashRouter, ExpertMLP, SharedExpertMLP, MoE
from torchforge.common.nn import RMSNorm, UnweightedRMSNorm, SwiGLU, GEGLU, FeedForward, MLP
from torchforge.common.embedding import Embedding, RotaryEmbedding
from torchforge.common.lm_head import LMHead
from torchforge.common.decoder import DecoderLayer
from torchforge.common.mask import CausalMask, SlidingWindowCausalMask
from torchforge.common.position import PositionIds
from torchforge.common.residual import ResidualAdd, ManifoldConstrainedHyperConnection
from torchforge.common.mtp import MultiTokenPredictionModule
```

Each public component is an `nn.Module` that can be instantiated and called directly.

## Component Assembly

TorchForge provides the pieces needed to assemble DeepSeek-V3 and DeepSeek-V4 style stacks directly from `torchforge.common` components. The assembly examples intentionally do not define full model classes.

```bash
python experiments/dsv3_assembly/deepseek_v3_assembly.py
python experiments/dsv4_assembly/deepseek_v4_assembly.py --variant flash
python experiments/dsv4_assembly/deepseek_v4_assembly.py --variant pro
```

## Repository Layout

```text
torchforge/
  common/
    attention/
    decoder/
    embedding/
    kv/
    lm_head/
    mask/
    moe/
    mtp/
    nn/
    position/
    residual/
experiments/
  dsv3_assembly/
  dsv4_assembly/
tests/
docs/
```

- `torchforge/common`: reusable foundation components.
- `experiments`: DeepSeek-V3 and DeepSeek-V4 component assembly examples.
- `tests`: public API and behavior tests for components.
- `docs`: project documentation.

## Design Principles

- Foundation components, not a training framework.
- Common components over model-specific implementations.
- Public APIs use `from torchforge.common.<family> import Component`.
- Components inherit directly from `torch.nn.Module`.
- No Core, Plugin, Factory, Registry, Builder, Manager, or Pipeline abstractions in common components.
