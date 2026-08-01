# TorchForge

TorchForge is a foundation components library for transformer research. Its
public package provides directly instantiable PyTorch modules for attention
(including KV compression), MLP, MoE, neural network layers, embeddings,
masks, and residual utilities without introducing a trainer, runtime,
inference engine, distributed framework, or model zoo.

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
from torchforge.common.attention import (
    CSACompressor,
    CausalMask,
    CompressedKVIndexer,
    GatedMLA,
    GQA,
    HCACompressor,
    KDAState,
    KimiDeltaAttention,
    MHA,
    MLA,
    MQA,
    SlidingWindowCausalMask,
)
from torchforge.common.mlp import FeedForward, GatedMLP
from torchforge.common.moe import (
    TopKRouter, HashRouter, QuantileBalancingRouter, ExpertMLP,
    SharedExpertMLP, StableLatentMoE, MoE,
)
from torchforge.common.nn import RMSNorm, UnweightedRMSNorm, SwiGLU, SiTUGLU, GEGLU, MLP
from torchforge.common.embedding import Embedding, RotaryEmbedding
from torchforge.common.lm_head import LMHead
from torchforge.common.position import PositionIds
from torchforge.common.residual import (
    ResidualAdd, ManifoldConstrainedHyperConnection,
    BlockAttentionResidual, BlockAttentionResidualState,
)
from torchforge.common.mtp import MultiTokenPredictionModule
from torchforge.common.optim import build_k3_optimizer_param_groups
```

Neural components are directly instantiable `nn.Module` classes; optimizer
helpers return ordinary PyTorch parameter groups.

## Component Assembly

TorchForge provides the pieces needed to assemble model stacks directly from
`torchforge.common` components. The DeepSeek examples are component-only
assemblies; the experiment-local K3 package contains its model and DDP trainer.

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
      mask/
    embedding/
    lm_head/
    mlp/
    moe/
    mtp/
    nn/
    optim/
    position/
    residual/
    train/
experiments/
  dsv3_assembly/
  dsv4_assembly/
  k3_small/
tests/
docs/
```

- `torchforge/common`: reusable foundation components.
- `experiments`: component assemblies and experiment-local training code.
- `tests`: public API and behavior tests for components.
- `docs`: project documentation.

## Design Principles

- Foundation components, not a training framework.
- Common components over model-specific implementations.
- Public APIs use `from torchforge.common.<family> import Component`.
- Components inherit directly from `torch.nn.Module`.
- No Core, Plugin, Factory, Registry, Builder, Manager, or Pipeline abstractions in common components.
