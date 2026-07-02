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

TorchForge provides the pieces needed to assemble DeepSeek-style stacks without
a model-zoo wrapper:

```python
import torch
from torchforge.common.decoder import DecoderLayer
from torchforge.common.embedding import Embedding, RotaryEmbedding
from torchforge.common.lm_head import LMHead
from torchforge.common.mask import CausalMask
from torchforge.common.nn import RMSNorm
from torchforge.common.position import PositionIds

input_ids = torch.randint(0, 128, (2, 16))
embed = Embedding(vocab_size=128, hidden_size=32)
positions = PositionIds()(input_ids)
rotary = RotaryEmbedding(head_dim=4)(positions)
mask = CausalMask()(input_ids, dtype=torch.float32)
layer = DecoderLayer(
    hidden_size=32,
    num_attention_heads=4,
    num_key_value_heads=4,
    intermediate_size=64,
    q_lora_rank=8,
    kv_lora_rank=8,
    qk_nope_head_dim=4,
    qk_rope_head_dim=4,
    v_head_dim=8,
)
hidden = layer(embed(input_ids), position_embeddings=rotary, attention_mask=mask, return_dict=False)
logits = LMHead(hidden_size=32, vocab_size=128)(RMSNorm(32)(hidden))
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
tests/
docs/
```

- `torchforge/common`: reusable foundation components.
- `experiments`: DeepSeek-V3 and DeepSeek-V4 component assembly examples.
- `tests`: public API and behavior tests.
- `docs`: project documentation.

## Experiments

The assembly examples show how to build DeepSeek-V3 and DeepSeek-V4 style stacks
directly from `torchforge.common` components:

```bash
python experiments/dsv3_assembly/deepseek_v3_assembly.py
python experiments/dsv4_assembly/deepseek_v4_assembly.py --variant flash
python experiments/dsv4_assembly/deepseek_v4_assembly.py --variant pro
```

## Design Principles

- Foundation components, not a training framework.
- Common components over model-specific implementations.
- Public APIs use `from torchforge.common.<family> import Component`.
- Components inherit directly from `torch.nn.Module`.
- No Core, Plugin, Factory, Registry, Builder, Manager, or Pipeline abstractions in common components.
