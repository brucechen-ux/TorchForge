# TorchForge

TorchForge is a foundation components library for transformer research. Its
public package provides directly instantiable PyTorch modules for attention
(including KV compression), MLP, MoE, neural network layers, embeddings,
masks, losses, and residual utilities, together with reference model assemblies
built from those components.

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
from torchforge.common.loss import CausalLMLoss
from torchforge.common.position import PositionIds
from torchforge.common.residual import (
    ResidualAdd, ManifoldConstrainedHyperConnection,
    BlockAttentionResidual, BlockAttentionResidualState,
)
from torchforge.common.mtp import MultiTokenPredictionModule
from torchforge.common.optim import (
    AdamW, Muon, build_param_groups,
    build_hybrid_optimizer_param_groups, build_k3_optimizer_param_groups,
)
from torchforge.common.train import TrainStep, random_token_batches
```

Neural components are directly instantiable `nn.Module` classes; optimizer
helpers return ordinary PyTorch parameter groups.

## Gated Activations

`SwiGLU`, `GEGLU`, and `SiTUGLU` accept either a `(gate, value)` tuple or a
single tensor whose last dimension is split evenly into gate and value halves.

`SiTUGLU` is the Kimi-K3 soft-capped gated activation:

```text
beta_gate * tanh(gate / beta_gate) * sigmoid(gate) * beta_up * tanh(value / beta_up)
```

With the defaults `beta_gate=4.0` and `beta_up=25.0` the output magnitude is
bounded by `beta_gate * beta_up`, and the activation stays close to SwiGLU while
pre-activations are small. `beta_gate` and `beta_up` are plumbed through
`FeedForward`, `GatedMLP`, `ExpertMLP`, `SharedExpertMLP`, `MoE` (as
`expert_beta_gate` / `expert_beta_up`), and `StableLatentMoE`.

Note that `FeedForward` uses one fused `up_proj` of width `2 * intermediate_size`
for gated activations, while `GatedMLP` uses separate `gate_proj` and `up_proj`
layers. The two are mathematically equivalent but their parameter layouts are not
interchangeable.

## Reference Models

Reference models live under `torchforge.model`. Both DeepSeek packages are
component-only assemblies that print their layout and can run a minimal
training loop on random data.

```bash
python -m torchforge.model.deepseekV3.deepseek_v3_assembly
python -m torchforge.model.deepseekV4.deepseek_v4_assembly --variant flash
python -m torchforge.model.deepseekV4.deepseek_v4_assembly --variant pro
```

Useful flags: `--paper-scale` prints the paper-scale component layout, `--train`
runs the training loop, and `--steps` / `--lr` / `--batch-size` / `--seq-length`
control it.

## Repository Layout

```text
torchforge/
  common/
    attention/
      mask/
    embedding/
    lm_head/
    loss/
    mlp/
    moe/
    mtp/
    nn/
    optim/
    position/
    residual/
    train/
  model/
    deepseekV3/
    deepseekV4/
experiments/
  dsv4_muon_report_aligned/
tests/
docs/
```

- `torchforge/common`: reusable foundation components.
- `torchforge/model`: reference model assemblies built from common components.
- `experiments`: isolated experiments that compare or combine components.
- `tests`: public API and behavior tests for components.
- `docs`: project documentation.

## Design Principles

- Foundation components with explicit reference model assemblies, not a general
  training framework.
- Common components over model-specific implementations.
- Public APIs use `from torchforge.common.<family> import Component`.
- Components inherit directly from `torch.nn.Module`.
- No Core, Plugin, Factory, Registry, Builder, Manager, or Pipeline abstractions in common components.

## DeepSeek-V4.1 Attention

`CSA2Attention`, `CSA2Compressor`, `CSA2SharedCache`, `CSA2LayerState`, and
`HierarchicalSparseIndexer` are available from `torchforge.common.attention`.
They implement shared-key/value MQA with local SWA, learned block compression,
Full/Reindex/Reuse modes, CED encoder-sourced global KV, and per-query hierarchical
selection. See the [attention audit and usage guide](docs/DeepSeek_V4.1_attention_audit.md)
for report mappings, corrected API shapes, quantization emulation, and validation
limits. The existing DeepSeek-V4 model assembly remains a V4 model.
