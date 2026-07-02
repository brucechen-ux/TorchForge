# TorchForge Documentation

TorchForge is organized as a foundation components library. The public component families are:

- `torchforge.common.attention`
- `torchforge.common.decoder`
- `torchforge.common.embedding`
- `torchforge.common.kv`
- `torchforge.common.lm_head`
- `torchforge.common.mask`
- `torchforge.common.moe`
- `torchforge.common.mtp`
- `torchforge.common.nn`
- `torchforge.common.position`
- `torchforge.common.residual`

Mathematical implementations belong in `torchforge.common`. DeepSeek-V3 and
DeepSeek-V4 assembly examples live under `experiments/` and instantiate these
components directly.

## Before First Release

The documentation set should be expanded with:

- Public API reference pages for each component family.
- Shape conventions and return-value conventions.
- DeepSeek-V3 and DeepSeek-V4 assembly walkthroughs.
- CI and release instructions.
