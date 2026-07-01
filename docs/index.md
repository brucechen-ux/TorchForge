# TorchForge Documentation

TorchForge is organized as a foundation components library. The public component families are:

- `torchforge.common.attention`
- `torchforge.common.kv`
- `torchforge.common.moe`
- `torchforge.common.nn`

Model patches should stay thin: they adapt model configuration, select component policies, and assemble components. Mathematical implementations belong in `torchforge.common`.

## Before First Release

The documentation set should be expanded with:

- Public API reference pages for each component family.
- Shape conventions and return-value conventions.
- Migration notes for model patches.
- Incremental replacement experiment guide based on `experiments/dsv3_reference` and `experiments/dsv3_torchforge`.
- CI and release instructions.
