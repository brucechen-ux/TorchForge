# torchforge.common.nn

`torchforge.common.nn` contains general neural-network building blocks. These are
foundation components, not model adapters and not training framework utilities.

Public components:

- `RMSNorm`
- `UnweightedRMSNorm`
- `SwiGLU`
- `GEGLU`
- `FeedForward`
- `MLP`

Design principles:

- Each component directly inherits `torch.nn.Module`.
- Each component can be imported, instantiated, and called directly.
- Components are model-neutral and contain no DeepSeek, DSV, Qwen, or Llama logic.
- The family does not introduce Core, Plugin, Factory, Registry, Base, or Pipeline abstractions.
- Components favor explicit PyTorch-style constructor arguments such as `hidden_size`,
  `intermediate_size`, `dropout`, and `bias`.
