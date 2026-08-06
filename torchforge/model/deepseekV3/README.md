# DeepSeek-V3 Assembly Example

This example assembles a DeepSeek-V3-style decoder stack directly from
`torchforge.common` Foundation Components. It does not define a reference model
class and does not wrap the stack in a model abstraction.

```bash
python -m torchforge.model.dsv3_assembly.deepseek_v3_assembly
python -m torchforge.model.dsv3_assembly.deepseek_v3_assembly --paper-scale
```
