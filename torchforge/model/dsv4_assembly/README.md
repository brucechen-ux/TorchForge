# DeepSeek-V4 Assembly Example

This example assembles a DeepSeek-V4-style decoder stack directly from
`torchforge.common` Foundation Components. It does not define a reference model
class and does not wrap the stack in a model abstraction.

```bash
python -m torchforge.model.dsv4_assembly.deepseek_v4_assembly --variant flash
python -m torchforge.model.dsv4_assembly.deepseek_v4_assembly --variant pro
python -m torchforge.model.dsv4_assembly.deepseek_v4_assembly --variant flash --paper-scale
python -m torchforge.model.dsv4_assembly.deepseek_v4_assembly --variant pro --paper-scale
```
