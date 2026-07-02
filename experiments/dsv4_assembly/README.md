# DeepSeek-V4 Assembly Example

This example assembles a DeepSeek-V4-style decoder stack directly from
`torchforge.common` Foundation Components. It does not define a reference model
class and does not wrap the stack in a model abstraction.

```bash
python experiments/dsv4_assembly/deepseek_v4_assembly.py --variant flash
python experiments/dsv4_assembly/deepseek_v4_assembly.py --variant pro
python experiments/dsv4_assembly/deepseek_v4_assembly.py --variant flash --paper-scale
python experiments/dsv4_assembly/deepseek_v4_assembly.py --variant pro --paper-scale
```
