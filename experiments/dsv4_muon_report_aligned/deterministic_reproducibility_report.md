# PyTorch 确定性训练：原理与实验结果

## 1. 原理

PyTorch 训练可以抽象为：

```text
theta_(t+1) = Update(theta_t, batch_t, rng_t, kernel_t)
loss_t      = Loss(theta_t, batch_t, kernel_t)
```

其中：

- `theta_t` 是当前模型参数；
- `batch_t` 是当前训练数据；
- `rng_t` 是随机数生成器状态；
- `kernel_t` 是 CUDA 算子和分布式归约路径。

要让两次训练的 loss 曲线逐点相同，这四部分必须全部一致。

### 1.1 固定 seed 为什么不够

固定 seed 只能控制显式随机数，例如：

- 模型参数初始化；
- Dropout 和随机采样；
- 数据 shuffle；
- DataLoader worker 的随机状态。

GPU 并行计算还存在与 seed 无关的不确定性。多个 CUDA 线程可能以不同顺序执行浮点累加，而浮点加法不满足严格结合律：

```text
(a + b) + c != a + (b + c)
```

因此，即使 seed 相同，原子累加、并行归约或不同 CUDA 内核仍可能产生最后几位不同的结果。这些微小误差会进入梯度和优化器状态，并在后续 step 中持续累积。

### 1.2 要固定哪些因素

严格复现 loss 曲线需要同时固定：

1. **初始参数**：相同 seed、代码和模型构造顺序，或直接加载同一份初始化权重。
2. **数据顺序**：相同数据文件、sampler seed、batch size、梯度累积次数和 GPU 数量。
3. **随机数状态**：固定 Python、PyTorch CPU、所有 CUDA device 和 DataLoader worker 的 RNG。
4. **CUDA 算法**：启用 PyTorch 确定性算法，固定 cuBLAS/cuDNN，关闭 TF32 和非确定性 SDPA 后端。
5. **分布式执行**：保持相同 world size、rank 顺序和通信环境。

其中 `torch.use_deterministic_algorithms(True)` 的作用不是消除所有随机性，而是在算子没有确定性实现时立即报错，避免训练在未知的不确定状态下继续运行。

严格复现只承诺相同硬件拓扑、GPU 数量和软件栈下的重复运行。更换 GPU、PyTorch、CUDA、cuDNN、NCCL 或驱动后，通常只能要求数值近似一致，不能保证位级一致。

## 2. 实验结果

在以下环境中执行了两次相互独立的训练：

| 项目 | 配置 |
| --- | --- |
| GPU | 8 × NVIDIA H800 80GB |
| PyTorch | `2.3.0a0+ebedce2` |
| CUDA | 12.3 |
| 精度 | BF16 训练，FP32 validation |
| 模型 | TorchForge 397M DSV4-inspired model |
| 序列长度 | 4096 |
| micro batch size | 4 / GPU |
| gradient accumulation | 2 |
| 随机种子 | 2026 |
| 训练长度 | 100 steps |
| 训练 token | 6,553,600 |

两次运行使用相同代码、配置、数据指纹和 GPU 顺序，并从头独立启动。比较时不使用平滑、插值或数值容差，要求：

```text
loss_run_1 == loss_run_2
```

比较结果如下：

| 指标 | 对齐点数 | 平均绝对差 | 最大绝对差 | 最大相对差 |
| --- | ---: | ---: | ---: | ---: |
| total loss | 100 | 0.0 | 0.0 | 0.0 |
| LM loss | 100 | 0.0 | 0.0 | 0.0 |
| MTP loss | 100 | 0.0 | 0.0 | 0.0 |
| auxiliary loss | 100 | 0.0 | 0.0 | 0.0 |
| gradient norm | 100 | 0.0 | 0.0 | 0.0 |
| learning rate | 100 | 0.0 | 0.0 | 0.0 |
| Muon update RMS | 100 | 0.0 | 0.0 | 0.0 |
| validation loss | 100 | 0.0 | 0.0 | 0.0 |

两份日志各包含 100 个训练点，累计 token 位置完全一致，没有缺失点或额外点。运行元数据中的模型配置、优化器、并行规模和数据 SHA-256 也全部匹配。

实验结果表明：在相同的 H800 硬件、PyTorch/CUDA 环境、代码、配置和数据条件下，两次独立训练的 loss 曲线实现了逐 step、零容差一致。
