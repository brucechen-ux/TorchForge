# TorchForge - 深度学习组件拼装库

TorchForge 是一个模块化的深度学习组件库，专注于前沿模型架构的可复现消融实验。所有组件均可独立使用或组合，支持等比例缩放以适应不同的硬件环境。

---

## 🎯 核心特性

- **📦 组件化设计**: 每个模块独立可测，API 统一
- **🔬 消融实验友好**: 所有组件可等比例缩小，支持单卡测试
- **✅ 数学原理正确**: 严格遵循原始论文，带完整单元测试
- **🚀 前沿架构支持**: 实现 DeepSeek V3/V4、Qwen3.8-Flash、GLM-5.3-Flash、Kimi K3 等

---

## 📚 已实现组件清单

### 注意力机制 (Attention)

#### 递归/线性注意力
- **KDA** (`KimiDeltaAttention`) - Kimi K3 Delta Attention，channel-wise decay
- **GDN** (`GatedDeltaNet`) - Qwen3.8-Flash Gated Delta Net，fixed recurrent state

#### 稀疏注意力
- **DSA** (`GLMDynamicSparseAttention`) ⭐ **NEW** - GLM-5.3-Flash 动态稀疏注意力
  - KPool×4 压缩 + Top-K 索引 + Sparse MLA
  - O(L²) → O(L×K) 复杂度
- **QSA** (`QwenQuerySparseAttention`) ⭐ **NEW** - Qwen3.8-Flash 查询稀疏注意力
  - Micro-block 索引器 + Block-sparse attention
  - 更友好的内存访问模式

#### 多头潜在注意力 (MLA)
- **MLA** - Multi-head Latent Attention，KV 压缩
- **Gated MLA** - Kimi K3 输出门控 MLA
- **Sparse MLA** - DSA 中的稀疏 MLA 变体

#### 压缩 KV 注意力
- **CSA** - Compressed Sparse Attention (DeepSeek V4)
- **HCA** - Hierarchical Compressed Attention (DeepSeek V4)

#### 标准注意力
- **MHA** - Multi-Head Attention
- **GQA** - Grouped Query Attention
- **MQA** - Multi-Query Attention

#### 索引器 (Indexers)
- **CompressedKVIndexer** - 压缩 KV 索引
- **MicroBlockIndexer** ⭐ **NEW** - 微块级稀疏索引
- **KPool** ⭐ **NEW** - K 维池化压缩
- **HierarchicalIndexer** - 层次化索引
- **LightningIndexer** - 快速 token 打分
- **StreamingAwareIndexer** - 流式感知索引

### 残差连接 (Residual)

- **mHC** (`ManifoldConstrainedHyperConnection`) - 4-way 流形约束超连接
  - Sinkhorn-Knopp 投影
  - DeepSeek V4 / GLM-5.3-Flash 使用
- **Gated Residual** (`QwenGatedResidual`) ⭐ **NEW** - Qwen 简化版 4-way 残差
  - Element-wise Read + Per-branch scalar Write
  - **无 4×4 branch mixing**，更低内存访问
- **Attention Residuals** (`BlockAttentionResidual`) - Kimi K3 跨深度历史检索
- **Residual Add** - 标准残差连接

### 混合专家 (MoE)

- **GLM MoE** (`GLM53FlashMoE`) - GLM-5.3-Flash MoE
  - 前 3 层 Dense FFN + 后 42 层 Sparse MoE
  - 288 routed / Top-8 + 1 shared
- **Stable Latent MoE** - Kimi K3 稳定潜在 MoE
  - 896 routed / Top-16 + 2 shared
  - Quantile Balancing
- **Standard MoE** - 基础 MoE 实现
- **Hash Router** - DeepSeek V4 哈希路由
- **Quantile Router** - Kimi K3 分位数均衡路由
- **TopK Router** - 标准 Top-K 路由

### Embedding

- **Token Embedding** - 标准 token lookup
- **N-gram Embedding** (`NgramEmbedding`) ⭐ **NEW** - Qwen N-gram 查找表
  - Bigram + Trigram lookup
  - ~51B 参数容量扩展（可缩小）
  - 确定性哈希，极低计算开销
- **N-gram Embedding Layer** (`NgramEmbeddingLayer`) ⭐ **NEW** - Layer 2 注入层
- **Rotary Embedding** - RoPE 位置编码

### FFN & 激活函数

- **Gated MLP** - SwiGLU / GeGLU
- **SiTU-GLU** - Kimi K3 激活函数
- **Clamp-SwiGLU** - DeepSeek V4 限幅激活
- **Standard FFN** - 标准前馈网络

### 优化器 (Optimizer)

- **Muon** - 矩阵级预条件优化器
  - Hybrid Newton-Schulz (DeepSeek)
  - Per-Head Muon (Kimi K3)
- **AdamW** - 标准 AdamW

### 其他组件

- **RMSNorm / LayerNorm** - 归一化层
- **Causal Mask** - 因果注意力 mask
- **Sliding Window Mask** - 滑动窗口 mask
- **Cross Entropy Loss** - 交叉熵损失
- **LM Head** - 语言模型输出层
- **MTP** (Multi-Token Prediction) - 多 token 预测

---

## 🆕 最新更新 (2026-09-06)

### 新增 GLM-5.3-Flash 组件

1. **Dynamic Sparse Attention (DSA)**
   - 文件: `torchforge/common/attention/dsa.py`
   - 组件: `KPool`, `DSAIndexer`, `GLMDynamicSparseAttention`
   - 特性: KPool×4 压缩 + Top-2048 稀疏索引 + Sparse MLA

### 新增 Qwen3.8-Flash-Next 组件

2. **Query Sparse Attention (QSA)**
   - 文件: `torchforge/common/attention/qsa.py`
   - 组件: `MicroBlockIndexer`, `QwenQuerySparseAttention`
   - 特性: Block-level 稀疏 (512 blocks × 4 tokens = 2048 budget)

3. **Gated Residual**
   - 文件: `torchforge/common/residual/gated_residual.py`
   - 组件: `QwenGatedResidual`
   - 特性: 简化版 mHC，无 Sinkhorn，更低内存访问

4. **N-gram Embedding**
   - 文件: `torchforge/common/embedding/ngram_embedding.py`
   - 组件: `NgramEmbedding`, `NgramEmbeddingLayer`
   - 特性: 51B lookup 容量，极低计算开销

详细文档: `docs/新增组件实现总结.md`

---

## 🚀 快速开始

### 安装

```bash
pip install torch  # 需要 PyTorch
git clone <repo-url>
cd TorchForge
pip install -e .
```

### 使用示例

#### GLM DSA (动态稀疏注意力)

```python
from torchforge.common.attention import DSA
import torch

dsa = DSA(
    hidden_size=256,
    num_attention_heads=8,
    num_key_value_heads=4,
    q_lora_rank=128,
    kv_lora_rank=256,
    qk_nope_head_dim=64,
    v_head_dim=64,
    compress_rate=4,     # KPool 压缩率
    top_k=128,           # 稀疏预算
)

hidden_states = torch.randn(2, 512, 256)
output = dsa(hidden_states)
print(output['hidden_states'].shape)  # (2, 512, 256)
```

#### Qwen QSA (查询稀疏注意力)

```python
from torchforge.common.attention import QSA
import torch

qsa = QSA(
    hidden_size=256,
    num_attention_heads=8,
    num_key_value_heads=4,
    head_dim=32,
    block_size=4,        # 块大小
    num_blocks=32,       # 选择 32 块 (128 tokens)
)

hidden_states = torch.randn(2, 128, 256)
output = qsa(hidden_states)
print(output['hidden_states'].shape)  # (2, 128, 256)
```

#### Qwen Gated Residual

```python
from torchforge.common.residual import GatedResidual
import torch

gr = GatedResidual(
    hidden_size=256,
    num_branches=4,
    bottleneck_rank=64,
)

# 初始化残差状态
hidden = torch.randn(2, 128, 256)
state = gr.init_state(hidden)  # (2, 128, 4, 256)

# 模块输出
module_out = torch.randn(2, 128, 256)

# 更新残差并读取下一输入
next_state, next_input = gr(state, module_out)
print(next_state.shape)   # (2, 128, 4, 256)
print(next_input.shape)   # (2, 128, 256)
```

#### Qwen N-gram Embedding

```python
from torchforge.common.embedding import NgramEmbeddingLayer
import torch

ngram_layer = NgramEmbeddingLayer(
    vocab_size=50000,
    hidden_size=256,
    max_ngram=3,         # Bigram + Trigram
    table_size=1_000_000,
    injection_mode="add",
)

# Layer 2 注入
hidden_states = torch.randn(2, 32, 256)
input_ids = torch.randint(0, 50000, (2, 32))
enhanced = ngram_layer(hidden_states, input_ids)
print(enhanced.shape)  # (2, 32, 256)
```

更多示例: `docs/新增组件快速参考.md`

---

## 🧪 测试

### 运行所有测试

```bash
pytest tests/ -v
```

### 运行新增组件测试

```bash
pytest tests/test_dsa_public_api.py -v              # DSA
pytest tests/test_qsa_public_api.py -v              # QSA
pytest tests/test_gated_residual_public_api.py -v   # Gated Residual
pytest tests/test_ngram_embedding_public_api.py -v  # N-gram Embedding
```

### 快速验证

```bash
python validate_new_components.py
```

---

## 📖 文档

- **实现总结**: `docs/新增组件实现总结.md` - 完整技术文档
- **快速参考**: `docs/新增组件快速参考.md` - 速查卡片
- **原始对比**: 四模型预训练结构对比文档 - 架构对比分析

---

## 🔬 消融实验配置

### 单卡 H100 推荐配置

| 参数 | 原始配置 | 单卡测试 |
|------|----------|----------|
| `hidden_size` | 2560-4096 | 256-512 |
| `num_layers` | 45-48 | 12-24 |
| `num_heads` | 32-40 | 8-16 |
| `top_k (DSA)` | 2048 | 128-256 |
| `num_blocks (QSA)` | 512 | 32-64 |
| `ngram_table` | 20M | 1M |
| `batch_size` | 大规模 | 4-8 |
| `seq_len` | 1M | 2K-8K |

所有组件支持等比例缩放，保持核心架构特性不变。

---

## 📊 组件对比表

### Attention 机制对比

| 模型 | 主干 Attention | 补偿层 Attention | 复杂度 | 特点 |
|------|---------------|------------------|--------|------|
| **Kimi K3** | KDA (69层) | Gated MLA (24层) | O(L×D) + O(L²) | 全局 token-level 检索 |
| **GLM-5.3-Flash** | KDA (34层) | DSA (11层) | O(L×D) + O(L×K) | Token-level 稀疏 (K=2048) |
| **Qwen3.8-Flash** | GDN (36层) | QSA (12层) | O(L×D) + O(L×B×S) | Block-level 稀疏 (B×S=2048) |
| **DeepSeek V4** | CSA + HCA | - | O(L×C) | Compressed KV |

### Residual 机制对比

| 模型 | Residual 类型 | Read 机制 | Write 机制 | Branch Mixing |
|------|--------------|-----------|-----------|---------------|
| **DeepSeek V4** | mHC | 向量权重 | 向量权重 | 4×4 Sinkhorn |
| **GLM-5.3-Flash** | mHC | 向量权重 | 向量权重 | 4×4 Sinkhorn |
| **Qwen3.8-Flash** | Gated Residual | Element-wise gate | Scalar gate | **无** |
| **Kimi K3** | AttnRes | 跨深度检索 | 累积 | - |

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

贡献指南：
1. Fork 本仓库
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

---

## 📄 许可证

MIT License - 详见 `LICENSE` 文件

---

## 🙏 致谢

- **DeepSeek V3/V4** - Compressed KV Attention, mHC, MoE innovations
- **Qwen3.8-Flash-Next** - GDN, QSA, Gated Residual, N-gram Embedding
- **GLM-5.3-Flash** - DSA, Hybrid Attention, Native Multimodal
- **Kimi K3** - KDA, Gated MLA, Per-Head Muon, AttnRes
- **Muon Optimizer** - Matrix-level preconditioning

---

## 📮 联系

项目链接: [TorchForge Repository]

**实现状态**: ✅ 所有核心组件已实现，数学原理正确，可用于消融实验
