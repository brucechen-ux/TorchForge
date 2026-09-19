"""DeepSeek-V4.1-Flash 组件验证脚本.

验证所有核心组件的基本功能:
1. CSA2 三种模式 (Full/Reindex/Reuse)
2. Hierarchical Sparse Indexer
3. Single-Pass mHC
4. 跨层集成

运行: python validate_deepseek_v41_components.py
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("=" * 80)
print("DeepSeek-V4.1-Flash 组件验证")
print("=" * 80)
print()

# 测试导入
print("1. 测试组件导入...")
try:
    from torchforge.common.attention import (
        CSA2Attention,
        CSA2Compressor,
        CSA2Mode,
        CSA2SharedCache,
        HierarchicalSparseIndexer,
    )
    from torchforge.common.residual import SinglePassMHC, SinglePassMHCBlock
    
    print("   ✓ 所有组件导入成功")
except ImportError as e:
    print(f"   ✗ 导入失败: {e}")
    sys.exit(1)

print()

# 测试 CSA2 基本功能
print("2. 测试 CSA2 基本功能...")
try:
    import torch
    
    # 创建共享缓存
    shared_cache = CSA2SharedCache()
    
    # 配置
    config = {
        "hidden_size": 256,
        "num_attention_heads": 8,
        "num_key_value_heads": 1,
        "head_dim": 32,
        "compress_rate": 2,
        "top_k": 16,
        "q_lora_rank": 128,
        "index_num_heads": 4,
        "index_head_dim": 32,
    }
    
    batch_size, seq_len = 2, 32
    
    # Full Mode
    compressor_full = CSA2Compressor(
        **config,
        mode=CSA2Mode.FULL,
        shared_cache=shared_cache,
    )
    
    hidden_states = torch.randn(batch_size, seq_len, config["hidden_size"])
    q_residual = torch.randn(batch_size, seq_len, config["q_lora_rank"])
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    
    main_kv, top_k_indices = compressor_full(hidden_states, q_residual, position_ids, layer_idx=0)
    
    # 验证形状
    compressed_len = seq_len // config["compress_rate"]
    assert main_kv.shape == (batch_size, config["num_key_value_heads"], compressed_len, config["head_dim"])
    assert top_k_indices.shape == (batch_size, seq_len, config["top_k"])
    
    print("   ✓ CSA2 Full Mode 工作正常")
    
    # Reindex Mode
    compressor_reindex = CSA2Compressor(
        **config,
        mode=CSA2Mode.REINDEX,
        shared_cache=shared_cache,
    )
    
    q_residual_new = torch.randn(batch_size, seq_len, config["q_lora_rank"])
    main_kv_reindex, indices_reindex = compressor_reindex(
        hidden_states, q_residual_new, position_ids, layer_idx=1
    )
    
    # 验证 KV 重用
    assert torch.equal(main_kv_reindex, shared_cache.main_kv)
    print("   ✓ CSA2 Reindex Mode 工作正常 (KV 重用验证通过)")
    
    # Reuse Mode
    compressor_reuse = CSA2Compressor(
        **config,
        mode=CSA2Mode.REUSE,
        shared_cache=shared_cache,
    )
    
    main_kv_reuse, indices_reuse = compressor_reuse(
        hidden_states, q_residual, position_ids, layer_idx=2
    )
    
    # 验证完全重用
    assert torch.equal(main_kv_reuse, shared_cache.main_kv)
    assert torch.equal(indices_reuse, indices_reindex)
    print("   ✓ CSA2 Reuse Mode 工作正常 (KV 和索引重用验证通过)")
    
except Exception as e:
    print(f"   ✗ CSA2 测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# 测试 CSA2Attention
print("3. 测试 CSA2Attention 层...")
try:
    shared_cache_attn = CSA2SharedCache()
    compressor_attn = CSA2Compressor(
        **config,
        mode=CSA2Mode.FULL,
        shared_cache=shared_cache_attn,
    )
    
    attention = CSA2Attention(
        hidden_size=config["hidden_size"],
        num_attention_heads=config["num_attention_heads"],
        num_key_value_heads=config["num_key_value_heads"],
        head_dim=config["head_dim"],
        q_lora_rank=config["q_lora_rank"],
        compressor=compressor_attn,
    )
    
    output = attention(hidden_states, q_residual, position_ids, layer_idx=0)
    assert output.shape == (batch_size, seq_len, config["hidden_size"])
    
    print("   ✓ CSA2Attention 前向传播成功")
    
    # 测试梯度
    hidden_states_grad = torch.randn(batch_size, seq_len, config["hidden_size"], requires_grad=True)
    q_residual_grad = torch.randn(batch_size, seq_len, config["q_lora_rank"], requires_grad=True)
    
    output_grad = attention(hidden_states_grad, q_residual_grad, position_ids, layer_idx=0)
    loss = output_grad.sum()
    loss.backward()
    
    assert hidden_states_grad.grad is not None
    assert q_residual_grad.grad is not None
    print("   ✓ CSA2Attention 梯度流正常")
    
except Exception as e:
    print(f"   ✗ CSA2Attention 测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# 测试 Hierarchical Sparse Indexer
print("4. 测试 Hierarchical Sparse Indexer...")
try:
    indexer = HierarchicalSparseIndexer(
        hidden_size=256,
        q_lora_rank=128,
        index_num_heads=4,
        index_head_dim=32,
        top_k=16,
        block_size=4,
        num_candidate_blocks=8,
    )
    
    batch_size, seq_len = 2, 32
    compressed_len = 64
    
    q_residual_idx = torch.randn(batch_size, seq_len, 128)
    indexer_k = torch.randn(batch_size, 1, compressed_len, 32)
    index_kwargs = dict(
        hidden_states=torch.randn(batch_size, seq_len, 256),
        position_ids=torch.arange(seq_len).expand(batch_size, -1),
        key_end_position_ids=torch.arange(compressed_len).expand(batch_size, -1),
    )
    
    # Full Mode
    top_k_indices_full, candidate_pool = indexer.forward_full_mode(q_residual_idx, indexer_k, **index_kwargs)
    
    assert top_k_indices_full.shape == (batch_size, seq_len, 16)
    assert candidate_pool.shape == (batch_size, seq_len, 8 * 4)  # num_blocks * block_size
    
    print("   ✓ Hierarchical Indexer Full Mode 工作正常")
    
    # Reindex Mode
    q_residual_idx_new = torch.randn(batch_size, seq_len, 128)
    top_k_indices_reindex = indexer.forward_reindex_mode(q_residual_idx_new, indexer_k, **index_kwargs)
    
    assert top_k_indices_reindex.shape == (batch_size, seq_len, 16)
    print("   ✓ Hierarchical Indexer Reindex Mode 工作正常")
    
    # 验证候选池大小小于全上下文
    pool_size = candidate_pool.shape[-1]
    assert pool_size < compressed_len
    print(f"   ✓ 候选池大小 ({pool_size}) < 压缩长度 ({compressed_len})")
    
except Exception as e:
    print(f"   ✗ Hierarchical Indexer 测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# 测试 Single-Pass mHC
print("5. 测试 Single-Pass mHC...")
try:
    mhc = SinglePassMHC(
        num_branches=4,
        hidden_size=256,
    )
    
    batch_size, seq_len = 2, 32
    hidden_states_mhc = torch.randn(batch_size, seq_len, 256)
    
    # 初始化残差流
    X = mhc.init_state(hidden_states_mhc)
    assert X.shape == (batch_size, seq_len, 4, 256)
    print("   ✓ 残差流初始化成功")
    
    # 前向传播
    Y = hidden_states_mhc
    X_new, X_hat = mhc(X, Y)
    
    assert X_new.shape == (batch_size, seq_len, 4, 256)
    assert X_hat.shape == (batch_size, seq_len, 256)
    print("   ✓ Single-Pass mHC 前向传播成功")
    
    # 验证系数缓存
    assert mhc.prev_A is not None
    print("   ✓ 系数缓存 (A_{l-1}) 工作正常")
    
    # 测试梯度
    X_grad = torch.randn(batch_size, seq_len, 4, 256, requires_grad=True)
    Y_grad = torch.randn(batch_size, seq_len, 256, requires_grad=True)
    
    X_new_grad, X_hat_grad = mhc(X_grad, Y_grad)
    loss_mhc = X_hat_grad.sum()
    loss_mhc.backward()
    
    assert X_grad.grad is not None
    assert Y_grad.grad is not None
    print("   ✓ Single-Pass mHC 梯度流正常")
    
except Exception as e:
    print(f"   ✗ Single-Pass mHC 测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# 集成测试
print("6. 测试跨层集成...")
try:
    # 创建 3 层 CSA2 流水线
    shared_cache_int = CSA2SharedCache()
    
    layers = [
        CSA2Compressor(**config, mode=CSA2Mode.FULL, shared_cache=shared_cache_int),
        CSA2Compressor(**config, mode=CSA2Mode.REINDEX, shared_cache=shared_cache_int),
        CSA2Compressor(**config, mode=CSA2Mode.REUSE, shared_cache=shared_cache_int),
    ]
    
    hidden_int = torch.randn(2, 32, 256)
    q_residuals = [torch.randn(2, 32, 128) for _ in range(3)]
    pos_ids_int = torch.arange(32).unsqueeze(0).expand(2, -1)
    
    kv_list = []
    idx_list = []
    
    for layer_idx, (layer, q_res) in enumerate(zip(layers, q_residuals)):
        kv, idx = layer(hidden_int, q_res, pos_ids_int, layer_idx)
        kv_list.append(kv)
        idx_list.append(idx)
    
    # 验证 KV 重用
    assert torch.equal(kv_list[0], kv_list[1])
    assert torch.equal(kv_list[1], kv_list[2])
    print("   ✓ 跨层 KV 重用验证通过")
    
    # 验证索引更新
    assert torch.equal(idx_list[2], idx_list[1])  # Reuse 使用 Reindex 的索引
    print("   ✓ 跨层索引重用验证通过")
    
except Exception as e:
    print(f"   ✗ 集成测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# 总结
print("=" * 80)
print("✓ 所有组件验证通过!")
print("=" * 80)
print()
print("已验证组件:")
print("  1. CSA2Compressor (Full/Reindex/Reuse 模式)")
print("  2. CSA2Attention (前向和反向传播)")
print("  3. HierarchicalSparseIndexer (候选池构建和搜索)")
print("  4. SinglePassMHC (残差流管理和系数缓存)")
print("  5. 跨层集成 (KV 和索引重用)")
print()
print("本脚本仅验证基本行为；报告对齐结论与范围见 docs/DeepSeek_V4.1_attention_audit.md")
print()
print("下一步:")
print("  - 运行完整测试套件: pytest tests/test_csa2_public_api.py")
print("  - 查看文档: docs/DeepSeek_V4.1_实现文档.md")
print("  - 快速参考: docs/DeepSeek_V4.1_快速参考.md")
print()
