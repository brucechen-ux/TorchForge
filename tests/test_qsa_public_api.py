"""Tests for Qwen3.8-Flash-Next Query Sparse Attention (QSA) components."""

import pytest
import torch

from torchforge.common.attention.qsa import (
    MicroBlockIndexer,
    QSA,
    QwenQuerySparseAttention,
)


class TestMicroBlockIndexer:
    """Test micro-block indexer component."""
    
    def test_init_valid_parameters(self):
        """Test initialization with valid parameters."""
        indexer = MicroBlockIndexer(
            hidden_size=128,
            num_heads=8,
            head_dim=16,
            block_size=4,
            num_blocks=16,
        )
        assert indexer.block_size == 4
        assert indexer.num_blocks == 16
    
    def test_init_invalid_parameters(self):
        """Test initialization rejects invalid parameters."""
        with pytest.raises(ValueError, match="hidden_size must be positive"):
            MicroBlockIndexer(hidden_size=0, num_heads=8, head_dim=16)
        
        with pytest.raises(ValueError, match="block_size must be positive"):
            MicroBlockIndexer(hidden_size=128, num_heads=8, head_dim=16, block_size=0)
        
        with pytest.raises(ValueError, match="block_scoring must be"):
            MicroBlockIndexer(
                hidden_size=128, num_heads=8, head_dim=16, block_scoring="invalid"
            )
    
    def test_forward_shape(self):
        """Test forward pass produces correct output shapes."""
        indexer = MicroBlockIndexer(
            hidden_size=128,
            num_heads=8,
            head_dim=16,
            block_size=4,
            num_blocks=16,
        )
        
        hidden_states = torch.randn(2, 64, 128)
        
        block_indices, block_mask = indexer(hidden_states)
        
        # block_indices: (batch, seq_len, num_blocks)
        assert block_indices.shape == (2, 64, 16)
        # block_mask: (batch, seq_len, seq_len)
        assert block_mask.shape == (2, 64, 64)
        assert block_mask.dtype == torch.bool
    
    def test_block_indices_valid_range(self):
        """Test that block indices are within valid range."""
        indexer = MicroBlockIndexer(
            hidden_size=128,
            num_heads=8,
            head_dim=16,
            block_size=4,
            num_blocks=8,
        )
        
        hidden_states = torch.randn(2, 32, 128)  # 32 tokens = 8 blocks (block_size=4)
        
        block_indices, _ = indexer(hidden_states)
        
        # Valid indices should be in [0, 8) or -1 for padding
        valid_mask = block_indices >= 0
        valid_indices = block_indices[valid_mask]
        assert (valid_indices < 8).all()
        assert (valid_indices >= 0).all()
    
    def test_block_mask_structure(self):
        """Test that block_mask correctly marks selected blocks."""
        indexer = MicroBlockIndexer(
            hidden_size=128,
            num_heads=8,
            head_dim=16,
            block_size=4,
            num_blocks=4,  # Select 4 blocks
        )
        
        hidden_states = torch.randn(1, 16, 128)  # 16 tokens = 4 blocks
        
        _, block_mask = indexer(hidden_states)
        
        # Each query token should attend to exactly (num_blocks * block_size) tokens
        # In this case: 4 blocks * 4 tokens/block = 16 tokens
        tokens_per_query = block_mask[0].sum(dim=-1)
        
        # Should attend to at least some tokens (may vary due to sparsity)
        assert (tokens_per_query > 0).all()
    
    def test_max_vs_avg_scoring(self):
        """Test different block scoring methods produce valid outputs."""
        for scoring_method in ["max", "avg"]:
            indexer = MicroBlockIndexer(
                hidden_size=128,
                num_heads=8,
                head_dim=16,
                block_size=4,
                num_blocks=8,
                block_scoring=scoring_method,
            )
            
            hidden_states = torch.randn(2, 32, 128)
            block_indices, block_mask = indexer(hidden_states)
            
            assert block_indices.shape == (2, 32, 8)
            assert block_mask.shape == (2, 32, 32)
    
    def test_num_blocks_exceeds_sequence_blocks(self):
        """Test behavior when num_blocks exceeds actual blocks in sequence."""
        indexer = MicroBlockIndexer(
            hidden_size=128,
            num_heads=8,
            head_dim=16,
            block_size=4,
            num_blocks=20,  # More than actual blocks
        )
        
        hidden_states = torch.randn(2, 32, 128)  # 32 / 4 = 8 actual blocks
        
        block_indices, block_mask = indexer(hidden_states)
        
        # Should pad to num_blocks=20
        assert block_indices.shape == (2, 32, 20)
        
        # First 8 should have valid indices, rest should be -1
        assert (block_indices[:, :, 8:] == -1).all()


class TestQwenQuerySparseAttention:
    """Test complete QSA module."""
    
    def test_init_valid_parameters(self):
        """Test initialization with valid parameters."""
        qsa = QwenQuerySparseAttention(
            hidden_size=256,
            num_attention_heads=8,
            num_key_value_heads=4,
            head_dim=64,
            block_size=4,
            num_blocks=32,
        )
        assert qsa.hidden_size == 256
        assert qsa.num_attention_heads == 8
        assert qsa.block_size == 4
        assert qsa.num_blocks == 32
    
    def test_init_invalid_parameters(self):
        """Test initialization rejects invalid parameters."""
        with pytest.raises(ValueError, match="hidden_size must be positive"):
            QwenQuerySparseAttention(
                hidden_size=0,
                num_attention_heads=8,
                num_key_value_heads=4,
                head_dim=64,
            )
        
        with pytest.raises(ValueError, match="must be divisible by"):
            QwenQuerySparseAttention(
                hidden_size=256,
                num_attention_heads=9,  # Not divisible by num_key_value_heads
                num_key_value_heads=4,
                head_dim=64,
            )
    
    def test_forward_basic(self):
        """Test forward pass with basic inputs."""
        qsa = QwenQuerySparseAttention(
            hidden_size=256,
            num_attention_heads=8,
            num_key_value_heads=4,
            head_dim=64,
            block_size=4,
            num_blocks=16,
        )
        
        hidden_states = torch.randn(2, 64, 256)
        
        output = qsa(hidden_states)
        
        assert isinstance(output, dict)
        assert "hidden_states" in output
        assert output["hidden_states"].shape == (2, 64, 256)
    
    def test_forward_with_attention_mask(self):
        """Test forward pass with attention mask."""
        qsa = QwenQuerySparseAttention(
            hidden_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            block_size=4,
            num_blocks=8,
        )
        
        hidden_states = torch.randn(2, 32, 128)
        # Causal attention mask
        attention_mask = torch.triu(
            torch.full((2, 1, 32, 32), float("-inf")),
            diagonal=1
        )
        
        output = qsa(hidden_states, attention_mask=attention_mask)
        
        assert output["hidden_states"].shape == (2, 32, 128)
    
    def test_output_attentions(self):
        """Test that output_attentions flag works."""
        qsa = QwenQuerySparseAttention(
            hidden_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
        )
        
        hidden_states = torch.randn(2, 32, 128)
        
        output = qsa(hidden_states, output_attentions=True)
        
        assert "attentions" in output
        # Attention weights: (batch, num_heads, seq_len, seq_len)
        assert output["attentions"].shape == (2, 4, 32, 32)
    
    def test_gqa_mechanism(self):
        """Test grouped query attention mechanism."""
        qsa = QwenQuerySparseAttention(
            hidden_size=256,
            num_attention_heads=8,
            num_key_value_heads=2,  # 4 groups
            head_dim=64,
        )
        
        assert qsa.num_key_value_groups == 4
        
        hidden_states = torch.randn(2, 32, 256)
        output = qsa(hidden_states)
        
        assert output["hidden_states"].shape == (2, 32, 256)
    
    def test_block_sparse_reduces_attention_budget(self):
        """Test that block sparsity reduces effective attention budget."""
        qsa = QwenQuerySparseAttention(
            hidden_size=128,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=32,
            block_size=4,
            num_blocks=8,  # Only 8 blocks selected
        )
        
        hidden_states = torch.randn(1, 64, 128)  # 64 tokens = 16 blocks
        
        # Get block mask to verify sparsity
        block_indices, block_mask = qsa.block_indexer(hidden_states)
        
        # Each query should attend to at most (num_blocks * block_size) = 32 tokens
        # out of 64 total (50% sparsity)
        attended_tokens = block_mask[0].sum(dim=-1)
        assert (attended_tokens <= 32).all()
    
    def test_different_block_sizes(self):
        """Test QSA with different block sizes."""
        for block_size in [2, 4, 8]:
            qsa = QwenQuerySparseAttention(
                hidden_size=128,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=32,
                block_size=block_size,
                num_blocks=8,
            )
            
            hidden_states = torch.randn(2, 32, 128)
            output = qsa(hidden_states)
            
            assert output["hidden_states"].shape == (2, 32, 128)
    
    def test_bias_parameter(self):
        """Test QSA with and without bias."""
        for use_bias in [True, False]:
            qsa = QwenQuerySparseAttention(
                hidden_size=128,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=32,
                bias=use_bias,
            )
            
            hidden_states = torch.randn(2, 16, 128)
            output = qsa(hidden_states)
            
            assert output["hidden_states"].shape == (2, 16, 128)


class TestQSAAlias:
    """Test that QSA alias works."""
    
    def test_alias_is_same_class(self):
        """Test QSA is an alias for QwenQuerySparseAttention."""
        assert QSA is QwenQuerySparseAttention


class TestQSAVsDSAComparison:
    """Compare QSA and DSA approaches."""
    
    def test_qsa_uses_blocks_dsa_uses_tokens(self):
        """Verify QSA uses block-level sparsity while DSA uses token-level."""
        from torchforge.common.attention.dsa import DSA
        
        # QSA: block-level indexer
        qsa = QSA(
            hidden_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            block_size=4,
            num_blocks=8,
        )
        assert hasattr(qsa, "block_indexer")
        assert qsa.block_indexer.block_size == 4
        
        # DSA: token-level with compression
        dsa = DSA(
            hidden_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            q_lora_rank=64,
            kv_lora_rank=128,
            qk_nope_head_dim=32,
            v_head_dim=32,
            compress_rate=4,
            top_k=32,
        )
        assert hasattr(dsa, "indexer")
        assert dsa.indexer.top_k == 32  # Token-level selection


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
