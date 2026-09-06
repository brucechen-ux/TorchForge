"""Tests for GLM-5.3-Flash Dynamic Sparse Attention (DSA) components."""

import pytest
import torch

from torchforge.common.attention.dsa import (
    DSA,
    DSAIndexer,
    GLMDynamicSparseAttention,
    KPool,
)


class TestKPool:
    """Test KPool compression component."""
    
    def test_init_valid_parameters(self):
        """Test initialization with valid parameters."""
        pool = KPool(compress_rate=4, pooling_mode="avg")
        assert pool.compress_rate == 4
        assert pool.pooling_mode == "avg"
    
    def test_init_invalid_compress_rate(self):
        """Test initialization rejects invalid compress_rate."""
        with pytest.raises(ValueError, match="compress_rate must be positive"):
            KPool(compress_rate=0)
    
    def test_init_invalid_pooling_mode(self):
        """Test initialization rejects invalid pooling_mode."""
        with pytest.raises(ValueError, match="pooling_mode must be"):
            KPool(compress_rate=4, pooling_mode="invalid")
    
    def test_avg_pooling_exact_division(self):
        """Test average pooling with sequence divisible by compress_rate."""
        pool = KPool(compress_rate=4, pooling_mode="avg")
        hidden_states = torch.randn(2, 16, 128)  # batch=2, seq=16, hidden=128
        
        compressed = pool(hidden_states)
        
        assert compressed.shape == (2, 4, 128)  # Compressed 16 -> 4
        
        # Verify average pooling: first compressed token should be avg of first 4 tokens
        expected_first = hidden_states[:, :4, :].mean(dim=1)
        torch.testing.assert_close(compressed[:, 0, :], expected_first)
    
    def test_avg_pooling_with_padding(self):
        """Test average pooling with sequence not divisible by compress_rate."""
        pool = KPool(compress_rate=4, pooling_mode="avg")
        hidden_states = torch.randn(2, 15, 128)  # 15 not divisible by 4
        
        compressed = pool(hidden_states)
        
        # Should pad to 16 (next multiple of 4), then compress to 4
        assert compressed.shape == (2, 4, 128)
    
    def test_max_pooling(self):
        """Test max pooling."""
        pool = KPool(compress_rate=4, pooling_mode="max")
        hidden_states = torch.randn(2, 16, 128)
        
        compressed = pool(hidden_states)
        
        assert compressed.shape == (2, 4, 128)
        
        # Verify max pooling: first compressed token should be max of first 4 tokens
        expected_first = hidden_states[:, :4, :].max(dim=1).values
        torch.testing.assert_close(compressed[:, 0, :], expected_first)
    
    def test_learned_weighted_pooling(self):
        """Test learned weighted pooling."""
        pool = KPool(compress_rate=4, pooling_mode="learned_weighted", hidden_size=128)
        hidden_states = torch.randn(2, 16, 128)
        
        compressed = pool(hidden_states)
        
        assert compressed.shape == (2, 4, 128)
        assert pool.pool_weights is not None
        assert pool.pool_weights.shape == (4,)


class TestDSAIndexer:
    """Test DSA sparse indexer."""
    
    def test_init_valid_parameters(self):
        """Test initialization with valid parameters."""
        indexer = DSAIndexer(
            hidden_size=128,
            q_lora_rank=64,
            num_heads=8,
            head_dim=16,
            top_k=32,
        )
        assert indexer.hidden_size == 128
        assert indexer.top_k == 32
    
    def test_forward_shape(self):
        """Test forward pass produces correct output shape."""
        indexer = DSAIndexer(
            hidden_size=128,
            q_lora_rank=64,
            num_heads=8,
            head_dim=16,
            top_k=32,
        )
        
        hidden_states = torch.randn(2, 100, 128)
        compressed_kv = torch.randn(2, 25, 128)  # Compressed by 4x
        q_residual = torch.randn(2, 100, 64)
        
        indices = indexer(hidden_states, compressed_kv, q_residual)
        
        # Should select top_k=32 positions for each of 100 query tokens
        assert indices.shape == (2, 100, 32)
        assert indices.dtype == torch.long
    
    def test_indices_within_valid_range(self):
        """Test that selected indices are within valid range."""
        indexer = DSAIndexer(
            hidden_size=128,
            q_lora_rank=64,
            num_heads=8,
            head_dim=16,
            top_k=16,
        )
        
        hidden_states = torch.randn(2, 50, 128)
        compressed_kv = torch.randn(2, 12, 128)  # 12 compressed positions
        q_residual = torch.randn(2, 50, 64)
        
        indices = indexer(hidden_states, compressed_kv, q_residual)
        
        # Valid indices should be in [0, 12) or -1 for padding
        valid_mask = indices >= 0
        valid_indices = indices[valid_mask]
        assert (valid_indices < 12).all()
        assert (valid_indices >= 0).all()
    
    def test_top_k_larger_than_compressed_length(self):
        """Test behavior when top_k exceeds compressed sequence length."""
        indexer = DSAIndexer(
            hidden_size=128,
            q_lora_rank=64,
            num_heads=8,
            head_dim=16,
            top_k=50,  # Larger than compressed_len
        )
        
        hidden_states = torch.randn(2, 100, 128)
        compressed_kv = torch.randn(2, 20, 128)  # Only 20 compressed positions
        q_residual = torch.randn(2, 100, 64)
        
        indices = indexer(hidden_states, compressed_kv, q_residual)
        
        # Should still return shape (2, 100, 50) with padding
        assert indices.shape == (2, 100, 50)
        
        # First 20 should be valid, rest should be -1
        assert (indices[:, :, :20] >= 0).any()
        assert (indices[:, :, 20:] == -1).all()


class TestGLMDynamicSparseAttention:
    """Test complete DSA module."""
    
    def test_init_valid_parameters(self):
        """Test initialization with valid parameters."""
        dsa = GLMDynamicSparseAttention(
            hidden_size=256,
            num_attention_heads=8,
            num_key_value_heads=4,
            q_lora_rank=128,
            kv_lora_rank=256,
            qk_nope_head_dim=64,
            v_head_dim=64,
            compress_rate=4,
            top_k=64,
        )
        assert dsa.hidden_size == 256
        assert dsa.num_attention_heads == 8
        assert dsa.compress_rate == 4
        assert dsa.top_k == 64
    
    def test_forward_basic(self):
        """Test forward pass with basic inputs."""
        dsa = GLMDynamicSparseAttention(
            hidden_size=256,
            num_attention_heads=8,
            num_key_value_heads=4,
            q_lora_rank=128,
            kv_lora_rank=256,
            qk_nope_head_dim=64,
            v_head_dim=64,
            compress_rate=4,
            top_k=32,
        )
        
        hidden_states = torch.randn(2, 64, 256)
        
        output = dsa(hidden_states)
        
        assert isinstance(output, dict)
        assert "hidden_states" in output
        assert output["hidden_states"].shape == (2, 64, 256)
    
    def test_forward_with_attention_mask(self):
        """Test forward pass with attention mask."""
        dsa = GLMDynamicSparseAttention(
            hidden_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            q_lora_rank=64,
            kv_lora_rank=128,
            qk_nope_head_dim=32,
            v_head_dim=32,
            compress_rate=4,
            top_k=16,
        )
        
        hidden_states = torch.randn(2, 32, 128)
        attention_mask = torch.zeros(2, 1, 32, 32)
        # Causal mask
        for i in range(32):
            attention_mask[:, :, i, :i+1] = 1.0
        
        output = dsa(hidden_states, attention_mask=attention_mask)
        
        assert output["hidden_states"].shape == (2, 32, 128)
    
    def test_output_attentions(self):
        """Test that output_attentions flag works."""
        dsa = GLMDynamicSparseAttention(
            hidden_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            q_lora_rank=64,
            kv_lora_rank=128,
            qk_nope_head_dim=32,
            v_head_dim=32,
        )
        
        hidden_states = torch.randn(2, 32, 128)
        
        output = dsa(hidden_states, output_attentions=True)
        
        assert "attentions" in output
        assert output["attentions"].shape == (2, 4, 32, 32)
    
    def test_gqa_correctness(self):
        """Test GQA (grouped query attention) mechanism."""
        # num_attention_heads must be divisible by num_key_value_heads
        dsa = GLMDynamicSparseAttention(
            hidden_size=128,
            num_attention_heads=8,
            num_key_value_heads=2,  # 4 groups
            q_lora_rank=64,
            kv_lora_rank=128,
            qk_nope_head_dim=32,
            v_head_dim=32,
        )
        
        assert dsa.num_key_value_groups == 4
        
        hidden_states = torch.randn(2, 16, 128)
        output = dsa(hidden_states)
        
        assert output["hidden_states"].shape == (2, 16, 128)
    
    def test_compression_reduces_complexity(self):
        """Test that KPool compression actually reduces sequence length."""
        dsa = GLMDynamicSparseAttention(
            hidden_size=128,
            num_attention_heads=4,
            num_key_value_heads=4,
            q_lora_rank=64,
            kv_lora_rank=128,
            qk_nope_head_dim=32,
            v_head_dim=32,
            compress_rate=4,
        )
        
        # Create input with length 64
        hidden_states = torch.randn(1, 64, 128)
        
        # Verify KPool compresses 64 -> 16
        compressed = dsa.kpool(hidden_states)
        assert compressed.shape == (1, 16, 128)


class TestDSAAlias:
    """Test that DSA alias works."""
    
    def test_alias_is_same_class(self):
        """Test DSA is an alias for GLMDynamicSparseAttention."""
        assert DSA is GLMDynamicSparseAttention


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
