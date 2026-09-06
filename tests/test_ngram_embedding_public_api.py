"""Tests for Qwen3.8-Flash-Next N-gram Embedding components."""

import pytest
import torch

from torchforge.common.embedding.ngram_embedding import (
    NgramEmbedding,
    NgramEmbeddingLayer,
)


class TestNgramEmbedding:
    """Test N-gram embedding lookup table."""
    
    def test_init_valid_parameters(self):
        """Test initialization with valid parameters."""
        ngram = NgramEmbedding(
            vocab_size=50000,
            hidden_size=256,
            max_ngram=3,
            table_size=1000000,
        )
        assert ngram.vocab_size == 50000
        assert ngram.hidden_size == 256
        assert ngram.max_ngram == 3
        assert ngram.table_size == 1000000
    
    def test_init_invalid_parameters(self):
        """Test initialization rejects invalid parameters."""
        with pytest.raises(ValueError, match="vocab_size must be positive"):
            NgramEmbedding(vocab_size=0, hidden_size=256)
        
        with pytest.raises(ValueError, match="max_ngram must be at least 2"):
            NgramEmbedding(vocab_size=50000, hidden_size=256, max_ngram=1)
        
        with pytest.raises(ValueError, match="max_ngram > 4 not supported"):
            NgramEmbedding(vocab_size=50000, hidden_size=256, max_ngram=5)
        
        with pytest.raises(ValueError, match="hash_function must be"):
            NgramEmbedding(
                vocab_size=50000, hidden_size=256, hash_function="invalid"
            )
    
    def test_ngram_tables_created(self):
        """Test that n-gram tables are created for each order."""
        ngram = NgramEmbedding(
            vocab_size=50000,
            hidden_size=256,
            max_ngram=3,
            table_size=1000000,
        )
        
        # Should have tables for 2-gram and 3-gram
        assert "2" in ngram.ngram_tables
        assert "3" in ngram.ngram_tables
        
        # Each table should have correct dimensions
        assert ngram.ngram_tables["2"].weight.shape == (1000000, 256)
        assert ngram.ngram_tables["3"].weight.shape == (1000000, 256)
    
    def test_ngram_weights_learnable(self):
        """Test that n-gram combination weights are learnable."""
        ngram = NgramEmbedding(
            vocab_size=50000,
            hidden_size=256,
            max_ngram=3,
        )
        
        # Should have learnable weights for combining 2-gram and 3-gram
        assert ngram.ngram_weights is not None
        assert ngram.ngram_weights.shape == (2,)  # max_ngram - 1
        assert ngram.ngram_weights.requires_grad
    
    def test_forward_shape(self):
        """Test forward pass produces correct output shape."""
        ngram = NgramEmbedding(
            vocab_size=50000,
            hidden_size=256,
            max_ngram=3,
            table_size=100000,
        )
        
        input_ids = torch.randint(0, 50000, (2, 32))
        
        output = ngram(input_ids, return_dict=False)
        
        # Should return (batch, seq_len, hidden_size)
        assert output.shape == (2, 32, 256)
    
    def test_forward_return_dict(self):
        """Test forward with return_dict=True."""
        ngram = NgramEmbedding(
            vocab_size=50000,
            hidden_size=256,
            max_ngram=3,
            table_size=100000,
        )
        
        input_ids = torch.randint(0, 50000, (2, 32))
        
        output = ngram(input_ids, return_dict=True)
        
        assert isinstance(output, dict)
        assert "ngram_embedding" in output
        assert "components" in output
        
        # Should have components for each n-gram order
        assert "2gram" in output["components"]
        assert "3gram" in output["components"]
        
        assert output["ngram_embedding"].shape == (2, 32, 256)
    
    def test_hash_function_simple(self):
        """Test simple polynomial hash function."""
        ngram = NgramEmbedding(
            vocab_size=50000,
            hidden_size=256,
            max_ngram=2,
            table_size=100000,
            hash_function="simple",
        )
        
        input_ids = torch.randint(0, 50000, (2, 16))
        output = ngram(input_ids, return_dict=False)
        
        assert output.shape == (2, 16, 256)
    
    def test_hash_function_murmur(self):
        """Test MurmurHash-inspired hash function."""
        ngram = NgramEmbedding(
            vocab_size=50000,
            hidden_size=256,
            max_ngram=2,
            table_size=100000,
            hash_function="murmur",
        )
        
        input_ids = torch.randint(0, 50000, (2, 16))
        output = ngram(input_ids, return_dict=False)
        
        assert output.shape == (2, 16, 256)
    
    def test_deterministic_lookup(self):
        """Test that same input produces same output (deterministic)."""
        ngram = NgramEmbedding(
            vocab_size=50000,
            hidden_size=256,
            max_ngram=3,
            table_size=100000,
        )
        
        input_ids = torch.randint(0, 50000, (2, 32))
        
        output1 = ngram(input_ids, return_dict=False)
        output2 = ngram(input_ids, return_dict=False)
        
        torch.testing.assert_close(output1, output2)
    
    def test_different_inputs_produce_different_outputs(self):
        """Test that different inputs produce different embeddings."""
        ngram = NgramEmbedding(
            vocab_size=50000,
            hidden_size=256,
            max_ngram=3,
            table_size=100000,
        )
        
        input_ids1 = torch.randint(0, 50000, (2, 32))
        input_ids2 = torch.randint(0, 50000, (2, 32))
        
        output1 = ngram(input_ids1, return_dict=False)
        output2 = ngram(input_ids2, return_dict=False)
        
        # Should produce different embeddings (with high probability)
        assert not torch.allclose(output1, output2)
    
    def test_max_ngram_orders(self):
        """Test different max_ngram values."""
        for max_n in [2, 3, 4]:
            ngram = NgramEmbedding(
                vocab_size=50000,
                hidden_size=256,
                max_ngram=max_n,
                table_size=100000,
            )
            
            input_ids = torch.randint(0, 50000, (2, 32))
            output = ngram(input_ids, return_dict=True)
            
            # Should have components for 2-gram through max_ngram
            expected_orders = [f"{n}gram" for n in range(2, max_n + 1)]
            for order in expected_orders:
                assert order in output["components"]
    
    def test_valid_mask_for_boundary_tokens(self):
        """Test that boundary tokens without full context are handled."""
        ngram = NgramEmbedding(
            vocab_size=50000,
            hidden_size=256,
            max_ngram=3,  # Needs 2 previous tokens
            table_size=100000,
        )
        
        input_ids = torch.randint(0, 50000, (2, 10))
        output = ngram(input_ids, return_dict=True)
        
        # First few tokens don't have full 3-gram context
        # but should still produce valid embeddings (masked)
        assert output["ngram_embedding"].shape == (2, 10, 256)
        
        # Check that first token has less contribution from 3-gram
        # (it only has itself, no previous context)
        trigram_contrib = output["components"]["3gram"]
        
        # First position should have near-zero 3-gram contribution
        first_pos_norm = trigram_contrib[:, 0, :].abs().mean()
        middle_pos_norm = trigram_contrib[:, 5, :].abs().mean()
        
        # Middle positions should have more contribution
        assert first_pos_norm <= middle_pos_norm
    
    def test_dropout_in_training_mode(self):
        """Test that dropout is applied in training mode."""
        ngram = NgramEmbedding(
            vocab_size=50000,
            hidden_size=256,
            max_ngram=3,
            table_size=100000,
            dropout=0.5,  # High dropout for testing
        )
        
        ngram.train()
        input_ids = torch.randint(0, 50000, (2, 32))
        
        # Multiple forward passes should produce different results due to dropout
        output1 = ngram(input_ids, return_dict=False)
        output2 = ngram(input_ids, return_dict=False)
        
        assert not torch.allclose(output1, output2)
    
    def test_no_dropout_in_eval_mode(self):
        """Test that dropout is not applied in eval mode."""
        ngram = NgramEmbedding(
            vocab_size=50000,
            hidden_size=256,
            max_ngram=3,
            table_size=100000,
            dropout=0.5,
        )
        
        ngram.eval()
        input_ids = torch.randint(0, 50000, (2, 32))
        
        # Multiple forward passes should produce same results (no dropout)
        output1 = ngram(input_ids, return_dict=False)
        output2 = ngram(input_ids, return_dict=False)
        
        torch.testing.assert_close(output1, output2)


class TestNgramEmbeddingLayer:
    """Test N-gram embedding injection layer."""
    
    def test_init_valid_parameters(self):
        """Test initialization with valid parameters."""
        layer = NgramEmbeddingLayer(
            vocab_size=50000,
            hidden_size=256,
            max_ngram=3,
            table_size=1000000,
            injection_mode="add",
        )
        assert layer.vocab_size == 50000
        assert layer.hidden_size == 256
        assert layer.injection_mode == "add"
    
    def test_init_invalid_injection_mode(self):
        """Test initialization rejects invalid injection_mode."""
        with pytest.raises(ValueError, match="injection_mode must be"):
            NgramEmbeddingLayer(
                vocab_size=50000,
                hidden_size=256,
                injection_mode="invalid",
            )
    
    def test_forward_add_mode(self):
        """Test forward with add injection mode."""
        layer = NgramEmbeddingLayer(
            vocab_size=50000,
            hidden_size=256,
            max_ngram=3,
            table_size=100000,
            injection_mode="add",
        )
        
        hidden_states = torch.randn(2, 32, 256)
        input_ids = torch.randint(0, 50000, (2, 32))
        
        output = layer(hidden_states, input_ids)
        
        # Should return same shape
        assert output.shape == (2, 32, 256)
        
        # Should be different from input (added n-gram embeddings)
        assert not torch.allclose(output, hidden_states)
    
    def test_forward_concat_project_mode(self):
        """Test forward with concat_project injection mode."""
        layer = NgramEmbeddingLayer(
            vocab_size=50000,
            hidden_size=256,
            max_ngram=3,
            table_size=100000,
            injection_mode="concat_project",
        )
        
        hidden_states = torch.randn(2, 32, 256)
        input_ids = torch.randint(0, 50000, (2, 32))
        
        output = layer(hidden_states, input_ids)
        
        # Should return same shape (projected back to hidden_size)
        assert output.shape == (2, 32, 256)
        
        # Check that projection layer exists
        assert layer.projection is not None
    
    def test_add_mode_no_projection(self):
        """Test that add mode does not create projection layer."""
        layer = NgramEmbeddingLayer(
            vocab_size=50000,
            hidden_size=256,
            injection_mode="add",
        )
        
        assert layer.projection is None
    
    def test_concat_mode_has_projection(self):
        """Test that concat_project mode creates projection layer."""
        layer = NgramEmbeddingLayer(
            vocab_size=50000,
            hidden_size=256,
            injection_mode="concat_project",
        )
        
        assert layer.projection is not None
        # Projection should be 2*hidden_size -> hidden_size
        assert layer.projection.in_features == 512
        assert layer.projection.out_features == 256
    
    def test_layer_2_injection_qwen_config(self):
        """Test typical Qwen Layer 2 injection scenario."""
        # Qwen injects n-gram at Layer 2
        layer = NgramEmbeddingLayer(
            vocab_size=248320,  # Qwen vocab size
            hidden_size=2560,   # Qwen hidden size
            max_ngram=3,
            table_size=20_000_000,  # Qwen table size
            injection_mode="add",
        )
        
        # Simulate Layer 2 hidden states
        hidden_states = torch.randn(2, 128, 2560)
        input_ids = torch.randint(0, 248320, (2, 128))
        
        output = layer(hidden_states, input_ids)
        
        assert output.shape == (2, 128, 2560)


class TestNgramEmbeddingCapacityScaling:
    """Test N-gram embedding as capacity scaling mechanism."""
    
    def test_parameter_count_scaling(self):
        """Test that table_size directly controls parameter count."""
        small_ngram = NgramEmbedding(
            vocab_size=50000,
            hidden_size=256,
            max_ngram=3,
            table_size=1_000_000,
        )
        
        large_ngram = NgramEmbedding(
            vocab_size=50000,
            hidden_size=256,
            max_ngram=3,
            table_size=10_000_000,  # 10x larger
        )
        
        small_params = sum(p.numel() for p in small_ngram.parameters())
        large_params = sum(p.numel() for p in large_ngram.parameters())
        
        # Large should have ~10x more parameters
        ratio = large_params / small_params
        assert 9.0 < ratio < 11.0  # Approximately 10x
    
    def test_bigram_vs_trigram_capacity(self):
        """Test capacity difference between bigram and trigram."""
        bigram = NgramEmbedding(
            vocab_size=50000,
            hidden_size=256,
            max_ngram=2,  # Only bigram
            table_size=1_000_000,
        )
        
        trigram = NgramEmbedding(
            vocab_size=50000,
            hidden_size=256,
            max_ngram=3,  # Bigram + trigram
            table_size=1_000_000,
        )
        
        bigram_params = sum(p.numel() for p in bigram.parameters())
        trigram_params = sum(p.numel() for p in trigram.parameters())
        
        # Trigram should have more capacity (2 tables vs 1 table)
        assert trigram_params > bigram_params
    
    def test_lookup_sparse_no_extra_compute(self):
        """Test that n-gram lookup adds minimal compute (mostly memory)."""
        ngram = NgramEmbedding(
            vocab_size=50000,
            hidden_size=256,
            max_ngram=3,
            table_size=1_000_000,
        )
        
        input_ids = torch.randint(0, 50000, (2, 32))
        
        # Forward should be fast (just lookup + addition)
        import time
        start = time.time()
        for _ in range(100):
            _ = ngram(input_ids, return_dict=False)
        elapsed = time.time() - start
        
        # Should be very fast (< 1 second for 100 iterations)
        assert elapsed < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
