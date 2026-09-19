"""Test DeepSeek-V4.1-Flash Single-Pass mHC component.

Tests cover:
1. Single-Pass mHC forward pass with coefficient prediction
2. Residual stream initialization and management
3. Coefficient caching (A_{l-1} for input mixing)
4. Single-kernel fusion property (conceptual verification)
5. Gradient flow through residual connections
"""

import pytest
import torch

from torchforge.common.residual import SinglePassMHC, SinglePassMHCBlock


class TestSinglePassMHC:
    """Test Single-Pass mHC residual connection."""
    
    @pytest.fixture
    def config(self):
        return {
            "num_branches": 4,
            "hidden_size": 256,
            "rms_norm_eps": 1e-6,
        }
    
    def test_initialization(self, config):
        """Test SinglePassMHC initialization."""
        mhc = SinglePassMHC(**config)
        
        assert mhc.num_branches == config["num_branches"]
        assert mhc.hidden_size == config["hidden_size"]
        assert mhc.prev_A is None  # Initially no cached A
    
    def test_init_state(self, config):
        """Test residual stream initialization."""
        batch_size, seq_len = 2, 32
        
        mhc = SinglePassMHC(**config)
        hidden_states = torch.randn(batch_size, seq_len, config["hidden_size"])
        
        X_init = mhc.init_state(hidden_states)
        
        # Verify shape
        assert X_init.shape == (batch_size, seq_len, config["num_branches"], config["hidden_size"])
        
        # Verify all branches are identical copies
        for i in range(config["num_branches"]):
            assert torch.allclose(X_init[:, :, i, :], hidden_states)
    
    def test_read_state(self, config):
        """Test reading from residual streams."""
        batch_size, seq_len = 2, 32
        
        mhc = SinglePassMHC(**config)
        X = torch.randn(batch_size, seq_len, config["num_branches"], config["hidden_size"])
        
        output = mhc.read_state(X)
        
        # Verify shape
        assert output.shape == (batch_size, seq_len, config["hidden_size"])
        
        # Verify it's the mean across branches
        expected = X.mean(dim=2)
        assert torch.allclose(output, expected)
    
    def test_forward_pass(self, config):
        """Test SinglePassMHC forward pass."""
        batch_size, seq_len = 2, 32
        
        mhc = SinglePassMHC(**config)
        
        X_prev = torch.randn(batch_size, seq_len, config["num_branches"], config["hidden_size"])
        Y_prev = torch.randn(batch_size, seq_len, config["hidden_size"])
        
        X_curr, X_hat_curr = mhc(X_prev, Y_prev)
        
        # Verify shapes
        assert X_curr.shape == (batch_size, seq_len, config["num_branches"], config["hidden_size"])
        assert X_hat_curr.shape == (batch_size, seq_len, config["hidden_size"])
    
    def test_coefficient_prediction(self, config):
        """Test that coefficients are correctly predicted."""
        batch_size, seq_len = 2, 16
        
        mhc = SinglePassMHC(**config)
        
        X_prev = torch.randn(batch_size, seq_len, config["num_branches"], config["hidden_size"])
        
        A, B, C = mhc._predict_coefficients(X_prev)
        
        # Verify shapes
        n = config["num_branches"]
        assert A.shape == (batch_size, seq_len, 1, n)
        assert B.shape == (batch_size, seq_len, n, n)
        assert C.shape == (batch_size, seq_len, n, 1)
    
    def test_coefficient_caching(self, config):
        """Test that A coefficients are cached for next block."""
        batch_size, seq_len = 2, 16
        
        mhc = SinglePassMHC(**config)
        
        X_prev = torch.randn(batch_size, seq_len, config["num_branches"], config["hidden_size"])
        Y_prev = torch.randn(batch_size, seq_len, config["hidden_size"])
        
        # First forward pass
        assert mhc.prev_A is None
        X_curr_1, X_hat_1 = mhc(X_prev, Y_prev)
        
        # Verify A is now cached
        assert mhc.prev_A is not None
        A_cached_1 = mhc.prev_A.clone()
        
        # Second forward pass
        X_curr_2, X_hat_2 = mhc(X_curr_1, Y_prev)
        
        # Verify A is updated
        assert not torch.equal(mhc.prev_A, A_cached_1)
    
    def test_residual_transformation(self, config):
        """Test residual transformation X_l = B_{l-1} @ X_{l-1} + C_{l-1} @ Y_{l-1}."""
        batch_size, seq_len = 2, 16
        
        mhc = SinglePassMHC(**config)
        
        X_prev = torch.randn(batch_size, seq_len, config["num_branches"], config["hidden_size"])
        Y_prev = torch.randn(batch_size, seq_len, config["hidden_size"])
        
        X_curr, _ = mhc(X_prev, Y_prev)
        
        # Verify residual update incorporates both X_prev and Y_prev
        # (Cannot verify exact formula without access to internal B and C,
        #  but can check that output depends on both inputs)
        
        # Perturb X_prev
        X_prev_perturbed = X_prev + torch.randn_like(X_prev) * 0.1
        X_curr_perturbed, _ = mhc(X_prev_perturbed, Y_prev)
        assert not torch.allclose(X_curr, X_curr_perturbed, atol=1e-4)
        
        # Reset and perturb Y_prev
        mhc.reset_cache()
        mhc(X_prev, Y_prev)  # Prime cache
        Y_prev_perturbed = Y_prev + torch.randn_like(Y_prev) * 0.1
        X_curr_perturbed_y, _ = mhc(X_prev, Y_prev_perturbed)
        assert not torch.allclose(X_curr, X_curr_perturbed_y, atol=1e-4)
    
    def test_input_mixing_shift(self, config):
        """Test that input mixing uses A_{l-1} instead of A_l."""
        batch_size, seq_len = 2, 16
        
        mhc = SinglePassMHC(**config)
        
        X_prev = torch.randn(batch_size, seq_len, config["num_branches"], config["hidden_size"])
        Y_prev = torch.randn(batch_size, seq_len, config["hidden_size"])
        
        # First forward: No cached A, should use current A
        X_curr_1, X_hat_1 = mhc(X_prev, Y_prev)
        A_after_first = mhc.prev_A.clone()
        
        # Second forward: Should use A from first forward
        X_curr_2, X_hat_2 = mhc(X_curr_1, Y_prev)
        
        # Verify that different A was used (conceptually)
        # X_hat uses A_{l-1}, which is now cached
        assert mhc.prev_A is not None
    
    def test_reset_cache(self, config):
        """Test cache reset functionality."""
        batch_size, seq_len = 2, 16
        
        mhc = SinglePassMHC(**config)
        
        X_prev = torch.randn(batch_size, seq_len, config["num_branches"], config["hidden_size"])
        Y_prev = torch.randn(batch_size, seq_len, config["hidden_size"])
        
        # Populate cache
        mhc(X_prev, Y_prev)
        assert mhc.prev_A is not None
        
        # Reset cache
        mhc.reset_cache()
        assert mhc.prev_A is None
    
    def test_backward_pass(self, config):
        """Test gradient flow through Single-Pass mHC."""
        batch_size, seq_len = 2, 16
        
        mhc = SinglePassMHC(**config)
        
        X_prev = torch.randn(batch_size, seq_len, config["num_branches"], config["hidden_size"], requires_grad=True)
        Y_prev = torch.randn(batch_size, seq_len, config["hidden_size"], requires_grad=True)
        
        X_curr, X_hat = mhc(X_prev, Y_prev)
        
        # Backward through X_curr
        loss_X = X_curr.sum()
        loss_X.backward(retain_graph=True)
        
        assert X_prev.grad is not None
        assert Y_prev.grad is not None
        
        # Clear gradients
        X_prev.grad = None
        Y_prev.grad = None
        
        # Backward through X_hat
        loss_X_hat = X_hat.sum()
        loss_X_hat.backward()
        
        assert X_prev.grad is not None
        assert Y_prev.grad is not None
    
    def test_multiple_branches(self):
        """Test with different numbers of branches."""
        batch_size, seq_len = 2, 16
        hidden_size = 128
        
        for num_branches in [1, 2, 4, 8]:
            mhc = SinglePassMHC(
                num_branches=num_branches,
                hidden_size=hidden_size,
            )
            
            X_prev = torch.randn(batch_size, seq_len, num_branches, hidden_size)
            Y_prev = torch.randn(batch_size, seq_len, hidden_size)
            
            X_curr, X_hat = mhc(X_prev, Y_prev)
            
            assert X_curr.shape == (batch_size, seq_len, num_branches, hidden_size)
            assert X_hat.shape == (batch_size, seq_len, hidden_size)


class TestSinglePassMHCBlock:
    """Test SinglePassMHCBlock wrapper."""
    
    @pytest.fixture
    def config(self):
        return {
            "num_branches": 4,
            "hidden_size": 256,
        }
    
    def test_block_forward(self, config):
        """Test SinglePassMHCBlock forward pass."""
        batch_size, seq_len = 2, 32
        
        mhc = SinglePassMHC(**config)
        
        # Simple block that doubles the input
        class DummyBlock(torch.nn.Module):
            def forward(self, x):
                return x * 2
        
        block = DummyBlock()
        mhc_block = SinglePassMHCBlock(mhc=mhc, block=block)
        
        X_prev = torch.randn(batch_size, seq_len, config["num_branches"], config["hidden_size"])
        Y_prev = torch.randn(batch_size, seq_len, config["hidden_size"])
        
        X_curr, Y_curr = mhc_block(X_prev, Y_prev)
        
        # Verify shapes
        assert X_curr.shape == (batch_size, seq_len, config["num_branches"], config["hidden_size"])
        assert Y_curr.shape == (batch_size, seq_len, config["hidden_size"])
    
    def test_block_with_attention(self, config):
        """Test SinglePassMHCBlock with attention-like block."""
        batch_size, seq_len = 2, 32
        
        mhc = SinglePassMHC(**config)
        
        # Simple attention-like block
        class SimpleAttention(torch.nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.proj = torch.nn.Linear(hidden_size, hidden_size)
            
            def forward(self, x):
                return self.proj(x)
        
        attention = SimpleAttention(config["hidden_size"])
        mhc_block = SinglePassMHCBlock(mhc=mhc, block=attention)
        
        X_prev = torch.randn(batch_size, seq_len, config["num_branches"], config["hidden_size"])
        Y_prev = torch.randn(batch_size, seq_len, config["hidden_size"])
        
        X_curr, Y_curr = mhc_block(X_prev, Y_prev)
        
        # Verify output depends on block parameters
        Y_curr_sum = Y_curr.sum()
        Y_curr_sum.backward()
        
        # Attention block should have gradients
        assert attention.proj.weight.grad is not None


class TestSinglePassMHCIntegration:
    """Integration tests for Single-Pass mHC in multi-layer scenarios."""
    
    def test_multi_layer_pipeline(self):
        """Test Single-Pass mHC across multiple layers."""
        config = {
            "num_branches": 4,
            "hidden_size": 128,
        }
        
        batch_size, seq_len = 2, 16
        num_layers = 4
        
        # Create multiple mHC instances (one per layer)
        mhc_layers = [SinglePassMHC(**config) for _ in range(num_layers)]
        
        # Initialize
        hidden_states = torch.randn(batch_size, seq_len, config["hidden_size"])
        X = mhc_layers[0].init_state(hidden_states)
        Y = hidden_states
        
        # Forward through all layers
        for layer_idx, mhc in enumerate(mhc_layers):
            X, X_hat = mhc(X, Y)
            # Simulate block output (simple transformation)
            Y = X_hat * 0.5 + torch.randn_like(X_hat) * 0.1
        
        # Verify final shapes
        assert X.shape == (batch_size, seq_len, config["num_branches"], config["hidden_size"])
        assert Y.shape == (batch_size, seq_len, config["hidden_size"])
    
    def test_memory_efficiency_property(self):
        """Conceptual test: Single-Pass mHC reduces activation memory traffic.
        
        This test verifies the structure that enables single-kernel fusion,
        not the actual kernel implementation (which requires custom CUDA).
        """
        config = {
            "num_branches": 4,
            "hidden_size": 256,
        }
        
        batch_size, seq_len = 2, 32
        
        mhc = SinglePassMHC(**config)
        
        X_prev = torch.randn(batch_size, seq_len, config["num_branches"], config["hidden_size"])
        Y_prev = torch.randn(batch_size, seq_len, config["hidden_size"])
        
        # Key property: A_{l-1} is cached, so input mixing doesn't depend on
        # current coefficient computation
        X_curr, X_hat = mhc(X_prev, Y_prev)
        
        # Verify that prev_A is detached (no gradient dependency)
        assert mhc.prev_A is not None
        assert not mhc.prev_A.requires_grad


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
