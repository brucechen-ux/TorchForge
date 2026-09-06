"""Tests for Qwen3.8-Flash-Next Gated Residual components."""

import pytest
import torch

from torchforge.common.residual.gated_residual import (
    GatedResidual,
    QwenGatedResidual,
)


class TestQwenGatedResidual:
    """Test Qwen Gated Residual component."""
    
    def test_init_valid_parameters(self):
        """Test initialization with valid parameters."""
        gr = QwenGatedResidual(
            hidden_size=256,
            num_branches=4,
            bottleneck_rank=128,
        )
        assert gr.hidden_size == 256
        assert gr.num_branches == 4
        assert gr.bottleneck_rank == 128
    
    def test_init_invalid_parameters(self):
        """Test initialization rejects invalid parameters."""
        with pytest.raises(ValueError, match="hidden_size must be positive"):
            QwenGatedResidual(hidden_size=0)
        
        with pytest.raises(ValueError, match="num_branches must be positive"):
            QwenGatedResidual(hidden_size=256, num_branches=0)
        
        with pytest.raises(ValueError, match="bottleneck_rank must be positive"):
            QwenGatedResidual(hidden_size=256, bottleneck_rank=0)
    
    def test_init_state_shape(self):
        """Test init_state creates correct residual state shape."""
        gr = QwenGatedResidual(hidden_size=256, num_branches=4)
        hidden_states = torch.randn(2, 32, 256)
        
        residual_state = gr.init_state(hidden_states)
        
        # Should expand to (batch, seq_len, num_branches, hidden_size)
        assert residual_state.shape == (2, 32, 4, 256)
    
    def test_init_state_replicates_across_branches(self):
        """Test that init_state replicates input across branches."""
        gr = QwenGatedResidual(hidden_size=256, num_branches=4)
        hidden_states = torch.randn(2, 32, 256)
        
        residual_state = gr.init_state(hidden_states)
        
        # All branches should start with the same values
        for i in range(4):
            torch.testing.assert_close(
                residual_state[..., i, :],
                hidden_states,
                rtol=1e-5,
                atol=1e-5
            )
    
    def test_read_shape(self):
        """Test read produces correct output shape."""
        gr = QwenGatedResidual(hidden_size=256, num_branches=4)
        residual_state = torch.randn(2, 32, 4, 256)
        
        hidden_states = gr.read(residual_state)
        
        # Should collapse branches: (batch, seq_len, hidden_size)
        assert hidden_states.shape == (2, 32, 256)
    
    def test_read_element_wise_gating(self):
        """Test that read uses element-wise dynamic gates."""
        gr = QwenGatedResidual(hidden_size=256, num_branches=4, bottleneck_rank=128)
        residual_state = torch.randn(2, 32, 4, 256)
        
        hidden_states = gr.read(residual_state)
        
        # Output should be weighted combination of branches
        assert hidden_states.shape == (2, 32, 256)
        
        # Gates should be applied (check that output is not just the first branch)
        assert not torch.allclose(hidden_states, residual_state[..., 0, :])
    
    def test_write_shape(self):
        """Test write produces correct output shape."""
        gr = QwenGatedResidual(hidden_size=256, num_branches=4)
        residual_state = torch.randn(2, 32, 4, 256)
        module_output = torch.randn(2, 32, 256)
        
        next_state = gr.write(residual_state, module_output)
        
        # Should maintain residual state shape
        assert next_state.shape == (2, 32, 4, 256)
    
    def test_write_updates_all_branches(self):
        """Test that write updates all branches."""
        gr = QwenGatedResidual(hidden_size=256, num_branches=4)
        residual_state = torch.randn(2, 32, 4, 256)
        module_output = torch.randn(2, 32, 256)
        
        next_state = gr.write(residual_state, module_output)
        
        # Next state should differ from original (updated)
        assert not torch.allclose(next_state, residual_state)
        
        # Each branch should be updated differently (due to scalar gates)
        for i in range(4):
            assert not torch.allclose(next_state[..., i, :], residual_state[..., i, :])
    
    def test_write_preserves_residual_structure(self):
        """Test that write maintains residual connection property."""
        gr = QwenGatedResidual(hidden_size=256, num_branches=4)
        residual_state = torch.randn(2, 32, 4, 256)
        module_output = torch.zeros(2, 32, 256)  # Zero output
        
        next_state = gr.write(residual_state, module_output)
        
        # With zero module output, state should change minimally (only normalization)
        # But not be identical due to RMSNorm
        diff = (next_state - residual_state).abs().mean()
        assert diff < 1.0  # Should be small change
    
    def test_forward_full_cycle(self):
        """Test full forward cycle: write then read."""
        gr = QwenGatedResidual(hidden_size=256, num_branches=4)
        residual_state = torch.randn(2, 32, 4, 256)
        module_output = torch.randn(2, 32, 256)
        
        next_state, next_input = gr(residual_state, module_output)
        
        # Check shapes
        assert next_state.shape == (2, 32, 4, 256)
        assert next_input.shape == (2, 32, 256)
    
    def test_forward_information_flow(self):
        """Test that forward properly flows information through write and read."""
        gr = QwenGatedResidual(hidden_size=256, num_branches=4)
        residual_state = torch.randn(2, 32, 4, 256)
        module_output = torch.randn(2, 32, 256)
        
        # Run multiple cycles
        state = residual_state
        for _ in range(3):
            state, next_input = gr(state, module_output)
        
        # State should evolve over cycles
        assert not torch.allclose(state, residual_state)
    
    def test_scalar_write_gates(self):
        """Test that write gates are per-branch scalars."""
        gr = QwenGatedResidual(hidden_size=256, num_branches=4)
        
        # Write gates should be learnable parameters
        assert gr.write_gates is not None
        assert gr.write_gates.shape == (4,)  # One scalar per branch
    
    def test_no_branch_mixing_matrix(self):
        """Test that Gated Residual does NOT have 4x4 mixing matrix (unlike mHC)."""
        gr = QwenGatedResidual(hidden_size=256, num_branches=4)
        
        # Should NOT have parameters for 4×4 branch mixing
        param_names = [name for name, _ in gr.named_parameters()]
        
        # Check that there's no residual_mapping or similar mHC-style parameters
        assert not any("residual_mapping" in name for name in param_names)
        
        # Should only have read gates and write gates
        assert any("read" in name for name in param_names)
        assert any("write" in name for name in param_names)
    
    def test_bottleneck_reduces_parameters(self):
        """Test that bottleneck rank reduces parameter count."""
        gr_small = QwenGatedResidual(hidden_size=256, num_branches=4, bottleneck_rank=64)
        gr_large = QwenGatedResidual(hidden_size=256, num_branches=4, bottleneck_rank=256)
        
        # Count parameters in read projection
        small_params = sum(p.numel() for p in gr_small.read_down_proj.parameters())
        large_params = sum(p.numel() for p in gr_large.read_down_proj.parameters())
        
        # Smaller bottleneck should have fewer parameters
        assert small_params < large_params
    
    def test_different_num_branches(self):
        """Test Gated Residual with different number of branches."""
        for num_branches in [2, 4, 8]:
            gr = QwenGatedResidual(hidden_size=256, num_branches=num_branches)
            
            hidden_states = torch.randn(2, 32, 256)
            residual_state = gr.init_state(hidden_states)
            
            assert residual_state.shape == (2, 32, num_branches, 256)
            
            # Test read/write work
            output = gr.read(residual_state)
            assert output.shape == (2, 32, 256)
    
    def test_gradient_flow(self):
        """Test that gradients flow through read and write operations."""
        gr = QwenGatedResidual(hidden_size=256, num_branches=4)
        
        hidden_states = torch.randn(2, 32, 256, requires_grad=True)
        residual_state = gr.init_state(hidden_states)
        module_output = torch.randn(2, 32, 256, requires_grad=True)
        
        next_state, next_input = gr(residual_state, module_output)
        loss = next_input.sum()
        loss.backward()
        
        # Gradients should flow to inputs
        assert hidden_states.grad is not None
        assert module_output.grad is not None
        assert not torch.allclose(hidden_states.grad, torch.zeros_like(hidden_states.grad))


class TestGatedResidualAlias:
    """Test that GatedResidual alias works."""
    
    def test_alias_is_same_class(self):
        """Test GatedResidual is an alias for QwenGatedResidual."""
        assert GatedResidual is QwenGatedResidual


class TestGatedResidualVsManifoldHC:
    """Compare Gated Residual with mHC to verify simplifications."""
    
    def test_gated_residual_simpler_than_mhc(self):
        """Verify Gated Residual has fewer parameters than mHC."""
        from torchforge.common.residual import ManifoldConstrainedHyperConnection
        
        gr = QwenGatedResidual(hidden_size=256, num_branches=4, bottleneck_rank=128)
        mhc = ManifoldConstrainedHyperConnection(
            hidden_size=256,
            expansion_factor=4,
            dynamic_residual_gate=True,
        )
        
        gr_params = sum(p.numel() for p in gr.parameters())
        mhc_params = sum(p.numel() for p in mhc.parameters())
        
        # GR should have fewer parameters (no 4×4 mixing matrix)
        assert gr_params < mhc_params
    
    def test_gated_residual_no_sinkhorn(self):
        """Verify Gated Residual does not use Sinkhorn projection."""
        gr = QwenGatedResidual(hidden_size=256, num_branches=4)
        
        # Should not have project_residual_mapping method
        assert not hasattr(gr, "project_residual_mapping")
        
        # Should not have constrained parameters method
        assert not hasattr(gr, "_constrained_parameters")


class TestGatedResidualQwenConfig:
    """Test Gated Residual with Qwen-specific configuration."""
    
    def test_qwen_default_config(self):
        """Test with Qwen's reported configuration."""
        # Qwen3.8-Flash-Next: branches=4, bottleneck_rank=320
        gr = QwenGatedResidual(
            hidden_size=2560,  # Qwen hidden size
            num_branches=4,
            bottleneck_rank=320,  # Qwen bottleneck
        )
        
        hidden_states = torch.randn(2, 128, 2560)
        residual_state = gr.init_state(hidden_states)
        
        assert residual_state.shape == (2, 128, 4, 2560)
        
        module_output = torch.randn(2, 128, 2560)
        next_state, next_input = gr(residual_state, module_output)
        
        assert next_state.shape == (2, 128, 4, 2560)
        assert next_input.shape == (2, 128, 2560)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
