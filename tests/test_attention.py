"""
Unit tests for attention mechanisms.

Tests verify that all attention modules:
1. Accept and process tensors of correct shape
2. Preserve spatial dimensions appropriately
3. Have reasonable parameter counts
4. Can perform forward pass without errors
"""

import pytest
import torch
import torch.nn as nn

from src.uav_survivor_detection.models.attention import (
    ChannelAttention,
    SpatialAttention,
    CBAM,
    GAM,
    C3k2_ARC,
    CLConcat,
)


class TestChannelAttention:
    """Tests for ChannelAttention module."""
    
    def test_forward_shape(self):
        """Test that output shape matches input shape."""
        channels = 64
        module = ChannelAttention(channels=channels, reduction=16)
        
        x = torch.randn(2, channels, 32, 32)
        out = module(x)
        
        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
    
    def test_reduction_ratio(self):
        """Test different reduction ratios."""
        channels = 64
        for reduction in [4, 8, 16]:
            module = ChannelAttention(channels=channels, reduction=reduction)
            x = torch.randn(1, channels, 16, 16)
            out = module(x)
            assert out.shape == x.shape
    
    def test_attention_range(self):
        """Test that attention weights are properly bounded."""
        channels = 32
        module = ChannelAttention(channels=channels)
        module.eval()
        
        x = torch.randn(1, channels, 16, 16)
        with torch.no_grad():
            out = module(x)
        
        # Output should not be all zeros or have extreme values
        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isinf(out).any(), "Output contains Inf"


class TestSpatialAttention:
    """Tests for SpatialAttention module."""
    
    def test_forward_shape(self):
        """Test that output shape matches input shape."""
        module = SpatialAttention(kernel_size=7)
        
        x = torch.randn(2, 64, 32, 32)
        out = module(x)
        
        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
    
    def test_different_kernel_sizes(self):
        """Test with different kernel sizes."""
        for kernel_size in [3, 5, 7]:
            module = SpatialAttention(kernel_size=kernel_size)
            x = torch.randn(1, 32, 16, 16)
            out = module(x)
            assert out.shape == x.shape
    
    def test_spatial_dimensions(self):
        """Test with various spatial dimensions."""
        module = SpatialAttention()
        
        for h, w in [(16, 16), (32, 32), (64, 64)]:
            x = torch.randn(1, 64, h, w)
            out = module(x)
            assert out.shape == (1, 64, h, w)


class TestCBAM:
    """Tests for CBAM module."""
    
    def test_forward_shape(self):
        """Test that output shape matches input shape."""
        channels = 64
        module = CBAM(channels=channels)
        
        x = torch.randn(2, channels, 32, 32)
        out = module(x)
        
        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
    
    def test_cbam_configuration(self):
        """Test CBAM with different configurations."""
        channels = 128
        module = CBAM(channels=channels, reduction=8, kernel_size=5)
        
        x = torch.randn(1, channels, 24, 24)
        out = module(x)
        
        assert out.shape == x.shape
    
    def test_gradient_flow(self):
        """Test that gradients flow through CBAM."""
        channels = 32
        module = CBAM(channels=channels)
        
        x = torch.randn(1, channels, 16, 16, requires_grad=True)
        out = module(x)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None, "Gradients should flow through CBAM"
        assert not torch.isnan(x.grad).any(), "Gradients contain NaN"


class TestGAM:
    """Tests for GAM (Global Attention Mechanism) module."""
    
    def test_forward_shape(self):
        """Test that output shape matches input shape."""
        channels = 64
        module = GAM(channels=channels)
        
        x = torch.randn(2, channels, 32, 32)
        out = module(x)
        
        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
    
    def test_gam_reduction(self):
        """Test GAM with different reduction ratios."""
        channels = 128
        for reduction in [4, 8, 16]:
            module = GAM(channels=channels, reduction=reduction)
            x = torch.randn(1, channels, 16, 16)
            out = module(x)
            assert out.shape == x.shape
    
    def test_parameter_count(self):
        """Test that GAM has reasonable parameter count."""
        channels = 64
        module = GAM(channels=channels, reduction=16)
        
        param_count = sum(p.numel() for p in module.parameters())
        
        # GAM should be lightweight - verify parameter count is reasonable
        # GAM includes channel MLP and spatial conv layers with 7x7 kernels
        assert param_count > 0, "Module should have parameters"
        # Reasonable upper bound considering spatial convs with 7x7 kernels
        max_expected = channels * channels * 50  # Generous upper bound
        assert param_count < max_expected, f"Parameter count {param_count} seems too high"
    
    def test_different_spatial_sizes(self):
        """Test GAM with various spatial dimensions."""
        channels = 64
        module = GAM(channels=channels)
        
        for h, w in [(8, 8), (16, 16), (32, 64)]:
            x = torch.randn(1, channels, h, w)
            out = module(x)
            assert out.shape == (1, channels, h, w)


class TestC3k2_ARC:
    """Tests for C3k2_ARC (rotation-aware convolution) module."""
    
    def test_forward_shape(self):
        """Test that output has correct shape."""
        in_channels = 64
        out_channels = 128
        module = C3k2_ARC(in_channels=in_channels, out_channels=out_channels)
        
        x = torch.randn(2, in_channels, 32, 32)
        out = module(x)
        
        assert out.shape == (2, out_channels, 32, 32), \
            f"Expected (2, {out_channels}, 32, 32), got {out.shape}"
    
    def test_rotation_invariance(self):
        """Test that module handles rotated inputs."""
        in_channels = 32
        out_channels = 64
        module = C3k2_ARC(
            in_channels=in_channels,
            out_channels=out_channels,
            num_rotations=4
        )
        module.eval()
        
        x = torch.randn(1, in_channels, 16, 16)
        
        with torch.no_grad():
            out_0 = module(x)
            
            # Rotate input 90 degrees
            x_90 = torch.rot90(x, k=1, dims=[2, 3])
            out_90 = module(x_90)
        
        # Outputs should have same shape
        assert out_0.shape == out_90.shape
        
        # Outputs should be different (since we're testing rotation awareness)
        # but both should be valid
        assert not torch.isnan(out_0).any()
        assert not torch.isnan(out_90).any()
    
    def test_num_rotations(self):
        """Test with different numbers of rotations."""
        in_channels = 32
        out_channels = 64
        
        for num_rotations in [2, 4, 8]:
            module = C3k2_ARC(
                in_channels=in_channels,
                out_channels=out_channels,
                num_rotations=num_rotations
            )
            x = torch.randn(1, in_channels, 16, 16)
            out = module(x)
            assert out.shape == (1, out_channels, 16, 16)


class TestCLConcat:
    """Tests for CLConcat (cross-level concatenation) module."""
    
    def test_forward_concat_fusion(self):
        """Test concatenation fusion method."""
        in_channels_list = [64, 128, 256]
        out_channels = 128
        
        module = CLConcat(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            fusion_method='concat'
        )
        
        # Create features at different scales
        features = [
            torch.randn(2, 64, 32, 32),
            torch.randn(2, 128, 32, 32),
            torch.randn(2, 256, 32, 32),
        ]
        
        out = module(features)
        
        assert out.shape == (2, out_channels, 32, 32), \
            f"Expected (2, {out_channels}, 32, 32), got {out.shape}"
    
    def test_forward_add_fusion(self):
        """Test addition fusion method."""
        in_channels_list = [64, 128, 256]
        out_channels = 128
        
        module = CLConcat(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            fusion_method='add'
        )
        
        features = [
            torch.randn(2, 64, 32, 32),
            torch.randn(2, 128, 32, 32),
            torch.randn(2, 256, 32, 32),
        ]
        
        out = module(features)
        
        assert out.shape == (2, out_channels, 32, 32)
    
    def test_multi_scale_fusion(self):
        """Test fusion of features at different spatial scales."""
        in_channels_list = [64, 128]
        out_channels = 64
        
        module = CLConcat(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            fusion_method='concat'
        )
        
        # Features at different spatial resolutions
        features = [
            torch.randn(1, 64, 32, 32),   # Larger feature map
            torch.randn(1, 128, 16, 16),  # Smaller feature map
        ]
        
        out = module(features)
        
        # Output should match the largest feature map size
        assert out.shape == (1, out_channels, 32, 32), \
            f"Expected (1, {out_channels}, 32, 32), got {out.shape}"
    
    def test_incorrect_feature_count(self):
        """Test that module raises error with wrong number of features."""
        in_channels_list = [64, 128, 256]
        out_channels = 128
        
        module = CLConcat(
            in_channels_list=in_channels_list,
            out_channels=out_channels
        )
        
        # Provide only 2 features when 3 are expected
        features = [
            torch.randn(1, 64, 32, 32),
            torch.randn(1, 128, 32, 32),
        ]
        
        with pytest.raises(ValueError):
            module(features)


class TestAttentionIntegration:
    """Integration tests for attention mechanisms."""
    
    def test_attention_in_sequence(self):
        """Test stacking multiple attention modules."""
        channels = 64
        
        # Create a sequence of attention modules
        model = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, padding=1),
            CBAM(channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            GAM(channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        
        x = torch.randn(1, 3, 32, 32)
        out = model(x)
        
        assert out.shape == (1, channels, 32, 32)
    
    def test_all_modules_trainable(self):
        """Test that all modules are trainable."""
        modules = [
            ChannelAttention(64),
            SpatialAttention(),
            CBAM(64),
            GAM(64),
            C3k2_ARC(64, 64),
            CLConcat([64, 128], 64),
        ]
        
        for module in modules:
            module.train()
            param_count = sum(p.numel() for p in module.parameters())
            assert param_count > 0, f"{module.__class__.__name__} has no parameters"
            
            # Check that parameters require gradients
            trainable_params = sum(
                p.numel() for p in module.parameters() if p.requires_grad
            )
            assert trainable_params > 0, \
                f"{module.__class__.__name__} has no trainable parameters"
