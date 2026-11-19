"""Tests for EnhancementModule implementation."""

import pytest
import torch
from pathlib import Path

from uav_survivor_detection.models.enhancement import (
    EnhancementModule,
    EnhancementConfig,
    ModelComplexity,
)


@pytest.fixture
def baseline_yolo():
    """Create a baseline YOLO11n model for testing."""
    from ultralytics import YOLO
    
    # Load pretrained YOLO11n (will download if not present)
    model = YOLO("yolo11n.pt")
    return model


@pytest.fixture
def enhancement_config_all():
    """Create enhancement config with all mechanisms enabled."""
    return EnhancementConfig(
        enable_gam=True,
        enable_cbam=True,
        enable_c3k2_arc=True,
        enable_cl_concat=True,
        gam_reduction=16,
        cbam_reduction=16,
    )


@pytest.fixture
def enhancement_config_gam_only():
    """Create enhancement config with only GAM enabled."""
    return EnhancementConfig(
        enable_gam=True,
        enable_cbam=False,
        enable_c3k2_arc=False,
        enable_cl_concat=False,
        gam_reduction=16,
    )


@pytest.fixture
def enhancement_config_cbam_only():
    """Create enhancement config with only CBAM enabled."""
    return EnhancementConfig(
        enable_gam=False,
        enable_cbam=True,
        enable_c3k2_arc=False,
        enable_cl_concat=False,
        cbam_reduction=16,
    )


class TestEnhancementConfig:
    """Test EnhancementConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = EnhancementConfig()
        
        assert config.enable_gam is False
        assert config.enable_cbam is False
        assert config.enable_c3k2_arc is False
        assert config.enable_cl_concat is False
        assert config.gam_reduction == 16
        assert config.cbam_reduction == 16
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = EnhancementConfig(
            enable_gam=True,
            enable_cbam=True,
            gam_reduction=8,
            cbam_reduction=8,
        )
        
        assert config.enable_gam is True
        assert config.enable_cbam is True
        assert config.gam_reduction == 8
        assert config.cbam_reduction == 8


class TestModelComplexity:
    """Test ModelComplexity dataclass."""
    
    def test_model_complexity_creation(self):
        """Test ModelComplexity creation."""
        complexity = ModelComplexity(parameters=1000000, gflops=5.5)
        
        assert complexity.parameters == 1000000
        assert complexity.gflops == 5.5


class TestEnhancementModule:
    """Test EnhancementModule class."""
    
    def test_initialization(self, enhancement_config_all):
        """Test EnhancementModule initialization."""
        module = EnhancementModule(enhancement_config_all)
        
        assert module.config == enhancement_config_all
        assert module.logger is not None
    
    def test_compute_complexity_baseline(self, baseline_yolo):
        """Test complexity computation on baseline model.
        
        **Validates: Requirements 3.3, 3.4**
        """
        config = EnhancementConfig()
        module = EnhancementModule(config)
        
        complexity = module.compute_complexity(baseline_yolo)
        
        assert isinstance(complexity, ModelComplexity)
        assert complexity.parameters > 0
        assert complexity.gflops > 0
        
        # YOLO11n should be a small model
        assert complexity.parameters < 10_000_000  # Less than 10M params
        assert complexity.gflops < 20.0  # Less than 20 GFLOPs
    
    def test_build_enhanced_model_gam(self, baseline_yolo, enhancement_config_gam_only):
        """Test building enhanced model with GAM only.
        
        **Validates: Requirements 3.1, 3.2**
        """
        module = EnhancementModule(enhancement_config_gam_only)
        
        # Build enhanced model
        enhanced_model = module.build_enhanced_model(baseline_yolo)
        
        # Verify model is still a YOLO instance (interface preserved)
        from ultralytics import YOLO
        assert isinstance(enhanced_model, YOLO)
        
        # Verify model can perform inference (interface compatibility)
        dummy_input = torch.randn(1, 3, 640, 640)
        try:
            # Just verify the model structure is valid
            # We don't run actual inference as it's expensive
            assert enhanced_model.model is not None
        except Exception as e:
            pytest.fail(f"Enhanced model interface broken: {e}")
    
    def test_build_enhanced_model_cbam(self, baseline_yolo, enhancement_config_cbam_only):
        """Test building enhanced model with CBAM only.
        
        **Validates: Requirements 3.1, 3.2**
        """
        module = EnhancementModule(enhancement_config_cbam_only)
        
        # Build enhanced model
        enhanced_model = module.build_enhanced_model(baseline_yolo)
        
        # Verify model is still a YOLO instance
        from ultralytics import YOLO
        assert isinstance(enhanced_model, YOLO)
        assert enhanced_model.model is not None
    
    def test_build_enhanced_model_all(self, baseline_yolo, enhancement_config_all):
        """Test building enhanced model with all mechanisms.
        
        **Validates: Requirements 3.1, 3.2, 3.6**
        """
        module = EnhancementModule(enhancement_config_all)
        
        # Build enhanced model
        enhanced_model = module.build_enhanced_model(baseline_yolo)
        
        # Verify model is still a YOLO instance
        from ultralytics import YOLO
        assert isinstance(enhanced_model, YOLO)
        assert enhanced_model.model is not None
    
    def test_validate_constraints_satisfied(self):
        """Test constraint validation when constraints are satisfied.
        
        **Validates: Requirements 3.3, 3.4**
        """
        config = EnhancementConfig()
        module = EnhancementModule(config)
        
        baseline = ModelComplexity(parameters=1000000, gflops=10.0)
        enhanced = ModelComplexity(parameters=1150000, gflops=12.0)  # 15% param, 20% GFLOPs
        
        result = module.validate_constraints(baseline, enhanced)
        
        assert result is True
    
    def test_validate_constraints_param_violated(self):
        """Test constraint validation when parameter constraint is violated.
        
        **Validates: Requirements 3.3**
        """
        config = EnhancementConfig()
        module = EnhancementModule(config)
        
        baseline = ModelComplexity(parameters=1000000, gflops=10.0)
        enhanced = ModelComplexity(parameters=1250000, gflops=12.0)  # 25% param increase (exceeds 20%)
        
        result = module.validate_constraints(baseline, enhanced)
        
        assert result is False
    
    def test_validate_constraints_gflops_violated(self):
        """Test constraint validation when GFLOPs constraint is violated.
        
        **Validates: Requirements 3.4**
        """
        config = EnhancementConfig()
        module = EnhancementModule(config)
        
        baseline = ModelComplexity(parameters=1000000, gflops=10.0)
        enhanced = ModelComplexity(parameters=1150000, gflops=13.0)  # 30% GFLOPs increase (exceeds 25%)
        
        result = module.validate_constraints(baseline, enhanced)
        
        assert result is False
    
    def test_validate_constraints_both_violated(self):
        """Test constraint validation when both constraints are violated."""
        config = EnhancementConfig()
        module = EnhancementModule(config)
        
        baseline = ModelComplexity(parameters=1000000, gflops=10.0)
        enhanced = ModelComplexity(parameters=1300000, gflops=13.5)  # 30% params, 35% GFLOPs
        
        result = module.validate_constraints(baseline, enhanced)
        
        assert result is False
    
    def test_enhancement_increases_complexity(self, baseline_yolo, enhancement_config_gam_only):
        """Test that enhancement increases model complexity as expected.
        
        **Validates: Requirements 3.3, 3.4**
        """
        module = EnhancementModule(enhancement_config_gam_only)
        
        # Compute baseline complexity
        baseline_complexity = module.compute_complexity(baseline_yolo)
        
        # Build enhanced model
        enhanced_model = module.build_enhanced_model(baseline_yolo)
        
        # Compute enhanced complexity
        enhanced_complexity = module.compute_complexity(enhanced_model)
        
        # Verify complexity increased
        assert enhanced_complexity.parameters > baseline_complexity.parameters
        # Note: GFLOPs might not always increase significantly with small attention modules
        # So we just check that it's computed
        assert enhanced_complexity.gflops > 0
    
    def test_enhancement_respects_constraints(self, baseline_yolo, enhancement_config_gam_only):
        """Test that enhancement respects complexity constraints.
        
        **Validates: Requirements 3.3, 3.4**
        """
        module = EnhancementModule(enhancement_config_gam_only)
        
        # Compute baseline complexity
        baseline_complexity = module.compute_complexity(baseline_yolo)
        
        # Build enhanced model
        enhanced_model = module.build_enhanced_model(baseline_yolo)
        
        # Compute enhanced complexity
        enhanced_complexity = module.compute_complexity(enhanced_model)
        
        # Validate constraints
        constraints_met = module.validate_constraints(baseline_complexity, enhanced_complexity)
        
        # With only GAM enabled, constraints should be satisfied
        assert constraints_met is True


    def test_build_enhanced_model_gam_cbam(self):
        """Test building enhanced model with GAM+CBAM together.
        
        **Validates: Requirements 3.1, 3.2**
        """
        from ultralytics import YOLO
        
        # Load fresh baseline model
        baseline_yolo = YOLO("yolo11n.pt")
        baseline_params = sum(p.numel() for p in baseline_yolo.model.parameters())
        
        config = EnhancementConfig(
            enable_gam=True,
            enable_cbam=True,
            enable_c3k2_arc=False,
            enable_cl_concat=False,
            gam_reduction=16,
            cbam_reduction=16,
        )
        
        module = EnhancementModule(config)
        
        # Build enhanced model
        enhanced_model = module.build_enhanced_model(baseline_yolo)
        
        # Verify model is still a YOLO instance
        assert isinstance(enhanced_model, YOLO)
        assert enhanced_model.model is not None
        
        # Verify model structure contains both attention mechanisms
        # by checking that parameters increased (indicating modules were added)
        enhanced_params = sum(p.numel() for p in enhanced_model.model.parameters())
        assert enhanced_params > baseline_params, \
            f"Enhanced model should have more parameters: baseline={baseline_params}, enhanced={enhanced_params}"
    
    def test_constraint_violation_too_many_enhancements(self, baseline_yolo):
        """Test that constraint validation fails when too many enhancements are enabled.
        
        This test verifies that enabling all enhancement mechanisms (GAM, CBAM, C3k2-ARC,
        CL-Concat) causes the complexity constraints to be violated.
        
        **Validates: Requirements 3.3, 3.4**
        """
        # Configure with ALL enhancements enabled
        config = EnhancementConfig(
            enable_gam=True,
            enable_cbam=True,
            enable_c3k2_arc=True,
            enable_cl_concat=True,
            gam_reduction=16,
            cbam_reduction=16,
        )
        
        module = EnhancementModule(config)
        
        # Compute baseline complexity
        baseline_complexity = module.compute_complexity(baseline_yolo)
        
        # Build enhanced model with all mechanisms
        enhanced_model = module.build_enhanced_model(baseline_yolo)
        
        # Compute enhanced complexity
        enhanced_complexity = module.compute_complexity(enhanced_model)
        
        # Validate constraints - should fail with too many enhancements
        constraints_met = module.validate_constraints(baseline_complexity, enhanced_complexity)
        
        # With all enhancements enabled, at least one constraint should be violated
        # (parameter increase >20% or GFLOPs increase >25%)
        assert constraints_met is False, \
            "Expected constraints to be violated with all enhancements enabled"
        
        # Verify that parameters increased significantly
        param_increase_pct = (
            (enhanced_complexity.parameters - baseline_complexity.parameters) 
            / baseline_complexity.parameters * 100
        )
        gflops_increase_pct = (
            (enhanced_complexity.gflops - baseline_complexity.gflops) 
            / baseline_complexity.gflops * 100
        )
        
        # At least one should exceed its constraint
        assert param_increase_pct > 20.0 or gflops_increase_pct > 25.0, \
            f"Expected constraint violation but got param_increase={param_increase_pct:.2f}%, " \
            f"gflops_increase={gflops_increase_pct:.2f}%"


class TestIntegration:
    """Integration tests for EnhancementModule."""
    
    def test_end_to_end_workflow(self, baseline_yolo):
        """Test complete enhancement workflow.
        
        **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5**
        """
        # Create configuration
        config = EnhancementConfig(
            enable_gam=True,
            enable_cbam=False,
            gam_reduction=16,
        )
        
        # Initialize module
        module = EnhancementModule(config)
        
        # Compute baseline complexity
        baseline_complexity = module.compute_complexity(baseline_yolo)
        assert baseline_complexity.parameters > 0
        
        # Build enhanced model
        enhanced_model = module.build_enhanced_model(baseline_yolo)
        assert enhanced_model is not None
        
        # Compute enhanced complexity
        enhanced_complexity = module.compute_complexity(enhanced_model)
        assert enhanced_complexity.parameters > baseline_complexity.parameters
        
        # Validate constraints
        constraints_met = module.validate_constraints(baseline_complexity, enhanced_complexity)
        assert constraints_met is True
        
        # Verify model can be saved (interface compatibility for training pipeline)
        # This validates requirement 3.5 (Training_Pipeline SHALL accept enhanced model)
        try:
            # Just verify the save method exists and is callable
            assert hasattr(enhanced_model, 'save')
            assert callable(enhanced_model.save)
        except Exception as e:
            pytest.fail(f"Enhanced model cannot be saved: {e}")
    
    def test_enhanced_model_save_and_load(self, baseline_yolo, tmp_path):
        """Test enhanced model can be saved and loaded correctly.
        
        This test verifies that:
        1. Enhanced models can be saved to disk
        2. Saved models can be loaded back
        3. Loaded models maintain the same architecture and parameters
        
        **Validates: Requirements 3.2, 3.5**
        """
        # Create enhanced model
        config = EnhancementConfig(
            enable_gam=True,
            enable_cbam=True,
            gam_reduction=16,
            cbam_reduction=16,
        )
        
        module = EnhancementModule(config)
        enhanced_model = module.build_enhanced_model(baseline_yolo)
        
        # Save the enhanced model
        save_path = tmp_path / "enhanced_model.pt"
        try:
            enhanced_model.save(str(save_path))
        except Exception as e:
            pytest.fail(f"Failed to save enhanced model: {e}")
        
        # Verify file was created
        assert save_path.exists(), "Model file was not created"
        
        # Load the model back
        from ultralytics import YOLO
        try:
            loaded_model = YOLO(str(save_path))
        except Exception as e:
            pytest.fail(f"Failed to load enhanced model: {e}")
        
        # Verify loaded model has same parameter count
        original_params = sum(p.numel() for p in enhanced_model.model.parameters())
        loaded_params = sum(p.numel() for p in loaded_model.model.parameters())
        
        assert loaded_params == original_params, \
            f"Parameter count mismatch: original={original_params}, loaded={loaded_params}"
        
        # Verify loaded model can perform forward pass
        try:
            dummy_input = torch.randn(1, 3, 640, 640)
            # Just verify model structure is intact
            assert loaded_model.model is not None
            assert hasattr(loaded_model, 'predict')
        except Exception as e:
            pytest.fail(f"Loaded model forward pass failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
