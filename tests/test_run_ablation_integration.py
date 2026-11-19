"""Integration tests for run_ablation.py CLI script.

These tests verify the script integrates correctly with the ablation runner
and configuration manager.

**Validates: Requirements 4.1, 4.4**
"""

import sys
from pathlib import Path
import pytest

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from uav_survivor_detection.config.manager import ConfigurationManager
from uav_survivor_detection.ablation.runner import AblationRunner


def test_config_manager_loads_ablation_config():
    """Verify ConfigurationManager can load ablation_config.yaml."""
    config_path = PROJECT_ROOT / "configs" / "ablation_config.yaml"
    
    if not config_path.exists():
        pytest.skip(f"Config file not found: {config_path}")
    
    # Load config
    config_manager = ConfigurationManager(config_path)
    system_config = config_manager.load_config()
    
    # Verify ablation config exists
    assert system_config.ablation is not None, "Ablation config not found in YAML"
    
    # Verify variants are loaded
    assert len(system_config.ablation.variants) > 0, "No variants found in config"
    
    # Verify each variant has required attributes
    for variant in system_config.ablation.variants:
        assert variant.name, f"Variant missing name: {variant}"
        assert variant.enhancement_config is not None, f"Variant {variant.name} missing enhancement config"
        assert variant.training_config is not None, f"Variant {variant.name} missing training config"


def test_ablation_config_has_required_paths():
    """Verify ablation config has all required path attributes."""
    config_path = PROJECT_ROOT / "configs" / "ablation_config.yaml"
    
    if not config_path.exists():
        pytest.skip(f"Config file not found: {config_path}")
    
    config_manager = ConfigurationManager(config_path)
    system_config = config_manager.load_config()
    ablation_config = system_config.ablation
    
    # Check required attributes
    assert hasattr(ablation_config, 'output_dir'), "Missing output_dir"
    assert hasattr(ablation_config, 'dataset_yaml'), "Missing dataset_yaml"
    assert hasattr(ablation_config, 'test_subset_path'), "Missing test_subset_path"
    assert hasattr(ablation_config, 'variants'), "Missing variants"
    
    # Verify they are Path objects or can be converted to Path
    assert isinstance(ablation_config.output_dir, Path), "output_dir should be Path"
    assert isinstance(ablation_config.dataset_yaml, Path), "dataset_yaml should be Path"
    assert isinstance(ablation_config.test_subset_path, Path), "test_subset_path should be Path"


def test_ablation_runner_can_be_initialized():
    """Verify AblationRunner can be initialized with loaded config."""
    config_path = PROJECT_ROOT / "configs" / "ablation_config.yaml"
    
    if not config_path.exists():
        pytest.skip(f"Config file not found: {config_path}")
    
    config_manager = ConfigurationManager(config_path)
    system_config = config_manager.load_config()
    ablation_config = system_config.ablation
    
    # Initialize runner (should not raise)
    try:
        runner = AblationRunner(ablation_config)
        assert runner is not None, "AblationRunner initialization returned None"
        assert runner.config == ablation_config, "Runner config mismatch"
    except Exception as e:
        pytest.fail(f"AblationRunner initialization failed: {e}")


def test_cli_overrides_can_be_applied():
    """Verify CLI overrides work as expected."""
    config_path = PROJECT_ROOT / "configs" / "ablation_config.yaml"
    
    if not config_path.exists():
        pytest.skip(f"Config file not found: {config_path}")
    
    config_manager = ConfigurationManager(config_path)
    system_config = config_manager.load_config()
    ablation_config = system_config.ablation
    
    # Apply overrides
    original_output_dir = ablation_config.output_dir
    new_output_dir = Path("results/custom_ablation")
    
    ablation_config.output_dir = new_output_dir
    assert ablation_config.output_dir == new_output_dir, "Output dir override failed"
    
    # Apply device override to all variants
    new_device = "cpu"
    for variant in ablation_config.variants:
        variant.training_config.device = new_device
    
    for variant in ablation_config.variants:
        assert variant.training_config.device == new_device, \
            f"Device override failed for variant {variant.name}"


def test_variant_enhancement_configs_are_valid():
    """Verify each variant has valid enhancement configuration."""
    config_path = PROJECT_ROOT / "configs" / "ablation_config.yaml"
    
    if not config_path.exists():
        pytest.skip(f"Config file not found: {config_path}")
    
    config_manager = ConfigurationManager(config_path)
    system_config = config_manager.load_config()
    
    for variant in system_config.ablation.variants:
        enh = variant.enhancement_config
        
        # Check all boolean flags exist
        assert hasattr(enh, 'enable_gam'), f"Variant {variant.name} missing enable_gam"
        assert hasattr(enh, 'enable_cbam'), f"Variant {variant.name} missing enable_cbam"
        assert hasattr(enh, 'enable_c3k2_arc'), f"Variant {variant.name} missing enable_c3k2_arc"
        assert hasattr(enh, 'enable_cl_concat'), f"Variant {variant.name} missing enable_cl_concat"
        
        # Check reduction ratios are positive if enhancements are enabled
        if enh.enable_gam:
            assert enh.gam_reduction > 0, f"Invalid gam_reduction for {variant.name}"
        
        if enh.enable_cbam:
            assert enh.cbam_reduction > 0, f"Invalid cbam_reduction for {variant.name}"


def test_variant_training_configs_are_valid():
    """Verify each variant has valid training configuration."""
    config_path = PROJECT_ROOT / "configs" / "ablation_config.yaml"
    
    if not config_path.exists():
        pytest.skip(f"Config file not found: {config_path}")
    
    config_manager = ConfigurationManager(config_path)
    system_config = config_manager.load_config()
    
    for variant in system_config.ablation.variants:
        train = variant.training_config
        
        # Check required attributes
        assert train.epochs > 0, f"Invalid epochs for {variant.name}"
        assert train.batch_size > 0, f"Invalid batch_size for {variant.name}"
        assert train.img_size > 0, f"Invalid img_size for {variant.name}"
        assert train.learning_rate > 0, f"Invalid learning_rate for {variant.name}"
        assert train.workers >= 0, f"Invalid workers for {variant.name}"
        
        # Check device is valid
        assert train.device in ["cpu", "cuda"], f"Invalid device for {variant.name}"


def test_baseline_variant_has_no_enhancements():
    """Verify baseline variant has all enhancements disabled."""
    config_path = PROJECT_ROOT / "configs" / "ablation_config.yaml"
    
    if not config_path.exists():
        pytest.skip(f"Config file not found: {config_path}")
    
    config_manager = ConfigurationManager(config_path)
    system_config = config_manager.load_config()
    
    # Find baseline variant
    baseline = None
    for variant in system_config.ablation.variants:
        if variant.name.lower() == "baseline":
            baseline = variant
            break
    
    if baseline is None:
        pytest.skip("No baseline variant found in config")
    
    # Verify all enhancements are disabled
    enh = baseline.enhancement_config
    assert not enh.enable_gam, "Baseline should not have GAM enabled"
    assert not enh.enable_cbam, "Baseline should not have CBAM enabled"
    assert not enh.enable_c3k2_arc, "Baseline should not have C3k2-ARC enabled"
    assert not enh.enable_cl_concat, "Baseline should not have CL-Concat enabled"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
