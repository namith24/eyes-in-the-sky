"""Unit tests for dataset preparation script.

Tests the prepare_dataset.py CLI script functionality including:
- Argument parsing
- Configuration loading and overrides
- Error handling
- Exit codes

**Validates: Requirements 1.1, 1.4**
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from prepare_dataset import (
    parse_arguments,
    apply_cli_overrides,
    print_config_summary,
    print_manifest_summary,
)


class TestArgumentParsing(unittest.TestCase):
    """Test command-line argument parsing."""
    
    def test_parse_required_arguments(self):
        """Test parsing required config argument."""
        with patch('sys.argv', ['prepare_dataset.py', '--config', 'configs/test.yaml']):
            args = parse_arguments()
            self.assertEqual(args.config, 'configs/test.yaml')
            self.assertFalse(args.skip_verification)
            self.assertFalse(args.overwrite)
    
    def test_parse_optional_overrides(self):
        """Test parsing optional override arguments."""
        test_args = [
            'prepare_dataset.py',
            '--config', 'configs/test.yaml',
            '--dataset-path', 'data/custom',
            '--output-dir', 'output/custom',
            '--train-ratio', '0.8',
            '--val-ratio', '0.15',
            '--test-ratio', '0.05',
            '--random-seed', '123',
            '--overwrite',
            '--skip-verification'
        ]
        
        with patch('sys.argv', test_args):
            args = parse_arguments()
            self.assertEqual(args.dataset_path, 'data/custom')
            self.assertEqual(args.output_dir, 'output/custom')
            self.assertEqual(args.train_ratio, 0.8)
            self.assertEqual(args.val_ratio, 0.15)
            self.assertEqual(args.test_ratio, 0.05)
            self.assertEqual(args.random_seed, 123)
            self.assertTrue(args.overwrite)
            self.assertTrue(args.skip_verification)


class TestConfigurationOverrides(unittest.TestCase):
    """Test CLI override application to configuration."""
    
    def setUp(self):
        """Set up test fixtures."""
        from uav_survivor_detection.config.models import SystemConfig, DatasetConfig
        
        self.config = SystemConfig(
            dataset=DatasetConfig(
                dataset_path=Path('data/original'),
                output_dir=Path('output/original'),
                train_ratio=0.7,
                val_ratio=0.2,
                test_ratio=0.1,
                random_seed=42,
                overwrite_existing=False
            )
        )
    
    def test_apply_dataset_path_override(self):
        """Test dataset path override."""
        args = Mock()
        args.dataset_path = 'data/custom'
        args.output_dir = None
        args.train_ratio = None
        args.val_ratio = None
        args.test_ratio = None
        args.random_seed = None
        args.overwrite = False
        
        apply_cli_overrides(self.config, args)
        
        self.assertEqual(self.config.dataset.dataset_path, Path('data/custom'))
        self.assertEqual(self.config.dataset.output_dir, Path('output/original'))
    
    def test_apply_split_ratio_overrides(self):
        """Test split ratio overrides."""
        args = Mock()
        args.dataset_path = None
        args.output_dir = None
        args.train_ratio = 0.8
        args.val_ratio = 0.15
        args.test_ratio = 0.05
        args.random_seed = None
        args.overwrite = False
        
        apply_cli_overrides(self.config, args)
        
        self.assertEqual(self.config.dataset.train_ratio, 0.8)
        self.assertEqual(self.config.dataset.val_ratio, 0.15)
        self.assertEqual(self.config.dataset.test_ratio, 0.05)
    
    def test_apply_overwrite_flag(self):
        """Test overwrite flag override."""
        args = Mock()
        args.dataset_path = None
        args.output_dir = None
        args.train_ratio = None
        args.val_ratio = None
        args.test_ratio = None
        args.random_seed = None
        args.overwrite = True
        
        apply_cli_overrides(self.config, args)
        
        self.assertTrue(self.config.dataset.overwrite_existing)
    
    def test_no_overrides_applied(self):
        """Test that None values don't override config."""
        args = Mock()
        args.dataset_path = None
        args.output_dir = None
        args.train_ratio = None
        args.val_ratio = None
        args.test_ratio = None
        args.random_seed = None
        args.overwrite = False
        
        original_path = self.config.dataset.dataset_path
        original_ratios = (
            self.config.dataset.train_ratio,
            self.config.dataset.val_ratio,
            self.config.dataset.test_ratio
        )
        
        apply_cli_overrides(self.config, args)
        
        self.assertEqual(self.config.dataset.dataset_path, original_path)
        self.assertEqual(self.config.dataset.train_ratio, original_ratios[0])
        self.assertEqual(self.config.dataset.val_ratio, original_ratios[1])
        self.assertEqual(self.config.dataset.test_ratio, original_ratios[2])


class TestPrintFunctions(unittest.TestCase):
    """Test output formatting functions."""
    
    def test_print_config_summary(self):
        """Test config summary printing doesn't raise errors."""
        from uav_survivor_detection.config.models import DatasetConfig
        
        config = DatasetConfig(
            dataset_path=Path('data/C2A'),
            output_dir=Path('data/processed'),
            train_ratio=0.7,
            val_ratio=0.2,
            test_ratio=0.1,
            random_seed=42,
            overwrite_existing=False
        )
        
        # Should not raise any exceptions
        with patch('sys.stdout'):
            print_config_summary(config)
    
    def test_print_manifest_summary(self):
        """Test manifest summary printing doesn't raise errors."""
        from uav_survivor_detection.dataset.pipeline import DatasetManifest
        
        manifest = DatasetManifest(
            total_images=1000,
            train_count=700,
            val_count=200,
            test_count=100,
            class_distribution={'class_0': 1500},
            split_ratios=(0.7, 0.2, 0.1),
            timestamp='2024-01-01T00:00:00Z'
        )
        
        # Should not raise any exceptions
        with patch('sys.stdout'):
            print_manifest_summary(manifest)


class TestScriptIntegration(unittest.TestCase):
    """Integration tests for the script with mock data."""
    
    def setUp(self):
        """Set up temporary directories for testing."""
        self.test_dir = tempfile.mkdtemp()
        self.dataset_dir = Path(self.test_dir) / 'dataset'
        self.output_dir = Path(self.test_dir) / 'output'
        self.config_file = Path(self.test_dir) / 'test_config.yaml'
        
        # Create minimal dataset structure
        self.dataset_dir.mkdir()
        (self.dataset_dir / 'images').mkdir()
        (self.dataset_dir / 'labels').mkdir()
        
        # Create a sample image and label
        img_path = self.dataset_dir / 'images' / 'test001.jpg'
        lbl_path = self.dataset_dir / 'labels' / 'test001.txt'
        
        # Create dummy image file
        img_path.write_text('fake image data')
        
        # Create valid YOLO label
        lbl_path.write_text('0 0.5 0.5 0.1 0.1\n')
        
        # Create config file (use forward slashes for cross-platform compatibility)
        dataset_dir_str = str(self.dataset_dir).replace('\\', '/')
        output_dir_str = str(self.output_dir).replace('\\', '/')
        
        config_content = f"""
dataset:
  dataset_path: "{dataset_dir_str}"
  output_dir: "{output_dir_str}"
  train_ratio: 0.7
  val_ratio: 0.2
  test_ratio: 0.1
  random_seed: 42
  overwrite_existing: false
"""
        self.config_file.write_text(config_content)
    
    def tearDown(self):
        """Clean up temporary directories."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_config_loading(self):
        """Test configuration loading from YAML."""
        from uav_survivor_detection.config.manager import ConfigurationManager
        
        manager = ConfigurationManager(self.config_file)
        config = manager.load_config()
        
        self.assertIsNotNone(config.dataset)
        self.assertEqual(config.dataset.dataset_path, self.dataset_dir)
        self.assertEqual(config.dataset.output_dir, self.output_dir)
        self.assertEqual(config.dataset.train_ratio, 0.7)
    
    def test_missing_config_file(self):
        """Test error handling for missing config file."""
        from uav_survivor_detection.config.manager import ConfigurationManager, ConfigurationError
        
        with self.assertRaises(ConfigurationError):
            ConfigurationManager(Path('nonexistent.yaml'))


class TestErrorHandling(unittest.TestCase):
    """Test error handling and exit codes."""
    
    def test_missing_dataset_section(self):
        """Test error when dataset section is missing from config."""
        from uav_survivor_detection.config.models import SystemConfig
        
        config = SystemConfig(dataset=None)
        args = Mock()
        args.dataset_path = None
        
        # Should exit with code 1
        with self.assertRaises(SystemExit) as cm:
            with patch('sys.stderr'):
                apply_cli_overrides(config, args)
        
        self.assertEqual(cm.exception.code, 1)


if __name__ == '__main__':
    unittest.main()
