"""
Unit tests for setup verification script

Requirements: 10.4
"""

import sys
import os
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

import setup_verify


class TestSetupVerify:
    """Test setup verification script functionality."""
    
    def test_check_python_version(self):
        """Test Python version checking."""
        # Should pass with Python 3.8+
        result = setup_verify.check_python_version()
        assert isinstance(result, bool)
        # Should return True for Python 3.8+
        if sys.version_info.major == 3 and sys.version_info.minor >= 8:
            assert result is True
    
    def test_check_directory_structure(self):
        """Test directory structure verification."""
        result = setup_verify.check_directory_structure()
        assert isinstance(result, bool)
        # All required directories should exist (they were created in task 1.1)
        assert result is True
    
    def test_check_torch_capabilities(self):
        """Test PyTorch capabilities check."""
        try:
            import torch
            capabilities = setup_verify.check_torch_capabilities()
            
            # Should return a dictionary
            assert isinstance(capabilities, dict)
            
            # Should contain GPU availability info
            assert 'gpu_available' in capabilities
            assert isinstance(capabilities['gpu_available'], bool)
            
            # Should contain CPU thread info
            if 'cpu_threads' in capabilities:
                assert isinstance(capabilities['cpu_threads'], int)
                assert capabilities['cpu_threads'] > 0
                
        except ImportError:
            # If torch not installed, should return empty dict
            capabilities = setup_verify.check_torch_capabilities()
            assert capabilities == {}
    
    def test_test_opencv(self):
        """Test OpenCV functionality check."""
        try:
            import cv2
            result = setup_verify.test_opencv()
            # Should return True if OpenCV is working
            assert result is True
        except ImportError:
            # If cv2 not installed, should return False
            result = setup_verify.test_opencv()
            assert result is False
    
    def test_print_functions(self):
        """Test print utility functions don't crash."""
        # These should not raise exceptions
        setup_verify.print_header("Test Header")
        setup_verify.print_success("Test Success")
        setup_verify.print_error("Test Error")
        setup_verify.print_warning("Test Warning")
        setup_verify.print_info("Test Info")
    
    def test_check_dependencies_structure(self):
        """Test dependency check returns correct structure."""
        all_installed, missing_packages = setup_verify.check_dependencies()
        
        # Should return tuple with bool and list
        assert isinstance(all_installed, bool)
        assert isinstance(missing_packages, list)
        
        # If not all installed, missing_packages should not be empty
        if not all_installed:
            assert len(missing_packages) > 0
    
    @patch('setup_verify.check_python_version')
    @patch('setup_verify.check_dependencies')
    @patch('setup_verify.check_directory_structure')
    @patch('setup_verify.check_torch_capabilities')
    @patch('setup_verify.test_opencv')
    @patch('setup_verify.test_yolo_model_loading')
    @patch('setup_verify.test_image_processing')
    def test_main_returns_exit_code(
        self,
        mock_image_proc,
        mock_yolo,
        mock_opencv,
        mock_torch,
        mock_dirs,
        mock_deps,
        mock_python
    ):
        """Test main function returns appropriate exit code."""
        # Mock all checks to pass
        mock_python.return_value = True
        mock_deps.return_value = (True, [])
        mock_dirs.return_value = True
        mock_torch.return_value = {'gpu_available': False, 'cpu_threads': 4}
        mock_opencv.return_value = True
        mock_yolo.return_value = True
        mock_image_proc.return_value = True
        
        exit_code = setup_verify.main()
        
        # All checks passed, should return 0
        assert exit_code == 0
    
    @patch('setup_verify.check_python_version')
    @patch('setup_verify.check_dependencies')
    @patch('setup_verify.check_directory_structure')
    @patch('setup_verify.check_torch_capabilities')
    @patch('setup_verify.test_opencv')
    @patch('setup_verify.test_yolo_model_loading')
    @patch('setup_verify.test_image_processing')
    def test_main_returns_failure_code(
        self,
        mock_image_proc,
        mock_yolo,
        mock_opencv,
        mock_torch,
        mock_dirs,
        mock_deps,
        mock_python
    ):
        """Test main function returns failure code when checks fail."""
        # Mock some checks to fail
        mock_python.return_value = False
        mock_deps.return_value = (False, ['ultralytics'])
        mock_dirs.return_value = True
        mock_torch.return_value = {}
        mock_opencv.return_value = True
        mock_yolo.return_value = False
        mock_image_proc.return_value = False
        
        exit_code = setup_verify.main()
        
        # Some checks failed, should return 1
        assert exit_code == 1


class TestReportGeneration:
    """Test report generation functionality."""
    
    def test_generate_report_all_pass(self):
        """Test report generation when all checks pass."""
        results = {
            'Python Version': True,
            'Dependencies': True,
            'Directory Structure': True,
            'PyTorch': True,
            'OpenCV': True,
            'YOLO Model Loading': True,
            'Image Processing': True
        }
        
        exit_code = setup_verify.generate_report(results)
        assert exit_code == 0
    
    def test_generate_report_some_fail(self):
        """Test report generation when some checks fail."""
        results = {
            'Python Version': True,
            'Dependencies': False,
            'Directory Structure': True,
            'PyTorch': False,
            'OpenCV': True,
            'YOLO Model Loading': False,
            'Image Processing': False
        }
        
        exit_code = setup_verify.generate_report(results)
        assert exit_code == 1
    
    def test_generate_report_all_fail(self):
        """Test report generation when all checks fail."""
        results = {
            'Python Version': False,
            'Dependencies': False,
            'Directory Structure': False,
            'PyTorch': False,
            'OpenCV': False,
            'YOLO Model Loading': False,
            'Image Processing': False
        }
        
        exit_code = setup_verify.generate_report(results)
        assert exit_code == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
