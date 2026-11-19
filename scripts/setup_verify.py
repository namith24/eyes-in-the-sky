"""
Setup Verification Script for UAV Survivor Detection System

This script verifies that all dependencies are correctly installed and that
the Detection Engine can load a sample YOLO model and process test images.
Reports CPU/GPU availability and thread count.

Requirements: 10.4
"""

import sys
import os
from pathlib import Path
import subprocess
from typing import Tuple, List, Dict

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'


def print_header(text: str) -> None:
    """Print formatted section header."""
    print(f"\n{BLUE}{'=' * 70}{RESET}")
    print(f"{BLUE}{text:^70}{RESET}")
    print(f"{BLUE}{'=' * 70}{RESET}\n")


def print_success(text: str) -> None:
    """Print success message."""
    print(f"{GREEN}✓ {text}{RESET}")


def print_error(text: str) -> None:
    """Print error message."""
    print(f"{RED}✗ {text}{RESET}")


def print_warning(text: str) -> None:
    """Print warning message."""
    print(f"{YELLOW}⚠ {text}{RESET}")


def print_info(text: str) -> None:
    """Print informational message."""
    print(f"  {text}")


def check_python_version() -> bool:
    """Check if Python version meets minimum requirements."""
    print_header("Python Version Check")
    
    version_info = sys.version_info
    version_str = f"{version_info.major}.{version_info.minor}.{version_info.micro}"
    
    if version_info.major == 3 and version_info.minor >= 8:
        print_success(f"Python {version_str} detected")
        return True
    else:
        print_error(f"Python {version_str} detected (requires Python 3.8+)")
        return False


def check_dependencies() -> Tuple[bool, List[str]]:
    """Check if all required dependencies are installed."""
    print_header("Dependency Check")
    
    required_packages = {
        'torch': '2.1.0',
        'torchvision': '0.16.0',
        'ultralytics': '8.1.0',
        'cv2': '4.8.0',  # opencv-python
        'yaml': '6.0',  # pyyaml
        'pandas': '2.0.0',
        'thop': '0.1.1',
        'pycocotools': '2.0.6',
        'matplotlib': '3.7.0',
        'pytest': '7.4.0',
        'numpy': '1.24.0',
        'PIL': '10.0.0',  # pillow
        'tqdm': '4.65.0',
    }
    
    all_installed = True
    missing_packages = []
    
    for package_name, min_version in required_packages.items():
        try:
            if package_name == 'cv2':
                import cv2
                version = cv2.__version__
                display_name = 'opencv-python'
            elif package_name == 'yaml':
                import yaml
                version = yaml.__version__
                display_name = 'pyyaml'
            elif package_name == 'PIL':
                from PIL import Image
                import PIL
                version = PIL.__version__
                display_name = 'pillow'
            else:
                module = __import__(package_name)
                version = module.__version__
                display_name = package_name
            
            print_success(f"{display_name} {version} installed")
        except ImportError:
            print_error(f"{package_name} not installed")
            missing_packages.append(package_name)
            all_installed = False
        except AttributeError:
            print_warning(f"{package_name} installed but version cannot be determined")
    
    return all_installed, missing_packages


def check_torch_capabilities() -> Dict[str, any]:
    """Check PyTorch capabilities including CPU/GPU availability."""
    print_header("PyTorch Capabilities")
    
    capabilities = {}
    
    try:
        import torch
        
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
            print_success(f"GPU available: {gpu_name} (CUDA {torch.version.cuda})")
            print_info(f"GPU count: {gpu_count}")
            capabilities['gpu_available'] = True
            capabilities['gpu_count'] = gpu_count
            capabilities['gpu_name'] = gpu_name
        else:
            print_info("GPU not available (CPU-only mode)")
            print_info("This is acceptable - the system is designed for CPU inference")
            capabilities['gpu_available'] = False
        
        # Check CPU thread count
        num_threads = torch.get_num_threads()
        num_interop_threads = torch.get_num_interop_threads()
        print_success(f"CPU threads: {num_threads} (interop: {num_interop_threads})")
        capabilities['cpu_threads'] = num_threads
        capabilities['interop_threads'] = num_interop_threads
        
        # Check MKL/OpenMP
        try:
            mkl_available = torch.backends.mkldnn.is_available()
            if mkl_available:
                print_success("Intel MKL-DNN optimization available")
                capabilities['mkl_available'] = True
            else:
                print_info("Intel MKL-DNN not available")
                capabilities['mkl_available'] = False
        except:
            capabilities['mkl_available'] = False
        
        return capabilities
        
    except ImportError:
        print_error("PyTorch not installed - cannot check capabilities")
        return {}


def test_yolo_model_loading() -> bool:
    """Test that a YOLO model can be loaded."""
    print_header("YOLO Model Loading Test")
    
    try:
        from ultralytics import YOLO
        import torch
        
        print_info("Attempting to load YOLO11n pretrained weights...")
        
        # This will download the model if not already cached
        model = YOLO('yolo11n.pt')
        
        print_success("YOLO11n model loaded successfully")
        
        # Get model info
        try:
            params = sum(p.numel() for p in model.model.parameters())
            print_info(f"Model parameters: {params:,}")
        except:
            pass
        
        return True
        
    except Exception as e:
        print_error(f"Failed to load YOLO model: {str(e)}")
        print_info("This may be a network issue - the model needs to be downloaded on first run")
        return False


def test_image_processing() -> bool:
    """Test that a sample image can be processed without error."""
    print_header("Image Processing Test")
    
    try:
        from ultralytics import YOLO
        import torch
        import numpy as np
        
        print_info("Creating test image (640x480 RGB)...")
        
        # Create a synthetic test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        print_info("Loading YOLO11n model...")
        model = YOLO('yolo11n.pt')
        
        print_info("Running inference on test image...")
        
        # Run inference
        results = model.predict(test_image, verbose=False)
        
        print_success("Image processing successful")
        print_info(f"Inference completed - {len(results)} result(s) returned")
        
        return True
        
    except Exception as e:
        print_error(f"Image processing failed: {str(e)}")
        return False


def check_directory_structure() -> bool:
    """Verify that all required directories exist."""
    print_header("Directory Structure Check")
    
    project_root = Path(__file__).parent.parent
    required_dirs = [
        'src',
        'tests',
        'configs',
        'scripts',
        'data',
        'checkpoints',
        'logs'
    ]
    
    all_exist = True
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists() and dir_path.is_dir():
            print_success(f"{dir_name}/ directory exists")
        else:
            print_error(f"{dir_name}/ directory missing")
            all_exist = False
    
    return all_exist


def test_opencv() -> bool:
    """Test OpenCV functionality."""
    print_header("OpenCV Test")
    
    try:
        import cv2
        import numpy as np
        
        # Create test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Test resize
        resized = cv2.resize(test_image, (320, 240))
        print_success("cv2.resize() works")
        
        # Test drawing functions
        test_image_copy = test_image.copy()
        cv2.rectangle(test_image_copy, (10, 10), (100, 100), (0, 255, 0), 2)
        cv2.putText(test_image_copy, "Test", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        print_success("cv2 drawing functions work")
        
        return True
        
    except Exception as e:
        print_error(f"OpenCV test failed: {str(e)}")
        return False


def generate_report(results: Dict[str, bool]) -> None:
    """Generate final verification report."""
    print_header("Verification Summary")
    
    passed_tests = sum(results.values())
    total_tests = len(results)
    
    for test_name, passed in results.items():
        if passed:
            print_success(test_name)
        else:
            print_error(test_name)
    
    print(f"\n{BLUE}{'─' * 70}{RESET}\n")
    
    if passed_tests == total_tests:
        print(f"{GREEN}✓ All checks passed ({passed_tests}/{total_tests}){RESET}")
        print(f"{GREEN}System is ready for use!{RESET}\n")
        return 0
    else:
        print(f"{RED}✗ Some checks failed ({passed_tests}/{total_tests} passed){RESET}")
        print(f"{YELLOW}Please resolve the issues above before using the system.{RESET}\n")
        return 1


def main() -> int:
    """Main verification routine."""
    print(f"\n{BLUE}{'=' * 70}{RESET}")
    print(f"{BLUE}UAV Survivor Detection System - Setup Verification{RESET}")
    print(f"{BLUE}{'=' * 70}{RESET}\n")
    
    results = {}
    
    # Run all verification checks
    results['Python Version'] = check_python_version()
    
    deps_ok, missing = check_dependencies()
    results['Dependencies'] = deps_ok
    
    if not deps_ok:
        print_warning(f"\nMissing packages: {', '.join(missing)}")
        print_info("Install with: pip install -r requirements.txt\n")
    
    results['Directory Structure'] = check_directory_structure()
    
    torch_caps = check_torch_capabilities()
    results['PyTorch'] = bool(torch_caps)
    
    results['OpenCV'] = test_opencv()
    
    results['YOLO Model Loading'] = test_yolo_model_loading()
    
    results['Image Processing'] = test_image_processing()
    
    # Generate final report
    exit_code = generate_report(results)
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
