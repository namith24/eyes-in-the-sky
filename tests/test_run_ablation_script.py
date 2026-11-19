"""Unit tests for run_ablation.py CLI script.

**Validates: Requirements 4.1, 4.4**
"""

import sys
import subprocess
from pathlib import Path
import pytest

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_ablation.py"
CONFIG_PATH = PROJECT_ROOT / "configs" / "ablation_config.yaml"


def test_script_exists():
    """Verify run_ablation.py script exists."""
    assert SCRIPT_PATH.exists(), f"Script not found: {SCRIPT_PATH}"
    assert SCRIPT_PATH.is_file(), f"Script is not a file: {SCRIPT_PATH}"


def test_script_is_executable():
    """Verify script has proper shebang and can be executed."""
    with open(SCRIPT_PATH, 'r') as f:
        first_line = f.readline()
    
    assert first_line.startswith("#!/usr/bin/env python"), \
        f"Script missing proper shebang: {first_line}"


def test_help_message():
    """Verify --help flag displays usage information."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--help"],
        capture_output=True,
        text=True
    )
    
    assert result.returncode == 0, "Script failed with --help flag"
    assert "ablation study" in result.stdout.lower(), "Help message missing ablation study description"
    assert "--config" in result.stdout, "Help message missing --config argument"
    assert "--output-dir" in result.stdout, "Help message missing --output-dir argument"
    assert "--device" in result.stdout, "Help message missing --device argument"


def test_missing_config_argument():
    """Verify script requires --config argument."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH)],
        capture_output=True,
        text=True
    )
    
    assert result.returncode != 0, "Script should fail without --config argument"
    assert "required" in result.stderr.lower() or "error" in result.stderr.lower(), \
        "Error message should mention missing required argument"


def test_nonexistent_config_file():
    """Verify script handles nonexistent config file gracefully."""
    nonexistent_config = PROJECT_ROOT / "configs" / "nonexistent.yaml"
    
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--config", str(nonexistent_config)],
        capture_output=True,
        text=True
    )
    
    assert result.returncode != 0, "Script should fail with nonexistent config file"
    assert "not found" in result.stdout.lower() or "not found" in result.stderr.lower(), \
        "Error message should mention file not found"


def test_import_ablation_runner():
    """Verify script can import AblationRunner."""
    # Add src to path
    src_path = PROJECT_ROOT / "src"
    sys.path.insert(0, str(src_path))
    
    try:
        from uav_survivor_detection.ablation.runner import AblationRunner
        assert AblationRunner is not None, "AblationRunner import failed"
    except ImportError as e:
        pytest.fail(f"Failed to import AblationRunner: {e}")


def test_import_configuration_manager():
    """Verify script can import ConfigurationManager."""
    # Add src to path
    src_path = PROJECT_ROOT / "src"
    sys.path.insert(0, str(src_path))
    
    try:
        from uav_survivor_detection.config.manager import ConfigurationManager
        assert ConfigurationManager is not None, "ConfigurationManager import failed"
    except ImportError as e:
        pytest.fail(f"Failed to import ConfigurationManager: {e}")


def test_parse_args_function_exists():
    """Verify parse_args() function is defined in script."""
    with open(SCRIPT_PATH, 'r') as f:
        content = f.read()
    
    assert "def parse_args():" in content, "parse_args() function not found"
    assert "argparse.ArgumentParser" in content, "ArgumentParser not used"


def test_main_function_exists():
    """Verify main() function is defined in script."""
    with open(SCRIPT_PATH, 'r') as f:
        content = f.read()
    
    assert "def main():" in content, "main() function not found"
    assert 'if __name__ == "__main__":' in content, "Script missing main guard"


def test_cli_overrides_in_help():
    """Verify CLI override options are documented in help."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--help"],
        capture_output=True,
        text=True
    )
    
    assert result.returncode == 0
    
    # Check for key CLI overrides
    assert "--output-dir" in result.stdout, "Missing --output-dir override"
    assert "--device" in result.stdout, "Missing --device override"
    assert "--test-subset" in result.stdout or "--dataset-yaml" in result.stdout, \
        "Missing dataset-related overrides"


def test_error_handling_in_main():
    """Verify main() has proper error handling."""
    with open(SCRIPT_PATH, 'r') as f:
        content = f.read()
    
    # Check for try-except blocks
    assert "try:" in content, "main() missing try block"
    assert "except" in content, "main() missing except block"
    assert "KeyboardInterrupt" in content, "main() missing KeyboardInterrupt handler"


def test_script_docstring():
    """Verify script has comprehensive docstring."""
    with open(SCRIPT_PATH, 'r') as f:
        lines = f.readlines()
    
    # Find docstring (after shebang)
    docstring_start = None
    for i, line in enumerate(lines):
        if '"""' in line and not line.startswith("#"):
            docstring_start = i
            break
    
    assert docstring_start is not None, "Script missing docstring"
    
    # Extract docstring
    docstring_lines = []
    in_docstring = False
    for line in lines[docstring_start:]:
        if '"""' in line:
            if in_docstring:
                docstring_lines.append(line)
                break
            else:
                in_docstring = True
                docstring_lines.append(line)
        elif in_docstring:
            docstring_lines.append(line)
    
    docstring = "".join(docstring_lines)
    
    # Check docstring content
    assert "ablation" in docstring.lower(), "Docstring missing ablation description"
    assert "Usage:" in docstring, "Docstring missing usage examples"
    assert "Requirements:" in docstring or "Validates:" in docstring, \
        "Docstring missing requirements validation"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
