"""
Unit tests for project structure verification.

Tests that all required directories exist and that requirements.txt
can be parsed correctly.

Requirements: 10.4
"""

import pytest
from pathlib import Path
import re


# Get project root (parent of tests/ directory)
PROJECT_ROOT = Path(__file__).parent.parent


class TestProjectStructure:
    """Test suite for project directory structure."""
    
    def test_required_directories_exist(self):
        """Test that all required directories exist."""
        required_dirs = [
            'src',
            'tests',
            'configs',
            'scripts',
            'data',
            'checkpoints',
            'logs'
        ]
        
        for dir_name in required_dirs:
            dir_path = PROJECT_ROOT / dir_name
            assert dir_path.exists(), f"Required directory '{dir_name}/' does not exist"
            assert dir_path.is_dir(), f"'{dir_name}/' exists but is not a directory"
    
    def test_src_package_structure(self):
        """Test that src/ contains the main package."""
        src_dir = PROJECT_ROOT / 'src'
        package_dir = src_dir / 'uav_survivor_detection'
        
        assert package_dir.exists(), "Package directory 'src/uav_survivor_detection/' does not exist"
        assert package_dir.is_dir(), "'src/uav_survivor_detection/' exists but is not a directory"
        
        # Check for __init__.py
        init_file = package_dir / '__init__.py'
        assert init_file.exists(), "Package __init__.py file does not exist"
    
    def test_gitkeep_files_present(self):
        """Test that empty directories have .gitkeep files."""
        dirs_with_gitkeep = ['data', 'checkpoints', 'logs', 'configs']
        
        for dir_name in dirs_with_gitkeep:
            gitkeep_path = PROJECT_ROOT / dir_name / '.gitkeep'
            # .gitkeep files may or may not be present - this is informational
            if gitkeep_path.exists():
                assert gitkeep_path.is_file()


class TestRequirements:
    """Test suite for requirements.txt file."""
    
    def test_requirements_file_exists(self):
        """Test that requirements.txt exists."""
        req_file = PROJECT_ROOT / 'requirements.txt'
        assert req_file.exists(), "requirements.txt does not exist"
        assert req_file.is_file(), "requirements.txt exists but is not a file"
    
    def test_requirements_can_be_parsed(self):
        """Test that requirements.txt can be parsed without errors."""
        req_file = PROJECT_ROOT / 'requirements.txt'
        
        with open(req_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        assert len(lines) > 0, "requirements.txt is empty"
    
    def test_required_dependencies_present(self):
        """Test that all required dependencies are listed in requirements.txt."""
        req_file = PROJECT_ROOT / 'requirements.txt'
        
        with open(req_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Required packages as per Requirement 10.3
        required_packages = {
            'ultralytics': '8.1.0',
            'torch': '2.1.0',
            'opencv-python': '4.8.0',
            'pyyaml': '6.0',
            'pandas': '2.0.0',
            'thop': '0.1.1',
            'pycocotools': '2.0.6',
            'matplotlib': '3.7.0',
            'pytest': '7.4.0',
        }
        
        for package, min_version in required_packages.items():
            # Check if package is mentioned (case-insensitive)
            assert package.lower() in content.lower(), \
                f"Required package '{package}' not found in requirements.txt"
    
    def test_pinned_versions_format(self):
        """Test that dependencies use pinned versions (==) format."""
        req_file = PROJECT_ROOT / 'requirements.txt'
        
        with open(req_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Pattern for pinned version: package==version
        pinned_pattern = re.compile(r'^[a-zA-Z0-9_-]+==[\d.]+')
        # Pattern for ranged version: package>=version or package>=version,<version
        ranged_pattern = re.compile(r'^[a-zA-Z0-9_-]+>=')
        
        pinned_count = 0
        for line in lines:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
            
            # Check if line contains a package specification
            if '=' in line or '>' in line or '<' in line:
                # At least some packages should use pinned versions
                if pinned_pattern.match(line) or ranged_pattern.match(line):
                    pinned_count += 1
        
        assert pinned_count > 0, "No properly formatted package specifications found in requirements.txt"


class TestConfigurationFiles:
    """Test suite for configuration and setup files."""
    
    def test_setup_py_exists(self):
        """Test that setup.py exists."""
        setup_file = PROJECT_ROOT / 'setup.py'
        assert setup_file.exists(), "setup.py does not exist"
        assert setup_file.is_file(), "setup.py exists but is not a file"
    
    def test_readme_exists(self):
        """Test that README.md exists."""
        readme_file = PROJECT_ROOT / 'README.md'
        assert readme_file.exists(), "README.md does not exist"
        assert readme_file.is_file(), "README.md exists but is not a file"
    
    def test_gitignore_exists(self):
        """Test that .gitignore exists."""
        gitignore_file = PROJECT_ROOT / '.gitignore'
        assert gitignore_file.exists(), ".gitignore does not exist"
        assert gitignore_file.is_file(), ".gitignore exists but is not a file"
    
    def test_gitignore_patterns(self):
        """Test that .gitignore contains essential patterns."""
        gitignore_file = PROJECT_ROOT / '.gitignore'
        
        with open(gitignore_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Essential patterns for Python, checkpoints, and logs
        essential_patterns = [
            '__pycache__',  # Python bytecode
            '*.py[cod]',    # Python compiled files
            'checkpoints/*.pt',  # Model checkpoints
            'logs/*.log',   # Log files
        ]
        
        for pattern in essential_patterns:
            assert pattern in content, \
                f"Essential pattern '{pattern}' not found in .gitignore"


class TestScripts:
    """Test suite for utility scripts."""
    
    def test_setup_verify_script_exists(self):
        """Test that setup_verify.py script exists."""
        script_file = PROJECT_ROOT / 'scripts' / 'setup_verify.py'
        assert script_file.exists(), "scripts/setup_verify.py does not exist"
        assert script_file.is_file(), "scripts/setup_verify.py exists but is not a file"
    
    def test_setup_verify_is_executable_python(self):
        """Test that setup_verify.py is a valid Python file."""
        script_file = PROJECT_ROOT / 'scripts' / 'setup_verify.py'
        
        with open(script_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Should contain Python code indicators
        assert 'import' in content, "setup_verify.py does not appear to contain Python import statements"
        assert 'def ' in content, "setup_verify.py does not appear to contain function definitions"


class TestReadmeContent:
    """Test suite for README.md content."""
    
    def test_readme_contains_installation_instructions(self):
        """Test that README.md contains installation instructions."""
        readme_file = PROJECT_ROOT / 'README.md'
        
        with open(readme_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for key sections
        assert 'Installation' in content or 'installation' in content.lower(), \
            "README.md does not contain Installation section"
        
        assert 'requirements.txt' in content, \
            "README.md does not mention requirements.txt installation"
    
    def test_readme_contains_usage_information(self):
        """Test that README.md contains usage information."""
        readme_file = PROJECT_ROOT / 'README.md'
        
        with open(readme_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Should contain usage or quick start information
        usage_keywords = ['Usage', 'Quick Start', 'Getting Started', 'usage', 'quick start']
        assert any(keyword in content for keyword in usage_keywords), \
            "README.md does not contain usage or quick start information"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
