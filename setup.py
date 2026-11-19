"""
UAV Survivor Detection System
Setup configuration for package installation
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="uav_survivor_detection",
    version="0.1.0",
    author="Capstone Software Team",
    description="Ground-side UAV survivor detection system using YOLO11n for disaster search-and-rescue",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/uav-survivor-detection",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "ultralytics>=8.1.0",
        "opencv-python>=4.8.0",
        "pyyaml>=6.0",
        "pandas>=2.0.0",
        "thop>=0.1.1",
        "pycocotools>=2.0.6",
        "matplotlib>=3.7.0",
        "pytest>=7.4.0",
        "numpy>=1.24.0,<2.0.0",
        "pillow>=10.0.0",
        "tqdm>=4.65.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="uav, survivor-detection, yolo, computer-vision, disaster-response, search-and-rescue",
    entry_points={
        "console_scripts": [
            "uav-detect=uav_survivor_detection.inference_pipeline:main",
            "uav-train=uav_survivor_detection.training_pipeline:main",
            "uav-evaluate=uav_survivor_detection.evaluator:main",
        ],
    },
)
