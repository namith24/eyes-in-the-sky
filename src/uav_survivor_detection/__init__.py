"""
UAV Survivor Detection System

A ground-side UAV survivor detection system that processes live video streams
from quadcopter UAVs using YOLO11n-based human detection on standard laptop
CPU hardware.

This package provides modules for:
- Dataset preparation and splitting
- Model training with YOLO11n baseline
- Small-object-aware model enhancements
- Ablation studies for enhancement evaluation
- Comprehensive model evaluation and metrics
- Real-time inference pipeline with video processing
- Detection engine with CPU-optimized YOLO inference
- Display module with annotated frame rendering
- Detection logging with CSV and JSON outputs
- Configuration management with YAML and CLI support

For more information, see the README.md file in the project root.
"""

__version__ = "0.1.0"
__author__ = "Capstone Software Team"
__license__ = "MIT"

# Package metadata
__all__ = [
    "__version__",
    "__author__",
    "__license__",
]
