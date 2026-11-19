"""
Model enhancement modules for UAV survivor detection.

This package contains attention mechanisms and model enhancement utilities
for augmenting the YOLO11n baseline architecture to improve small-object detection
in UAV aerial imagery.
"""

from .attention import (
    GAM,
    CBAM,
    ChannelAttention,
    SpatialAttention,
    C3k2_ARC,
    CLConcat,
)

from .enhancement import (
    EnhancementModule,
    EnhancementConfig,
    ModelComplexity,
)

__all__ = [
    "GAM",
    "CBAM",
    "ChannelAttention",
    "SpatialAttention",
    "C3k2_ARC",
    "CLConcat",
    "EnhancementModule",
    "EnhancementConfig",
    "ModelComplexity",
]
