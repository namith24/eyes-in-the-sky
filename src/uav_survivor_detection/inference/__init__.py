"""Real-time inference components."""

from .detection_engine import (
    BoundingBox,
    Detection,
    DetectionEngine,
    DetectionEngineConfig,
)
from .display import DisplayModule
from .drone_link import DroneLink, LinkStats, SourceSpec, discover, list_windows, parse_source
from .logger import Logger, SessionMetadata
from .pipeline import InferencePipeline, InferenceConfig

__all__ = [
    "BoundingBox",
    "Detection",
    "DetectionEngine",
    "DetectionEngineConfig",
    "DisplayModule",
    "DroneLink",
    "LinkStats",
    "SourceSpec",
    "discover",
    "list_windows",
    "parse_source",
    "Logger",
    "SessionMetadata",
    "InferencePipeline",
    "InferenceConfig",
]