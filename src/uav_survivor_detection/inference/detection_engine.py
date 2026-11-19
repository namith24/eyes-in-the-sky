"""YOLO-backed detection engine for single-frame inference."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np
import torch
from ultralytics import YOLO

from ..config.models import DetectionEngineConfig


@dataclass
class BoundingBox:
    """Pixel coordinates for a detected object."""

    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class Detection:
    """A detected object and its classification metadata."""

    bbox: BoundingBox
    confidence: float
    class_id: int
    class_name: Optional[str] = None


class DetectionEngine:
    """Load a YOLO checkpoint and convert predictions into project objects."""

    def __init__(
        self,
        config: Optional[DetectionEngineConfig] = None,
        model: Optional[Any] = None,
    ) -> None:
        self.config = config or DetectionEngineConfig()
        self.model = model

        if self.config.num_threads > 0:
            torch.set_num_threads(self.config.num_threads)

    def load_model(self, checkpoint_path: Optional[Union[str, Path]] = None) -> Any:
        """Load and retain a YOLO checkpoint.

        Args:
            checkpoint_path: Optional override for the configured checkpoint.

        Raises:
            ValueError: If neither an override nor a configured checkpoint exists.
        """
        checkpoint = checkpoint_path or self.config.checkpoint_path
        if checkpoint is None:
            raise ValueError("A YOLO checkpoint path must be configured before loading")

        self.model = YOLO(str(checkpoint))
        return self.model

    def predict(self, frame: np.ndarray) -> List[Detection]:
        """Run inference on one OpenCV frame and return converted detections."""
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call load_model() first.")
        if not isinstance(frame, np.ndarray):
            raise TypeError("frame must be a numpy.ndarray")
        if frame.ndim not in (2, 3):
            raise ValueError("frame must be a 2D or 3D image array")

        results = self.model.predict(
            source=frame,
            conf=self.config.conf_threshold,
            iou=self.config.iou_threshold,
            imgsz=self.config.img_size,
            device=self.config.device,
            verbose=False,
        )
        return self._extract_detections(results)

    def warmup(self, iterations: int = 1) -> None:
        """Run dummy predictions to initialize the model before live inference."""
        if iterations < 0:
            raise ValueError("iterations must be non-negative")
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        dummy_frame = np.zeros(
            (self.config.img_size, self.config.img_size, 3), dtype=np.uint8
        )
        for _ in range(iterations):
            self.predict(dummy_frame)

    @staticmethod
    def _extract_detections(results: Any) -> List[Detection]:
        """Convert the first Ultralytics result into serializable detections."""
        if results is None:
            return []

        if isinstance(results, (list, tuple)):
            if not results:
                return []
            result = results[0]
        else:
            result = results
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        xyxy = boxes.xyxy
        confidences = boxes.conf
        class_ids = boxes.cls
        names = getattr(result, "names", {}) or {}
        detections: List[Detection] = []

        for coordinates, confidence, class_id in zip(xyxy, confidences, class_ids):
            values = coordinates.tolist() if hasattr(coordinates, "tolist") else coordinates
            confidence_value = (
                confidence.item() if hasattr(confidence, "item") else float(confidence)
            )
            class_id_value = class_id.item() if hasattr(class_id, "item") else int(class_id)
            class_name = names.get(class_id_value) if isinstance(names, dict) else None
            detections.append(
                Detection(
                    bbox=BoundingBox(*(float(value) for value in values[:4])),
                    confidence=float(confidence_value),
                    class_id=int(class_id_value),
                    class_name=class_name,
                )
            )

        return detections