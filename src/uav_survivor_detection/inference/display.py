"""OpenCV rendering for annotated UAV detection frames."""

from datetime import datetime, timezone
from typing import Dict, Iterable, Optional, Tuple, Union

import cv2
import numpy as np

from ..config.models import DisplayConfig
from .detection_engine import Detection


class DisplayModule:
    """Render detections and status information on OpenCV frames."""

    def __init__(self, config: Optional[DisplayConfig] = None) -> None:
        self.config = config or DisplayConfig()
        self.bbox_enabled = self.config.bbox_enabled
        self._colors: Dict[Union[int, str], Tuple[int, int, int]] = {}

        if self.config.color_scheme:
            self._colors.update(self.config.color_scheme)

    def render_frame(
        self,
        frame: np.ndarray,
        detections: Iterable[Detection],
        fps: Optional[float] = None,
        timestamp: Optional[datetime] = None,
    ) -> np.ndarray:
        """Draw detections and a translucent FPS/timestamp overlay on a frame."""
        if not isinstance(frame, np.ndarray) or frame.ndim not in (2, 3):
            raise TypeError("frame must be a 2D or 3D numpy.ndarray")

        rendered = frame.copy()
        if rendered.ndim == 2:
            rendered = cv2.cvtColor(rendered, cv2.COLOR_GRAY2BGR)

        if self.bbox_enabled:
            for detection in detections:
                class_key = (
                    detection.class_name
                    if detection.class_name is not None
                    else detection.class_id
                )
                color = self._color_for(class_key)
                bbox = detection.bbox
                top_left = (round(bbox.x1), round(bbox.y1))
                bottom_right = (round(bbox.x2), round(bbox.y2))
                cv2.rectangle(
                    rendered,
                    top_left,
                    bottom_right,
                    color,
                    self.config.bbox_thickness,
                )

                label = detection.class_name or str(detection.class_id)
                label = f"{label} {detection.confidence:.2f}"
                label_origin = (top_left[0], max(15, top_left[1] - 6))
                cv2.putText(
                    rendered,
                    label,
                    label_origin,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.config.font_scale,
                    color,
                    self.config.font_thickness,
                    cv2.LINE_AA,
                )

        current_time = timestamp or datetime.now(timezone.utc)
        timestamp_text = current_time.astimezone(timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )
        fps_text = f"FPS: {fps:.1f}" if fps is not None else "FPS: --"
        self._draw_status_overlay(rendered, f"{fps_text}  {timestamp_text}")
        return rendered

    def show_frame(self, frame: np.ndarray) -> bool:
        """Show a frame and return whether the configured quit key was pressed."""
        cv2.imshow(self.config.window_name, frame)
        return self.handle_keyboard_input()

    def toggle_bounding_boxes(self, enabled: Optional[bool] = None) -> bool:
        """Enable or disable bounding-box rendering and return the new state."""
        self.bbox_enabled = not self.bbox_enabled if enabled is None else enabled
        return self.bbox_enabled

    def handle_keyboard_input(self, key: Optional[int] = None) -> bool:
        """Process one key code and return whether the application should quit."""
        if key is None:
            key = cv2.waitKey(1)
        key_code = key & 0xFF if isinstance(key, int) else ord(key)

        if key_code == ord(self.config.toggle_bbox_key):
            self.toggle_bounding_boxes()
        return key_code == ord(self.config.quit_key)

    def _color_for(self, class_key: Union[int, str]) -> Tuple[int, int, int]:
        if class_key not in self._colors:
            class_index = len(self._colors)
            hue = int((class_index * 179 / max(1, class_index + 1)) % 180)
            hsv_color = np.uint8([[[hue, 220, 230]]])
            bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0, 0]
            self._colors[class_key] = tuple(int(value) for value in bgr_color)
        return self._colors[class_key]

    @staticmethod
    def _draw_status_overlay(frame: np.ndarray, text: str) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, scale, thickness
        )
        padding = 6
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (0, 0),
            (text_width + padding * 2, text_height + baseline + padding * 2),
            (0, 0, 0),
            cv2.FILLED,
        )
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        cv2.putText(
            frame,
            text,
            (padding, text_height + padding),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )