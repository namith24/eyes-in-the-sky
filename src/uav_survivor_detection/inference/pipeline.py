"""Orchestration for video capture, detection, display, and logging.

Example:
    >>> from pathlib import Path
    >>> from uav_survivor_detection.config.models import InferenceConfig
    >>> from uav_survivor_detection.inference import InferencePipeline
    >>> config = InferenceConfig("0", Path("checkpoints/best.pt"))
    >>> pipeline = InferencePipeline(config)
    >>> # pipeline.run() starts the camera loop and blocks until shutdown.
"""

import signal
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from ..config.models import InferenceConfig, LoggerConfig
from .detection_engine import Detection, DetectionEngine, DetectionEngineConfig
from .display import DisplayModule
from .logger import Logger


class InferencePipeline:
    """Run the real-time inference loop and coordinate its components."""

    def __init__(
        self,
        config: InferenceConfig,
        detection_engine: Optional[DetectionEngine] = None,
        display: Optional[DisplayModule] = None,
        logger: Optional[Logger] = None,
        capture: Optional[Any] = None,
    ) -> None:
        self.config = config
        self.detection_engine = detection_engine or DetectionEngine(
            DetectionEngineConfig(
                checkpoint_path=config.model_checkpoint,
                conf_threshold=config.conf_threshold,
                iou_threshold=config.iou_threshold,
                img_size=config.img_size,
                device=config.device,
            )
        )
        self.display = display or DisplayModule()
        self.logger = logger or Logger(LoggerConfig(output_dir=config.log_output_dir))
        self.capture = capture
        self.frame_buffer: Deque[Tuple[float, np.ndarray]] = deque()
        self._frame_times: Deque[float] = deque(maxlen=30)
        self.frame_index = 0
        self._running = False
        self._shutdown_complete = False
        self._signal_handlers: Dict[int, Any] = {}

    def connect_video_source(self) -> Any:
        """Open the configured video source, retrying according to configuration."""
        if self.capture is not None and self._capture_is_open(self.capture):
            return self.capture

        source = self._parse_video_source(self.config.video_source)
        for attempt in range(self.config.max_reconnect_attempts):
            capture = cv2.VideoCapture(source)
            if self._capture_is_open(capture):
                self.capture = capture
                return capture
            capture.release()
            if attempt + 1 < self.config.max_reconnect_attempts:
                time.sleep(self.config.reconnect_interval_seconds)

        raise RuntimeError(
            f"Unable to connect to video source after "
            f"{self.config.max_reconnect_attempts} attempts: {self.config.video_source}"
        )

    def process_frame(self, frame: np.ndarray) -> List[Detection]:
        """Run detection on one frame and return the resulting detections."""
        return self.detection_engine.predict(frame)

    def handle_disconnection(self) -> bool:
        """Release a failed capture and attempt to reconnect."""
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        try:
            self.connect_video_source()
        except RuntimeError:
            return False
        return True

    def run(self) -> None:
        """Run until the source ends, the user quits, or reconnection fails."""
        if self.detection_engine.model is None:
            self.detection_engine.load_model()
        self.connect_video_source()
        self._register_signal_handlers()
        if self.config.logging_enabled:
            self.logger.start_session(self.config.video_source)
        self._running = True
        self._shutdown_complete = False

        try:
            while self._running:
                assert self.capture is not None
                success, frame = self.capture.read()
                if not success or frame is None:
                    if not self._is_reconnectable_source(self.config.video_source):
                        break
                    if not self.handle_disconnection():
                        break
                    continue

                now = time.perf_counter()
                self._buffer_frame(now, frame)
                self._frame_times.append(now)
                fps = self._current_fps()
                detections = self.process_frame(frame)

                if self.config.logging_enabled:
                    for detection in detections:
                        self.logger.log_detection(
                            self.frame_index,
                            detection,
                            self.config.video_source,
                        )
                    if not detections:
                        self.logger.log_no_detection(
                            self.frame_index,
                            self.config.video_source,
                        )
                    self.logger.record_fps(fps)

                self.frame_index += 1

                if self.config.display_enabled:
                    rendered = self.display.render_frame(frame, detections, fps=fps)
                    if self.display.show_frame(rendered):
                        break
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Stop processing and release capture, logging, signals, and windows."""
        if self._shutdown_complete:
            return
        self._running = False
        if self.logger is not None:
            self.logger.end_session(mean_fps=self._current_fps())
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        if self.config.display_enabled:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
        self._restore_signal_handlers()
        self._shutdown_complete = True

    def _buffer_frame(self, timestamp: float, frame: np.ndarray) -> None:
        self.frame_buffer.append((timestamp, frame))
        cutoff = timestamp - self.config.max_frame_buffer_seconds
        while self.frame_buffer and self.frame_buffer[0][0] < cutoff:
            self.frame_buffer.popleft()

    def _current_fps(self) -> float:
        if len(self._frame_times) < 2:
            return 0.0
        elapsed = self._frame_times[-1] - self._frame_times[0]
        return (len(self._frame_times) - 1) / elapsed if elapsed > 0 else 0.0

    def _register_signal_handlers(self) -> None:
        for signum in (signal.SIGINT, signal.SIGTERM):
            self._signal_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, self._handle_signal)

    def _restore_signal_handlers(self) -> None:
        for signum, handler in self._signal_handlers.items():
            signal.signal(signum, handler)
        self._signal_handlers.clear()

    def _handle_signal(self, signum: int, frame: Any) -> None:
        self.shutdown()

    @staticmethod
    def _parse_video_source(source: Union[int, str]) -> Union[int, str]:
        if isinstance(source, int):
            return source
        source_text = str(source).strip()
        if source_text.isdigit():
            return int(source_text)
        return source_text

    @staticmethod
    def _is_reconnectable_source(source: str) -> bool:
        source_text = str(source).strip().lower()
        # http(s):// covers ESP32-CAM MJPEG streams, which drop often over WiFi.
        return source_text.isdigit() or source_text.startswith(
            ("rtsp://", "udp://", "http://", "https://")
        )

    @staticmethod
    def _capture_is_open(capture: Any) -> bool:
        is_open = getattr(capture, "isOpened", None)
        return bool(is_open()) if callable(is_open) else capture is not None