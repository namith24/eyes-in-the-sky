"""Persistent CSV and JSON logging for real-time detections."""

import csv
import json
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Iterable, List, Optional, Union

from ..config.models import LoggerConfig
from .detection_engine import Detection


@dataclass
class SessionMetadata:
    """Summary written when a logging session ends."""

    session_id: str
    start_time: str
    end_time: Optional[str] = None
    video_source: Optional[str] = None
    total_frames: int = 0
    total_detections: int = 0
    mean_fps: float = 0.0


class Logger:
    """Write detection records to a timestamped CSV and session JSON file."""

    CSV_FIELDS = [
        "timestamp",
        "frame_index",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "class_id",
        "class_name",
        "confidence",
        "video_source",
    ]

    def __init__(self, config: Optional[LoggerConfig] = None) -> None:
        self.config = config or LoggerConfig()
        self.csv_path: Optional[Path] = None
        self.json_path: Optional[Path] = None
        self.metadata: Optional[SessionMetadata] = None
        self._csv_file: Optional[IO[str]] = None
        self._csv_writer: Optional[csv.DictWriter] = None
        self._timer: Optional[threading.Timer] = None
        self._lock = threading.RLock()
        self._frame_indexes = set()
        self._fps_samples: List[float] = []

    def start_session(self, video_source: Optional[str] = None) -> SessionMetadata:
        """Create timestamped log files and start periodic flushing."""
        with self._lock:
            if self._csv_file is not None:
                raise RuntimeError("A logging session is already active")

            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            now = datetime.now(timezone.utc)
            session_id = now.strftime("%Y%m%d_%H%M%S")
            self.csv_path = output_dir / f"detections_{session_id}.csv"
            self.json_path = output_dir / f"session_{session_id}.json"
            self._csv_file = self.csv_path.open("w", newline="", encoding="utf-8")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self.CSV_FIELDS)
            self._csv_writer.writeheader()
            self.metadata = SessionMetadata(
                session_id=session_id,
                start_time=now.isoformat(),
                video_source=video_source,
            )
            self._frame_indexes.clear()
            self._fps_samples.clear()
            self._write_metadata()
            self._schedule_flush()
            return self.metadata

    def log_detection(
        self,
        frame_index: int,
        detection: Detection,
        video_source: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Append one detection record to the active CSV session."""
        with self._lock:
            self._require_active()
            assert self._csv_writer is not None and self.metadata is not None
            self._frame_indexes.add(frame_index)
            if video_source is not None:
                self.metadata.video_source = video_source
            bbox = detection.bbox
            self._csv_writer.writerow(
                {
                    "timestamp": (timestamp or datetime.now(timezone.utc)).isoformat(),
                    "frame_index": frame_index,
                    "bbox_x1": bbox.x1,
                    "bbox_y1": bbox.y1,
                    "bbox_x2": bbox.x2,
                    "bbox_y2": bbox.y2,
                    "class_id": detection.class_id,
                    "class_name": detection.class_name or "",
                    "confidence": detection.confidence,
                    "video_source": video_source or self.metadata.video_source or "",
                }
            )
            self.metadata.total_detections += 1
            self.metadata.total_frames = len(self._frame_indexes)

    def log_no_detection(
        self,
        frame_index: int,
        video_source: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Append a placeholder record for a frame without detections."""
        with self._lock:
            self._require_active()
            assert self._csv_writer is not None and self.metadata is not None
            self._frame_indexes.add(frame_index)
            if video_source is not None:
                self.metadata.video_source = video_source
            self._csv_writer.writerow(
                {
                    "timestamp": (timestamp or datetime.now(timezone.utc)).isoformat(),
                    "frame_index": frame_index,
                    "bbox_x1": "-",
                    "bbox_y1": "-",
                    "bbox_x2": "-",
                    "bbox_y2": "-",
                    "class_id": "-",
                    "class_name": "-",
                    "confidence": "-",
                    "video_source": video_source or self.metadata.video_source or "",
                }
            )
            self.metadata.total_frames = len(self._frame_indexes)

    def record_fps(self, fps: float) -> None:
        """Record an FPS sample for the session mean."""
        if fps >= 0:
            with self._lock:
                self._fps_samples.append(float(fps))

    def flush(self) -> None:
        """Flush buffered CSV data and metadata to disk."""
        with self._lock:
            if self._csv_file is not None:
                self._csv_file.flush()
            if self.metadata is not None and self.json_path is not None:
                self._write_metadata()

    def end_session(self, mean_fps: Optional[float] = None) -> Optional[SessionMetadata]:
        """Stop auto-flushing, close the CSV, and write final session metadata."""
        with self._lock:
            if self._csv_file is None or self.metadata is None:
                return None
            self._cancel_timer()
            self.metadata.end_time = datetime.now(timezone.utc).isoformat()
            if mean_fps is not None:
                self.metadata.mean_fps = float(mean_fps)
            elif self._fps_samples:
                self.metadata.mean_fps = sum(self._fps_samples) / len(self._fps_samples)
            self.metadata.total_frames = len(self._frame_indexes)
            self._write_metadata()
            self._csv_file.flush()
            self._csv_file.close()
            result = self.metadata
            self._csv_file = None
            self._csv_writer = None
            return result

    def _schedule_flush(self) -> None:
        self._timer = threading.Timer(self.config.flush_interval_seconds, self._auto_flush)
        self._timer.daemon = True
        self._timer.start()

    def _auto_flush(self) -> None:
        with self._lock:
            if self._csv_file is None:
                return
            self.flush()
            self._schedule_flush()

    def _cancel_timer(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _write_metadata(self) -> None:
        if self.json_path is not None and self.metadata is not None:
            self.json_path.write_text(
                json.dumps(asdict(self.metadata), indent=2), encoding="utf-8"
            )

    def _require_active(self) -> None:
        if self._csv_file is None:
            raise RuntimeError("No active logging session. Call start_session() first.")