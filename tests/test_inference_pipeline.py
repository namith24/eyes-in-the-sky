"""Mock integration tests for the inference pipeline."""

from pathlib import Path
import time

import csv
import cv2
import json
import numpy as np

from uav_survivor_detection.config.models import InferenceConfig
from uav_survivor_detection.inference import BoundingBox, Detection, InferencePipeline


class FakeCapture:
    def __init__(self):
        self.released = False
        self.read_count = 0

    def isOpened(self):
        return not self.released

    def read(self):
        self.read_count += 1
        return True, np.zeros((16, 16, 3), dtype=np.uint8)

    def release(self):
        self.released = True


class FakeEngine:
    model = object()

    def predict(self, frame):
        return [Detection(BoundingBox(1, 1, 5, 5), 0.9, 0, "survivor")]


class FakeDisplay:
    def render_frame(self, frame, detections, fps=None):
        return frame

    def show_frame(self, frame):
        return True


class StreamingDisplay(FakeDisplay):
    def show_frame(self, frame):
        return False


class FakeLogger:
    def __init__(self):
        self.detections = []
        self.no_detections = []
        self.started = False
        self.ended = False

    def start_session(self, video_source):
        self.started = True

    def log_detection(self, frame_index, detection, video_source):
        self.detections.append((frame_index, detection, video_source))

    def log_no_detection(self, frame_index, video_source):
        self.no_detections.append((frame_index, video_source))

    def record_fps(self, fps):
        pass

    def end_session(self, mean_fps=None):
        self.ended = True


def test_pipeline_processes_frame_and_shuts_down_cleanly(tmp_path: Path):
    config = InferenceConfig(
        video_source="0",
        model_checkpoint=tmp_path / "model.pt",
        display_enabled=True,
        logging_enabled=True,
    )
    logger = FakeLogger()
    capture = FakeCapture()
    pipeline = InferencePipeline(
        config,
        detection_engine=FakeEngine(),
        display=FakeDisplay(),
        logger=logger,
        capture=capture,
    )

    pipeline.run()

    assert capture.released
    assert logger.started and logger.ended
    assert len(logger.detections) == 1
    assert pipeline.frame_index == 1


def test_pipeline_parses_sources_and_prunes_frame_buffer(tmp_path: Path):
    config = InferenceConfig("0", tmp_path / "model.pt", max_frame_buffer_seconds=2.0)
    pipeline = InferencePipeline(config, detection_engine=FakeEngine())

    assert pipeline._parse_video_source("3") == 3
    assert pipeline._parse_video_source("rtsp://camera") == "rtsp://camera"
    pipeline._buffer_frame(1.0, np.zeros((2, 2, 3), dtype=np.uint8))
    pipeline._buffer_frame(4.0, np.zeros((2, 2, 3), dtype=np.uint8))
    assert len(pipeline.frame_buffer) == 1


def test_pipeline_processes_video_file_and_logs_30_frames(tmp_path: Path):
    video_path = tmp_path / "sample.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (16, 16),
    )
    for _ in range(30):
        writer.write(np.zeros((16, 16, 3), dtype=np.uint8))
    writer.release()

    log_dir = tmp_path / "logs"
    config = InferenceConfig(
        video_source=str(video_path),
        model_checkpoint=tmp_path / "model.pt",
        display_enabled=True,
        logging_enabled=True,
        log_output_dir=log_dir,
    )
    pipeline = InferencePipeline(
        config,
        detection_engine=FakeEngine(),
        display=StreamingDisplay(),
    )
    pipeline.run()

    csv_path = next(log_dir.glob("*.csv"))
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    expected_detections = 30
    assert abs(len(rows) - expected_detections) <= expected_detections * 0.10
    assert rows[0]["frame_index"] == "0"
    assert {"timestamp", "frame_index", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "class_id", "class_name", "confidence", "video_source"}.issubset(rows[0])
    assert pipeline.frame_index == 30

    json_path = next(log_dir.glob("*.json"))
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    assert {
        "session_id",
        "start_time",
        "end_time",
        "video_source",
        "total_frames",
        "total_detections",
        "mean_fps",
    }.issubset(metadata)
    assert metadata["total_frames"] == 30
    assert metadata["total_detections"] == 30


def test_pipeline_cpu_configuration_is_usable(tmp_path: Path):
    config = InferenceConfig(
        video_source="0",
        model_checkpoint=tmp_path / "model.pt",
        device="cpu",
        display_enabled=False,
        logging_enabled=False,
    )
    pipeline = InferencePipeline(config, detection_engine=FakeEngine())
    assert pipeline.config.device == "cpu"
    assert pipeline._parse_video_source("0") == 0


def test_mock_inference_benchmark_exceeds_ten_fps(tmp_path: Path):
    pipeline = InferencePipeline(
        InferenceConfig("0", tmp_path / "model.pt", device="cpu"),
        detection_engine=FakeEngine(),
    )
    frames = [np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(100)]
    started = time.perf_counter()
    for frame in frames:
        pipeline.process_frame(frame)
    elapsed = time.perf_counter() - started

    assert elapsed > 0
    assert len(frames) / elapsed >= 10