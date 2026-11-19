"""Focused tests for real-time inference components."""

import csv
import json
import re
from datetime import datetime, timezone
from unittest.mock import patch

import numpy as np

from uav_survivor_detection.config.models import DisplayConfig, LoggerConfig
from uav_survivor_detection.inference import (
    BoundingBox,
    Detection,
    DetectionEngine,
    DetectionEngineConfig,
    DisplayModule,
    Logger,
)


class FakeModel:
    def predict(self, **kwargs):
        class Boxes:
            xyxy = [[1, 2, 10, 12]]
            conf = [0.8]
            cls = [0]

        class Result:
            boxes = Boxes()
            names = {0: "survivor"}

        return [Result()]


def test_detection_engine_converts_mock_result():
    engine = DetectionEngine(DetectionEngineConfig(num_threads=1), model=FakeModel())
    detections = engine.predict(np.zeros((32, 32, 3), dtype=np.uint8))

    assert len(detections) == 1
    assert detections[0].bbox == BoundingBox(1.0, 2.0, 10.0, 12.0)
    assert detections[0].class_name == "survivor"
    assert detections[0].confidence == 0.8


def test_detection_engine_handles_empty_results():
    class EmptyModel:
        def predict(self, **kwargs):
            return []

    engine = DetectionEngine(DetectionEngineConfig(num_threads=1), model=EmptyModel())
    assert engine.predict(np.zeros((32, 32, 3), dtype=np.uint8)) == []


def test_detection_engine_loads_checkpoint_and_warms_up(tmp_path):
    config = DetectionEngineConfig(checkpoint_path=tmp_path / "model.pt", img_size=32)
    fake_model = FakeModel()
    with patch("uav_survivor_detection.inference.detection_engine.YOLO", return_value=fake_model) as yolo:
        engine = DetectionEngine(config, model=None)
        engine.load_model()
        engine.warmup(iterations=2)

    yolo.assert_called_once_with(str(config.checkpoint_path))
    assert len(engine.predict(np.zeros((32, 32, 3), dtype=np.uint8))) == 1


def test_detection_engine_configures_cpu_threads():
    with patch("uav_survivor_detection.inference.detection_engine.torch.set_num_threads") as set_threads:
        DetectionEngine(DetectionEngineConfig(num_threads=3), model=FakeModel())
    set_threads.assert_called_once_with(3)


def test_display_renders_and_toggles_bounding_boxes():
    detection = Detection(BoundingBox(1, 2, 10, 12), 0.8, 0, "survivor")
    display = DisplayModule(DisplayConfig())
    frame = np.zeros((32, 32, 3), dtype=np.uint8)

    rendered = display.render_frame(frame, [detection], fps=10)
    assert rendered.shape == frame.shape
    assert np.any(rendered != frame)

    display.toggle_bounding_boxes(False)
    without_boxes = display.render_frame(frame, [detection], fps=10)
    assert np.any(without_boxes != frame)


def test_display_keyboard_controls_toggle_and_quit():
    display = DisplayModule(DisplayConfig())
    assert display.handle_keyboard_input(ord("b")) is False
    assert display.bbox_enabled is False
    assert display.handle_keyboard_input(ord("q")) is True


def test_display_uses_stable_class_color_and_status_overlay():
    display = DisplayModule(DisplayConfig())
    detection = Detection(BoundingBox(1, 1, 8, 8), 0.75, 0, "survivor")
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
    first = display.render_frame(frame, [detection], fps=12.0, timestamp=timestamp)
    color = display._color_for("survivor")
    second = display.render_frame(frame, [detection], fps=12.0, timestamp=timestamp)
    assert color == display._color_for("survivor")
    assert np.array_equal(first, second)
    assert np.any(first != frame)


def test_logger_writes_detection_and_no_detection_records(tmp_path):
    logger = Logger(LoggerConfig(output_dir=tmp_path, flush_interval_seconds=60))
    logger.start_session("mock-video")
    logger.log_detection(0, Detection(BoundingBox(1, 2, 3, 4), 0.9, 0, "survivor"))
    logger.log_no_detection(1)
    metadata = logger.end_session(mean_fps=10)

    assert metadata is not None
    assert metadata.total_frames == 2
    assert metadata.total_detections == 1
    assert metadata.mean_fps == 10

    csv_path = next(tmp_path.glob("*.csv"))
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert len(rows) == 2
    assert rows[1]["bbox_x1"] == "-"

    json_path = next(tmp_path.glob("*.json"))
    saved_metadata = json.loads(json_path.read_text(encoding="utf-8"))
    assert saved_metadata["total_frames"] == 2


def test_logger_writes_100_detections(tmp_path):
    logger = Logger(LoggerConfig(output_dir=tmp_path, flush_interval_seconds=60))
    logger.start_session("mock-video")
    detection = Detection(BoundingBox(1, 2, 3, 4), 0.9, 0, "survivor")
    for frame_index in range(100):
        logger.log_detection(frame_index, detection)
    logger.flush()
    metadata = logger.end_session()

    assert metadata.total_frames == 100
    assert metadata.total_detections == 100
    with next(tmp_path.glob("*.csv")).open(newline="", encoding="utf-8") as csv_file:
        assert len(list(csv.DictReader(csv_file))) == 100


def test_logger_auto_flush_can_be_triggered(tmp_path):
    logger = Logger(LoggerConfig(output_dir=tmp_path, flush_interval_seconds=60))
    logger.start_session("mock-video")
    with patch.object(logger, "flush") as flush:
        logger._auto_flush()
    flush.assert_called_once()
    logger.end_session()


def test_logger_filename_uses_timestamp_format(tmp_path):
    logger = Logger(LoggerConfig(output_dir=tmp_path, flush_interval_seconds=60))
    logger.start_session("mock-video")
    assert re.match(r"^detections_\d{8}_\d{6}\.csv$", logger.csv_path.name)
    assert re.match(r"^session_\d{8}_\d{6}\.json$", logger.json_path.name)
    logger.end_session()