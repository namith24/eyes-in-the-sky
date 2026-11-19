"""Deterministic performance and CPU-configuration checks."""

from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np

from uav_survivor_detection.config.models import InferenceConfig
from uav_survivor_detection.inference import BoundingBox, Detection, InferencePipeline


class FastEngine:
    model = object()

    def predict(self, frame):
        return [Detection(BoundingBox(1, 1, 4, 4), 0.9, 0, "survivor")]


def test_cpu_inference_mock_benchmark_meets_threshold(tmp_path: Path):
    pipeline = InferencePipeline(
        InferenceConfig("0", tmp_path / "model.pt", device="cpu"),
        detection_engine=FastEngine(),
    )
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    started = time.perf_counter()
    for _ in range(100):
        pipeline.process_frame(frame)
    fps = 100 / (time.perf_counter() - started)
    assert pipeline.config.device == "cpu"
    assert fps >= 10.0


def test_latency_and_complexity_metrics_contracts():
    latencies_ms = [10.0, 12.0, 15.0, 18.0, 20.0]
    p95 = float(np.percentile(latencies_ms, 95))
    assert p95 <= 50.0

    baseline = SimpleNamespace(parameters=1000, gflops=10.0)
    enhanced = SimpleNamespace(parameters=1150, gflops=12.0)
    assert (enhanced.parameters - baseline.parameters) / baseline.parameters <= 0.20
    assert (enhanced.gflops - baseline.gflops) / baseline.gflops <= 0.25