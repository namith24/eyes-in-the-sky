"""Fast end-to-end contract test for the offline training workflow."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

from uav_survivor_detection.config.models import EvaluatorConfig, TrainingConfig
from uav_survivor_detection.dataset.pipeline import DatasetPipeline
from uav_survivor_detection.evaluation.evaluator import Evaluator
from uav_survivor_detection.training.pipeline import TrainingPipeline


def test_training_checkpoint_can_be_evaluated(tmp_path: Path):
    """Exercise dataset config, two-epoch training artifacts, and evaluation metrics."""
    raw_dataset = tmp_path / "raw_dataset"
    raw_dataset.mkdir()
    for image_index in range(100):
        image_path = raw_dataset / f"image_{image_index:03d}.jpg"
        cv2.imwrite(str(image_path), np.zeros((32, 32, 3), dtype=np.uint8))
        image_path.with_suffix(".txt").write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")

    processed_dataset = tmp_path / "processed_dataset"
    dataset_pipeline = DatasetPipeline(
        dataset_path=raw_dataset,
        output_dir=processed_dataset,
        overwrite_existing=True,
    )
    assert dataset_pipeline.verify_annotations() == []
    dataset_pipeline.split_dataset()
    dataset_pipeline.export_yolo_format()
    manifest = dataset_pipeline.generate_manifest()
    assert manifest.total_images == 100
    dataset_yaml = processed_dataset / "data.yaml"
    assert dataset_yaml.exists()

    output_dir = tmp_path / "training"
    training_config = TrainingConfig(
        model_name="yolo11n",
        dataset_yaml=dataset_yaml,
        epochs=2,
        batch_size=2,
        img_size=32,
        learning_rate=0.01,
        random_seed=42,
        output_dir=output_dir,
        device="cpu",
        workers=0,
    )

    weights_dir = output_dir / "train" / "weights"
    results_csv = output_dir / "train" / "results.csv"

    mock_model = Mock()
    mock_model.train.return_value = SimpleNamespace()

    def create_training_artifacts(**kwargs):
        weights_dir.mkdir(parents=True, exist_ok=True)
        (weights_dir / "best.pt").write_bytes(b"mock checkpoint")
        (weights_dir / "last.pt").write_bytes(b"mock checkpoint")
        results_csv.parent.mkdir(parents=True, exist_ok=True)
        results_csv.write_text(
            "epoch,metrics/mAP50(B),metrics/mAP50-95(B)\n"
            "0,0.50,0.30\n1,0.60,0.40\n",
            encoding="utf-8",
        )

    mock_model.train.side_effect = create_training_artifacts
    with patch("uav_survivor_detection.training.pipeline.YOLO", return_value=mock_model):
        training_result = TrainingPipeline(training_config).train()

    assert training_result.best_checkpoint.exists()
    assert training_result.final_checkpoint.exists()
    assert len(training_result.metrics_history) == 2
    assert training_result.best_map50 == 0.6

    evaluator_config = EvaluatorConfig(
        checkpoint_path=training_result.best_checkpoint,
        test_dataset_path=tmp_path,
        conf_threshold=0.25,
        iou_threshold=0.5,
        img_size=32,
        device="cpu",
    )
    evaluation = Evaluator(evaluator_config)
    evaluation.model = Mock()
    evaluation.model.val.return_value = SimpleNamespace(
        box=SimpleNamespace(mp=0.8, mr=0.75, map50=0.7, map=0.5)
    )
    with patch.object(evaluation, "_compute_small_object_ap", return_value=0.4):
        metrics = evaluation.compute_detection_metrics()

    assert metrics.map50 == 0.7
    assert metrics.map50_95 == 0.5
    assert metrics.small_object_ap == 0.4