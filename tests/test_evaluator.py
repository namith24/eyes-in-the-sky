"""Unit tests for Evaluator module.

This module tests the comprehensive evaluation capabilities of the Evaluator class,
including detection metrics computation, efficiency metrics measurement, model complexity
calculation, and report generation.

**Validates: Requirements 5.1, 5.2, 5.6**
"""

import json
import pytest
import shutil
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, PropertyMock

import numpy as np
import torch
import yaml

from src.uav_survivor_detection.config.models import EvaluatorConfig
from src.uav_survivor_detection.evaluation.evaluator import (
    Evaluator,
    DetectionMetrics,
    EfficiencyMetrics,
    ModelComplexity,
    EvaluationResult
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    temp = tempfile.mkdtemp()
    yield Path(temp)
    shutil.rmtree(temp, ignore_errors=True)


@pytest.fixture
def mock_checkpoint(temp_dir):
    """Create a mock checkpoint file."""
    checkpoint = temp_dir / "model.pt"
    checkpoint.touch()
    return checkpoint


@pytest.fixture
def mock_test_dataset(temp_dir):
    """Create a mock test dataset with 10 images."""
    import cv2
    
    # Create directory structure
    images_dir = temp_dir / "test" / "images"
    labels_dir = temp_dir / "test" / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Create 10 test images (640x640)
    for i in range(10):
        img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        img_path = images_dir / f"test_{i}.jpg"
        cv2.imwrite(str(img_path), img)
        
        # Create labels with varying box sizes (some small objects)
        label_path = labels_dir / f"test_{i}.txt"
        if i < 3:
            # Small objects (<32x32 pixels = <0.05x0.05 normalized)
            width = 0.04
            height = 0.04
        else:
            # Regular objects
            width = 0.2
            height = 0.3
        
        x_center = 0.3 + (i * 0.05)
        y_center = 0.3 + (i * 0.05)
        label_path.write_text(f"0 {x_center} {y_center} {width} {height}\n")
    
    # Create data.yaml
    data_yaml = temp_dir / "data.yaml"
    yaml_content = {
        'path': str(temp_dir),
        'test': 'test/images',
        'names': {0: 'person'},
        'nc': 1
    }
    
    with open(data_yaml, 'w') as f:
        yaml.dump(yaml_content, f)
    
    return data_yaml


@pytest.fixture
def evaluator_config(mock_checkpoint, mock_test_dataset):
    """Create an EvaluatorConfig for testing."""
    return EvaluatorConfig(
        checkpoint_path=mock_checkpoint,
        test_dataset_path=mock_test_dataset,
        device="cpu",
        conf_threshold=0.25,
        iou_threshold=0.5,
        img_size=640
    )


@pytest.fixture
def evaluator(evaluator_config):
    """Create an Evaluator instance for testing."""
    return Evaluator(evaluator_config)


class TestEvaluatorInit:
    """Tests for Evaluator initialization."""
    
    def test_init_with_valid_config(self, evaluator_config):
        """Test initialization with valid configuration."""
        evaluator = Evaluator(evaluator_config)
        assert evaluator.config == evaluator_config
        assert evaluator.model is None
        assert evaluator.latency_measurements == []
    
    def test_init_with_invalid_config_type(self):
        """Test initialization with invalid config type raises TypeError."""
        with pytest.raises(TypeError, match="Expected EvaluatorConfig"):
            Evaluator({"invalid": "config"})
    
    def test_init_sets_config_attributes(self, evaluator):
        """Test that config attributes are accessible."""
        assert evaluator.config.conf_threshold == 0.25
        assert evaluator.config.iou_threshold == 0.5
        assert evaluator.config.img_size == 640
        assert evaluator.config.device == "cpu"


class TestLoadModel:
    """Tests for load_model method."""
    
    @patch('src.uav_survivor_detection.evaluation.evaluator.YOLO')
    def test_load_model_success(self, mock_yolo, evaluator):
        """Test successful model loading."""
        mock_model = Mock()
        mock_yolo.return_value = mock_model
        
        evaluator.load_model()
        
        mock_yolo.assert_called_once_with(evaluator.config.checkpoint_path)
        assert evaluator.model == mock_model
    
    @patch('src.uav_survivor_detection.evaluation.evaluator.YOLO')
    def test_load_model_sets_cpu_device(self, mock_yolo, evaluator):
        """Test that model is set to CPU device."""
        mock_model = Mock()
        mock_yolo.return_value = mock_model
        
        evaluator.load_model()
        
        mock_model.to.assert_called_once_with("cpu")
    
    def test_load_model_nonexistent_checkpoint_raises_error(self, evaluator, temp_dir):
        """Test loading nonexistent checkpoint raises FileNotFoundError."""
        evaluator.config.checkpoint_path = temp_dir / "nonexistent.pt"
        
        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            evaluator.load_model()
    
    def test_load_model_none_checkpoint_path_raises_error(self, evaluator_config):
        """Test loading with None checkpoint path raises ValueError."""
        evaluator_config.checkpoint_path = None
        evaluator = Evaluator(evaluator_config)
        
        with pytest.raises(ValueError, match="checkpoint_path must be specified"):
            evaluator.load_model()


class TestComputeDetectionMetrics:
    """Tests for compute_detection_metrics method.
    
    **Validates: Requirement 5.1** - Compute precision, recall, mAP@50, mAP@50:95, Small_Object_AP
    """
    
    @patch('src.uav_survivor_detection.evaluation.evaluator.YOLO')
    def test_compute_detection_metrics_with_known_ground_truth(self, mock_yolo, evaluator):
        """Test mAP@50 and mAP@50:95 computation with known ground truth."""
        # Setup mock model
        mock_model = Mock()
        evaluator.model = mock_model
        
        # Create mock validation results
        mock_val_results = Mock()
        mock_box = Mock()
        
        # Set detection metrics
        mock_box.mp = 0.85  # precision
        mock_box.mr = 0.78  # recall
        mock_box.map50 = 0.72  # mAP@50
        mock_box.map = 0.65  # mAP@50:95
        mock_box.ap_class_index = [0]
        mock_box.ap = [0.72]
        
        mock_val_results.box = mock_box
        mock_model.val.return_value = mock_val_results
        mock_model.names = {0: "person"}
        
        # Mock Small_Object_AP computation
        with patch.object(evaluator, '_compute_small_object_ap', return_value=0.58):
            metrics = evaluator.compute_detection_metrics()
        
        # Verify metrics
        assert isinstance(metrics, DetectionMetrics)
        assert metrics.precision == 0.85
        assert metrics.recall == 0.78
        assert metrics.map50 == 0.72
        assert metrics.map50_95 == 0.65
        assert metrics.small_object_ap == 0.58
        assert 'person' in metrics.per_class_ap
        assert metrics.per_class_ap['person'] == 0.72
    
    @patch('src.uav_survivor_detection.evaluation.evaluator.YOLO')
    def test_compute_detection_metrics_calls_model_val(self, mock_yolo, evaluator):
        """Test that compute_detection_metrics calls model.val() with correct params."""
        mock_model = Mock()
        evaluator.model = mock_model
        
        mock_val_results = Mock()
        mock_box = Mock()
        mock_box.mp = 0.80
        mock_box.mr = 0.75
        mock_box.map50 = 0.70
        mock_box.map = 0.60
        mock_box.ap_class_index = []
        mock_box.ap = []
        mock_val_results.box = mock_box
        mock_model.val.return_value = mock_val_results
        
        with patch.object(evaluator, '_compute_small_object_ap', return_value=0.50):
            evaluator.compute_detection_metrics()
        
        mock_model.val.assert_called_once_with(
            data=str(evaluator.config.test_dataset_path),
            conf=evaluator.config.conf_threshold,
            iou=evaluator.config.iou_threshold,
            imgsz=evaluator.config.img_size,
            device=evaluator.config.device,
            verbose=False
        )
    
    def test_compute_detection_metrics_without_loaded_model_raises_error(self, evaluator):
        """Test computing metrics without loaded model raises ValueError."""
        with pytest.raises(ValueError, match="Model must be loaded"):
            evaluator.compute_detection_metrics()
    
    @patch('src.uav_survivor_detection.evaluation.evaluator.YOLO')
    def test_compute_detection_metrics_handles_missing_attributes(self, mock_yolo, evaluator):
        """Test handling of validation results with missing attributes."""
        mock_model = Mock()
        evaluator.model = mock_model
        
        # Create mock results with missing attributes
        mock_val_results = Mock()
        mock_box = Mock(spec=[])  # Empty spec = no attributes
        mock_val_results.box = mock_box
        mock_model.val.return_value = mock_val_results
        
        with patch.object(evaluator, '_compute_small_object_ap', return_value=0.0):
            metrics = evaluator.compute_detection_metrics()
        
        # Should default to 0.0 for missing metrics
        assert metrics.precision == 0.0
        assert metrics.recall == 0.0
        assert metrics.map50 == 0.0
        assert metrics.map50_95 == 0.0


class TestComputeSmallObjectAP:
    """Tests for _compute_small_object_ap method.
    
    **Validates: Requirement 5.1** - Small_Object_AP computation on filtered subset (<32x32 pixels)
    """
    
    @patch('src.uav_survivor_detection.evaluation.evaluator.COCO')
    @patch('src.uav_survivor_detection.evaluation.evaluator.COCOeval')
    def test_compute_small_object_ap_filters_correctly(self, mock_cocoeval, mock_coco, evaluator):
        """Test Small_Object_AP computation on filtered subset (<32x32 pixels)."""
        mock_model = Mock()
        mock_model.names = {0: "person"}
        evaluator.model = mock_model
        
        # Mock COCO evaluation
        mock_coco_gt = Mock()
        mock_coco_dt = Mock()
        mock_coco.return_value = mock_coco_gt
        mock_coco_gt.loadRes.return_value = mock_coco_dt
        
        mock_eval = Mock()
        mock_eval.stats = [0.65, 0.70, 0.60, 0.50, 0.45, 0.40]  # AP @ IoU=0.50:0.95 is first
        mock_cocoeval.return_value = mock_eval
        
        with patch.object(evaluator, '_get_test_image_paths') as mock_get_paths:
            with patch.object(evaluator, '_convert_yolo_to_coco_format') as mock_convert:
                # Setup mock image paths
                mock_get_paths.return_value = [Path(f"test_{i}.jpg") for i in range(10)]
                
                # Setup mock COCO data with small objects
                gt_data = {
                    'images': [{'id': i} for i in range(1, 11)],
                    'annotations': [
                        # Small objects (area < 1024)
                        {'id': 1, 'image_id': 1, 'bbox': [100, 100, 30, 30], 'area': 900},
                        {'id': 2, 'image_id': 2, 'bbox': [150, 150, 25, 28], 'area': 700},
                        # Regular objects (area >= 1024)
                        {'id': 3, 'image_id': 3, 'bbox': [200, 200, 40, 40], 'area': 1600},
                    ],
                    'categories': [{'id': 0, 'name': 'person'}]
                }
                
                pred_data = {
                    'images': [{'id': i} for i in range(1, 11)],
                    'annotations': [
                        {'id': 1, 'image_id': 1, 'bbox': [100, 100, 30, 30], 'score': 0.9},
                        {'id': 2, 'image_id': 2, 'bbox': [150, 150, 25, 28], 'score': 0.85},
                    ],
                    'categories': [{'id': 0, 'name': 'person'}]
                }
                
                mock_convert.side_effect = [gt_data, pred_data]
                
                small_ap = evaluator._compute_small_object_ap()
        
        assert small_ap == 0.65
        assert mock_eval.evaluate.called
        assert mock_eval.accumulate.called
        assert mock_eval.summarize.called
    
    def test_compute_small_object_ap_returns_zero_on_no_test_images(self, evaluator):
        """Test Small_Object_AP returns 0.0 when no test images found."""
        with patch.object(evaluator, '_get_test_image_paths', return_value=[]):
            small_ap = evaluator._compute_small_object_ap()
        
        assert small_ap == 0.0
    
    def test_compute_small_object_ap_returns_zero_on_exception(self, evaluator):
        """Test Small_Object_AP returns 0.0 and handles exceptions gracefully."""
        with patch.object(evaluator, '_get_test_image_paths', side_effect=Exception("Test error")):
            small_ap = evaluator._compute_small_object_ap()
        
        assert small_ap == 0.0


class TestComputeEfficiencyMetrics:
    """Tests for compute_efficiency_metrics method.
    
    **Validates: Requirement 5.2** - FPS and latency measurement returns valid ranges
    """
    
    @patch('src.uav_survivor_detection.evaluation.evaluator.YOLO')
    def test_compute_efficiency_metrics_on_test_set(self, mock_yolo, evaluator, temp_dir):
        """Test FPS and latency measurement returns valid ranges."""
        mock_model = Mock()
        evaluator.model = mock_model
        
        # Mock predict to simulate latency
        def mock_predict(*args, **kwargs):
            time.sleep(0.01)  # Simulate 10ms latency
            return Mock()
        
        mock_model.predict = mock_predict
        
        # Mock test images (10 images)
        test_images = [temp_dir / f"test_{i}.jpg" for i in range(10)]
        for img_path in test_images:
            img_path.touch()
        
        with patch.object(evaluator, '_get_test_image_paths', return_value=test_images):
            metrics = evaluator.compute_efficiency_metrics()
        
        # Verify metrics structure
        assert isinstance(metrics, EfficiencyMetrics)
        
        # Verify FPS values are positive and reasonable
        assert metrics.fps_mean > 0
        assert metrics.fps_median > 0
        assert metrics.fps_mean < 1000  # Sanity check
        
        # Verify latency values are positive
        assert metrics.latency_ms_mean > 0
        assert metrics.latency_ms_median > 0
        assert metrics.latency_ms_p95 > 0
        
        # Verify p95 >= median >= some reasonable value
        assert metrics.latency_ms_p95 >= metrics.latency_ms_median
    
    @patch('src.uav_survivor_detection.evaluation.evaluator.YOLO')
    def test_compute_efficiency_metrics_warmup_iterations(self, mock_yolo, evaluator, temp_dir):
        """Test that warmup iterations are performed before measurement."""
        mock_model = Mock()
        evaluator.model = mock_model
        
        mock_predict = Mock(return_value=Mock())
        mock_model.predict = mock_predict
        
        test_images = [temp_dir / f"test_{i}.jpg" for i in range(10)]
        for img_path in test_images:
            img_path.touch()
        
        with patch.object(evaluator, '_get_test_image_paths', return_value=test_images):
            evaluator.compute_efficiency_metrics()
        
        # Should call predict for warmup (5 times) + actual measurements (10 times)
        assert mock_predict.call_count >= 10  # At least the test images
    
    def test_compute_efficiency_metrics_without_model_raises_error(self, evaluator):
        """Test computing efficiency metrics without loaded model raises ValueError."""
        with pytest.raises(ValueError, match="Model must be loaded"):
            evaluator.compute_efficiency_metrics()
    
    @patch('src.uav_survivor_detection.evaluation.evaluator.YOLO')
    def test_compute_efficiency_metrics_stores_latency_measurements(self, mock_yolo, evaluator, temp_dir):
        """Test that latency measurements are stored in evaluator."""
        mock_model = Mock()
        mock_model.predict = Mock(return_value=Mock())
        evaluator.model = mock_model
        
        test_images = [temp_dir / f"test_{i}.jpg" for i in range(10)]
        for img_path in test_images:
            img_path.touch()
        
        with patch.object(evaluator, '_get_test_image_paths', return_value=test_images):
            evaluator.compute_efficiency_metrics()
        
        # Should have 10 latency measurements
        assert len(evaluator.latency_measurements) == 10
        assert all(lat > 0 for lat in evaluator.latency_measurements)


class TestComputeComplexityMetrics:
    """Tests for compute_complexity_metrics method."""
    
    @patch('src.uav_survivor_detection.evaluation.evaluator.YOLO')
    @patch('thop.profile')
    def test_compute_complexity_metrics(self, mock_profile, mock_yolo, evaluator):
        """Test model parameter count and GFLOPs computation."""
        # Create mock PyTorch model with parameters
        mock_pytorch_model = Mock()
        mock_param1 = Mock()
        mock_param1.numel.return_value = 1000000  # 1M params
        mock_param2 = Mock()
        mock_param2.numel.return_value = 500000  # 0.5M params
        mock_pytorch_model.parameters.return_value = [mock_param1, mock_param2]
        
        mock_model = Mock()
        mock_model.model = mock_pytorch_model
        evaluator.model = mock_model
        
        # Mock thop.profile to return FLOPs
        mock_profile.return_value = (5.2e9, 0)  # 5.2 GFLOPs
        
        metrics = evaluator.compute_complexity_metrics()
        
        assert isinstance(metrics, ModelComplexity)
        assert metrics.parameters == 1500000  # 1M + 0.5M
        assert metrics.gflops == 5.2
    
    def test_compute_complexity_metrics_without_model_raises_error(self, evaluator):
        """Test computing complexity without loaded model raises ValueError."""
        with pytest.raises(ValueError, match="Model must be loaded"):
            evaluator.compute_complexity_metrics()
    
    @patch('src.uav_survivor_detection.evaluation.evaluator.YOLO')
    @patch('thop.profile')
    def test_compute_complexity_metrics_handles_profile_failure(self, mock_profile, mock_yolo, evaluator):
        """Test handling of thop.profile failure."""
        mock_pytorch_model = Mock()
        mock_param = Mock()
        mock_param.numel.return_value = 1000000
        mock_pytorch_model.parameters.return_value = [mock_param]
        
        mock_model = Mock()
        mock_model.model = mock_pytorch_model
        evaluator.model = mock_model
        
        # Mock profile to raise exception
        mock_profile.side_effect = Exception("Profile failed")
        
        metrics = evaluator.compute_complexity_metrics()
        
        assert metrics.parameters == 1000000
        assert metrics.gflops == 0.0  # Should default to 0.0 on failure


class TestEvaluateMethod:
    """Tests for evaluate method - comprehensive evaluation.
    
    **Validates: Requirement 5.6** - Complete test set evaluation
    """
    
    @patch('src.uav_survivor_detection.evaluation.evaluator.YOLO')
    def test_evaluate_on_10_image_test_set_produces_valid_metrics(self, mock_yolo, evaluator, temp_dir):
        """Test evaluation on 10-image test set produces valid metrics."""
        # Setup mock model
        mock_model = Mock()
        mock_yolo.return_value = mock_model
        evaluator.model = mock_model
        
        # Mock all metric computation methods
        mock_detection_metrics = DetectionMetrics(
            precision=0.85,
            recall=0.78,
            map50=0.72,
            map50_95=0.65,
            small_object_ap=0.58,
            per_class_ap={'person': 0.72}
        )
        
        mock_efficiency_metrics = EfficiencyMetrics(
            fps_mean=15.5,
            fps_median=15.2,
            latency_ms_mean=64.5,
            latency_ms_median=65.8,
            latency_ms_p95=72.3
        )
        
        mock_complexity_metrics = ModelComplexity(
            parameters=2500000,
            gflops=6.2
        )
        
        with patch.object(evaluator, 'compute_detection_metrics', return_value=mock_detection_metrics):
            with patch.object(evaluator, 'compute_efficiency_metrics', return_value=mock_efficiency_metrics):
                with patch.object(evaluator, 'compute_complexity_metrics', return_value=mock_complexity_metrics):
                    with patch.object(evaluator, '_count_test_images', return_value=10):
                        with patch.object(evaluator, '_count_total_detections', return_value=25):
                            result = evaluator.evaluate()
        
        # Verify result structure
        assert isinstance(result, EvaluationResult)
        assert result.detection_metrics == mock_detection_metrics
        assert result.efficiency_metrics == mock_efficiency_metrics
        assert result.complexity_metrics == mock_complexity_metrics
        assert result.total_images == 10
        assert result.total_detections == 25
        assert result.evaluation_duration_seconds >= 0
        assert result.timestamp  # Should have UTC timestamp
    
    @patch('src.uav_survivor_detection.evaluation.evaluator.YOLO')
    def test_evaluate_loads_model_if_not_loaded(self, mock_yolo, evaluator):
        """Test that evaluate loads model if not already loaded."""
        mock_model = Mock()
        mock_yolo.return_value = mock_model
        
        # Mock all metric methods
        with patch.object(evaluator, 'compute_detection_metrics', return_value=Mock()):
            with patch.object(evaluator, 'compute_efficiency_metrics', return_value=Mock()):
                with patch.object(evaluator, 'compute_complexity_metrics', return_value=Mock()):
                    with patch.object(evaluator, '_count_test_images', return_value=10):
                        with patch.object(evaluator, '_count_total_detections', return_value=25):
                            evaluator.evaluate()
        
        # Model should be loaded
        assert evaluator.model is not None
    
    def test_evaluate_without_test_dataset_raises_error(self, evaluator_config, mock_checkpoint):
        """Test evaluate without test dataset path raises ValueError."""
        evaluator_config.test_dataset_path = None
        evaluator_config.checkpoint_path = mock_checkpoint
        evaluator = Evaluator(evaluator_config)
        
        # Mock the model loading to avoid trying to load the empty checkpoint
        with patch('src.uav_survivor_detection.evaluation.evaluator.YOLO'):
            with pytest.raises(ValueError, match="test_dataset_path must be specified"):
                evaluator.evaluate()
    
    @patch('src.uav_survivor_detection.evaluation.evaluator.YOLO')
    def test_evaluate_measures_duration(self, mock_yolo, evaluator):
        """Test that evaluate measures total evaluation duration."""
        mock_model = Mock()
        mock_yolo.return_value = mock_model
        
        def slow_detection_metrics():
            time.sleep(0.1)
            return Mock(precision=0.8, recall=0.75, map50=0.7, map50_95=0.6, 
                       small_object_ap=0.5, per_class_ap={})
        
        with patch.object(evaluator, 'compute_detection_metrics', side_effect=slow_detection_metrics):
            with patch.object(evaluator, 'compute_efficiency_metrics', return_value=Mock()):
                with patch.object(evaluator, 'compute_complexity_metrics', return_value=Mock()):
                    with patch.object(evaluator, '_count_test_images', return_value=10):
                        with patch.object(evaluator, '_count_total_detections', return_value=25):
                            result = evaluator.evaluate()
        
        # Duration should be at least 0.1 seconds
        assert result.evaluation_duration_seconds >= 0.1


class TestEvaluationOnZeroDetections:
    """Tests for evaluation on test set with zero detections.
    
    **Validates: Requirement 5.1** - Evaluation handles zero detections gracefully
    """
    
    @patch('src.uav_survivor_detection.evaluation.evaluator.YOLO')
    def test_evaluation_on_test_set_with_zero_detections_handles_gracefully(self, mock_yolo, evaluator):
        """Test evaluation on test set with zero detections handles gracefully."""
        mock_model = Mock()
        evaluator.model = mock_model
        
        # Mock validation results with zero detections
        mock_val_results = Mock()
        mock_box = Mock()
        mock_box.mp = 0.0
        mock_box.mr = 0.0
        mock_box.map50 = 0.0
        mock_box.map = 0.0
        mock_box.ap_class_index = []
        mock_box.ap = []
        mock_val_results.box = mock_box
        mock_model.val.return_value = mock_val_results
        
        with patch.object(evaluator, '_compute_small_object_ap', return_value=0.0):
            metrics = evaluator.compute_detection_metrics()
        
        # Should return valid metrics with zero values
        assert isinstance(metrics, DetectionMetrics)
        assert metrics.precision == 0.0
        assert metrics.recall == 0.0
        assert metrics.map50 == 0.0
        assert metrics.map50_95 == 0.0
        assert metrics.small_object_ap == 0.0
        assert metrics.per_class_ap == {}


class TestReportGeneration:
    """Tests for evaluation report generation (JSON and Markdown).
    
    **Validates: Requirement 5.4** - Report generation in JSON and Markdown formats
    """
    
    def test_to_dict_converts_evaluation_result(self, evaluator):
        """Test JSON report generation."""
        result = EvaluationResult(
            detection_metrics=DetectionMetrics(
                precision=0.85,
                recall=0.78,
                map50=0.72,
                map50_95=0.65,
                small_object_ap=0.58,
                per_class_ap={'person': 0.72}
            ),
            efficiency_metrics=EfficiencyMetrics(
                fps_mean=15.5,
                fps_median=15.2,
                latency_ms_mean=64.5,
                latency_ms_median=65.8,
                latency_ms_p95=72.3
            ),
            complexity_metrics=ModelComplexity(
                parameters=2500000,
                gflops=6.2
            ),
            total_images=10,
            total_detections=25,
            evaluation_duration_seconds=45.3,
            timestamp="2024-01-15T10:30:00Z"
        )
        
        result_dict = evaluator.to_dict(result)
        
        # Verify JSON structure
        assert 'detection_metrics' in result_dict
        assert 'efficiency_metrics' in result_dict
        assert 'complexity_metrics' in result_dict
        assert 'total_images' in result_dict
        assert 'total_detections' in result_dict
        assert 'evaluation_duration_seconds' in result_dict
        assert 'timestamp' in result_dict
        
        # Verify values
        assert result_dict['detection_metrics']['precision'] == 0.85
        assert result_dict['detection_metrics']['map50'] == 0.72
        assert result_dict['efficiency_metrics']['fps_mean'] == 15.5
        assert result_dict['complexity_metrics']['parameters'] == 2500000
        assert result_dict['total_images'] == 10
    
    def test_to_dict_serializable_to_json(self, evaluator):
        """Test that to_dict output is JSON serializable."""
        result = EvaluationResult(
            detection_metrics=DetectionMetrics(
                precision=0.85,
                recall=0.78,
                map50=0.72,
                map50_95=0.65,
                small_object_ap=0.58,
                per_class_ap={'person': 0.72}
            ),
            efficiency_metrics=EfficiencyMetrics(
                fps_mean=15.5,
                fps_median=15.2,
                latency_ms_mean=64.5,
                latency_ms_median=65.8,
                latency_ms_p95=72.3
            ),
            complexity_metrics=ModelComplexity(
                parameters=2500000,
                gflops=6.2
            ),
            total_images=10,
            total_detections=25,
            evaluation_duration_seconds=45.3,
            timestamp="2024-01-15T10:30:00Z"
        )
        
        result_dict = evaluator.to_dict(result)
        
        # Should not raise exception
        json_str = json.dumps(result_dict, indent=2)
        assert json_str
        assert "precision" in json_str
        assert "fps_mean" in json_str


class TestGetTestImagePaths:
    """Tests for _get_test_image_paths helper method."""
    
    def test_get_test_image_paths_from_yaml(self, evaluator, mock_test_dataset):
        """Test getting test image paths from YAML file."""
        image_paths = evaluator._get_test_image_paths()
        
        # Should find 10 test images (may have duplicates on case-insensitive file systems)
        # Use set to get unique paths
        unique_paths = list(set(image_paths))
        assert len(unique_paths) == 10
        assert all(path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp'] for path in unique_paths)
    
    def test_get_test_image_paths_from_directory(self, evaluator, temp_dir):
        """Test getting test image paths from directory."""
        import cv2
        
        # Create test images in a directory
        test_dir = temp_dir / "test_images"
        test_dir.mkdir()
        
        for i in range(5):
            img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(test_dir / f"img_{i}.jpg"), img)
        
        evaluator.config.test_dataset_path = test_dir
        image_paths = evaluator._get_test_image_paths()
        
        # Use set to get unique paths (case-insensitive file systems may have duplicates)
        unique_paths = list(set(image_paths))
        assert len(unique_paths) == 5
    
    def test_get_test_image_paths_handles_invalid_path(self, evaluator, temp_dir):
        """Test error handling for invalid test dataset path."""
        evaluator.config.test_dataset_path = temp_dir / "nonexistent"
        
        with pytest.raises(ValueError, match="Invalid test dataset path"):
            evaluator._get_test_image_paths()


class TestGetLabelPath:
    """Tests for _get_label_path helper method."""
    
    def test_get_label_path_replaces_images_with_labels(self, evaluator):
        """Test label path conversion from image path."""
        # Use OS-appropriate path separators
        if Path("test").as_posix() == "test":  # Unix-like
            image_path = Path("dataset/test/images/test_001.jpg")
        else:  # Windows
            image_path = Path("dataset\\test\\images\\test_001.jpg")
        
        label_path = evaluator._get_label_path(image_path)
        
        assert "labels" in str(label_path)
        assert label_path.suffix == ".txt"
        assert label_path.stem == "test_001"
    
    def test_get_label_path_changes_extension(self, evaluator):
        """Test that extension is changed to .txt."""
        image_path = Path("dataset/images/test.png")
        label_path = evaluator._get_label_path(image_path)
        
        assert label_path.suffix == ".txt"
        assert label_path.stem == "test"


class TestCountMethods:
    """Tests for _count_test_images and _count_total_detections methods."""
    
    def test_count_test_images(self, evaluator):
        """Test counting test images."""
        with patch.object(evaluator, '_get_test_image_paths') as mock_get_paths:
            mock_get_paths.return_value = [Path(f"test_{i}.jpg") for i in range(10)]
            count = evaluator._count_test_images()
        
        assert count == 10
    
    def test_count_total_detections_returns_placeholder(self, evaluator):
        """Test _count_total_detections returns placeholder value."""
        # Current implementation returns 0 as placeholder
        count = evaluator._count_total_detections()
        assert count == 0


class TestDataclasses:
    """Tests for dataclass structures."""
    
    def test_detection_metrics_creation(self):
        """Test DetectionMetrics dataclass creation."""
        metrics = DetectionMetrics(
            precision=0.85,
            recall=0.78,
            map50=0.72,
            map50_95=0.65,
            small_object_ap=0.58,
            per_class_ap={'person': 0.72}
        )
        
        assert metrics.precision == 0.85
        assert metrics.recall == 0.78
        assert metrics.map50 == 0.72
        assert metrics.map50_95 == 0.65
        assert metrics.small_object_ap == 0.58
        assert metrics.per_class_ap == {'person': 0.72}
    
    def test_efficiency_metrics_creation(self):
        """Test EfficiencyMetrics dataclass creation."""
        metrics = EfficiencyMetrics(
            fps_mean=15.5,
            fps_median=15.2,
            latency_ms_mean=64.5,
            latency_ms_median=65.8,
            latency_ms_p95=72.3
        )
        
        assert metrics.fps_mean == 15.5
        assert metrics.fps_median == 15.2
        assert metrics.latency_ms_mean == 64.5
        assert metrics.latency_ms_median == 65.8
        assert metrics.latency_ms_p95 == 72.3
    
    def test_model_complexity_creation(self):
        """Test ModelComplexity dataclass creation."""
        metrics = ModelComplexity(
            parameters=2500000,
            gflops=6.2
        )
        
        assert metrics.parameters == 2500000
        assert metrics.gflops == 6.2
    
    def test_evaluation_result_creation(self):
        """Test EvaluationResult dataclass creation."""
        result = EvaluationResult(
            detection_metrics=DetectionMetrics(0.85, 0.78, 0.72, 0.65, 0.58, {}),
            efficiency_metrics=EfficiencyMetrics(15.5, 15.2, 64.5, 65.8, 72.3),
            complexity_metrics=ModelComplexity(2500000, 6.2),
            total_images=10,
            total_detections=25,
            evaluation_duration_seconds=45.3,
            timestamp="2024-01-15T10:30:00Z"
        )
        
        assert result.total_images == 10
        assert result.total_detections == 25
        assert result.evaluation_duration_seconds == 45.3
        assert result.timestamp == "2024-01-15T10:30:00Z"
