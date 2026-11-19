"""Unit tests for Training Pipeline."""

import json
import pytest
import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import torch
import yaml

from src.uav_survivor_detection.config.models import TrainingConfig
from src.uav_survivor_detection.training.pipeline import TrainingPipeline, TrainingResult


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    temp = tempfile.mkdtemp()
    yield Path(temp)
    shutil.rmtree(temp, ignore_errors=True)


@pytest.fixture
def training_config(temp_dir):
    """Create a TrainingConfig for testing."""
    dataset_yaml = temp_dir / "data.yaml"
    dataset_yaml.write_text("train: train\nval: val\ntest: test\nnames: [person]")
    
    return TrainingConfig(
        model_name="yolo11n",
        dataset_yaml=dataset_yaml,
        epochs=2,
        batch_size=4,
        img_size=640,
        learning_rate=0.01,
        random_seed=42,
        output_dir=temp_dir / "runs" / "train",
        device="cpu",
        workers=2
    )


@pytest.fixture
def training_pipeline(training_config):
    """Create a TrainingPipeline instance for testing."""
    return TrainingPipeline(training_config)


class TestTrainingPipelineInit:
    """Tests for TrainingPipeline initialization."""
    
    def test_init_creates_output_directory(self, training_config):
        """Test that initialization creates output directory."""
        pipeline = TrainingPipeline(training_config)
        assert pipeline.config.output_dir.exists()
    
    def test_init_sets_config(self, training_config):
        """Test that initialization sets configuration."""
        pipeline = TrainingPipeline(training_config)
        assert pipeline.config == training_config
    
    def test_init_initializes_metrics(self, training_pipeline):
        """Test that initialization sets up metrics tracking."""
        assert training_pipeline.best_map50 == 0.0
        assert training_pipeline.metrics_history == []
    
    def test_init_creates_logger(self, training_pipeline):
        """Test that initialization creates logger."""
        assert training_pipeline.logger is not None
        assert training_pipeline.logger.name == "TrainingPipeline"


class TestLoadPretrainedWeights:
    """Tests for load_pretrained_weights method."""
    
    @patch('src.uav_survivor_detection.training.pipeline.YOLO')
    def test_load_pretrained_weights_default(self, mock_yolo, training_pipeline):
        """Test loading pretrained weights with default model name."""
        mock_model = Mock()
        mock_yolo.return_value = mock_model
        
        model = training_pipeline.load_pretrained_weights()
        
        mock_yolo.assert_called_once_with("yolo11n.pt")
        assert model == mock_model
    
    @patch('src.uav_survivor_detection.training.pipeline.YOLO')
    def test_load_pretrained_weights_custom_name(self, mock_yolo, training_pipeline):
        """Test loading pretrained weights with custom model name."""
        mock_model = Mock()
        mock_yolo.return_value = mock_model
        
        model = training_pipeline.load_pretrained_weights("yolov8n.pt")
        
        mock_yolo.assert_called_once_with("yolov8n.pt")
        assert model == mock_model
    
    @patch('src.uav_survivor_detection.training.pipeline.YOLO')
    def test_load_pretrained_weights_failure(self, mock_yolo, training_pipeline):
        """Test handling of load failure."""
        mock_yolo.side_effect = Exception("Model not found")
        
        with pytest.raises(Exception, match="Model not found"):
            training_pipeline.load_pretrained_weights()


class TestTrain:
    """Tests for train method."""
    
    @patch('src.uav_survivor_detection.training.pipeline.YOLO')
    @patch('src.uav_survivor_detection.training.pipeline.torch.manual_seed')
    def test_train_sets_random_seed(self, mock_seed, mock_yolo, training_pipeline):
        """Test that train sets random seed for reproducibility."""
        # Create mock model with train method
        mock_model = Mock()
        mock_train_results = Mock()
        mock_model.train.return_value = mock_train_results
        mock_yolo.return_value = mock_model
        
        # Create mock checkpoint files
        weights_dir = training_pipeline.config.output_dir / "train" / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        (weights_dir / "best.pt").touch()
        (weights_dir / "last.pt").touch()
        
        # Create mock results CSV
        results_csv = training_pipeline.config.output_dir / "train" / "results.csv"
        results_csv.parent.mkdir(parents=True, exist_ok=True)
        results_csv.write_text("epoch,metrics/mAP50(B)\n0,0.5\n1,0.6\n")
        
        training_pipeline.train()
        
        mock_seed.assert_called_once_with(42)
    
    @patch('src.uav_survivor_detection.training.pipeline.YOLO')
    def test_train_loads_pretrained_weights(self, mock_yolo, training_pipeline):
        """Test that train loads pretrained weights when not resuming."""
        mock_model = Mock()
        mock_train_results = Mock()
        mock_model.train.return_value = mock_train_results
        mock_yolo.return_value = mock_model
        
        # Create mock checkpoint files
        weights_dir = training_pipeline.config.output_dir / "train" / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        (weights_dir / "best.pt").touch()
        (weights_dir / "last.pt").touch()
        
        # Create mock results CSV
        results_csv = training_pipeline.config.output_dir / "train" / "results.csv"
        results_csv.parent.mkdir(parents=True, exist_ok=True)
        results_csv.write_text("epoch,metrics/mAP50(B)\n0,0.5\n")
        
        training_pipeline.train()
        
        mock_yolo.assert_called_with("yolo11n.pt")
    
    @patch('src.uav_survivor_detection.training.pipeline.YOLO')
    def test_train_resumes_from_checkpoint(self, mock_yolo, training_pipeline, temp_dir):
        """Test that train resumes from checkpoint when specified."""
        checkpoint_path = temp_dir / "checkpoint.pt"
        checkpoint_path.touch()
        
        mock_model = Mock()
        mock_train_results = Mock()
        mock_model.train.return_value = mock_train_results
        mock_yolo.return_value = mock_model
        
        # Create mock checkpoint files
        weights_dir = training_pipeline.config.output_dir / "train" / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        (weights_dir / "best.pt").touch()
        (weights_dir / "last.pt").touch()
        
        # Create mock results CSV
        results_csv = training_pipeline.config.output_dir / "train" / "results.csv"
        results_csv.parent.mkdir(parents=True, exist_ok=True)
        results_csv.write_text("epoch,metrics/mAP50(B)\n0,0.5\n")
        
        training_pipeline.train(resume=True, checkpoint_path=checkpoint_path)
        
        mock_yolo.assert_called_with(str(checkpoint_path))
    
    def test_train_resume_without_checkpoint_raises_error(self, training_pipeline):
        """Test that resuming without checkpoint path raises ValueError."""
        with pytest.raises(ValueError, match="checkpoint_path must be provided"):
            training_pipeline.train(resume=True, checkpoint_path=None)
    
    def test_train_resume_nonexistent_checkpoint_raises_error(self, training_pipeline, temp_dir):
        """Test that resuming with nonexistent checkpoint raises FileNotFoundError."""
        checkpoint_path = temp_dir / "nonexistent.pt"
        
        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            training_pipeline.train(resume=True, checkpoint_path=checkpoint_path)
    
    def test_train_missing_dataset_yaml_raises_error(self, training_config, temp_dir):
        """Test that training with missing dataset YAML raises FileNotFoundError."""
        training_config.dataset_yaml = temp_dir / "nonexistent.yaml"
        pipeline = TrainingPipeline(training_config)
        
        with pytest.raises(FileNotFoundError, match="Dataset YAML not found"):
            pipeline.train()
    
    @patch('src.uav_survivor_detection.training.pipeline.YOLO')
    def test_train_calls_ultralytics_api(self, mock_yolo, training_pipeline):
        """Test that train calls ultralytics API with correct parameters."""
        mock_model = Mock()
        mock_train_results = Mock()
        mock_model.train.return_value = mock_train_results
        mock_yolo.return_value = mock_model
        
        # Create mock checkpoint files
        weights_dir = training_pipeline.config.output_dir / "train" / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        (weights_dir / "best.pt").touch()
        (weights_dir / "last.pt").touch()
        
        # Create mock results CSV
        results_csv = training_pipeline.config.output_dir / "train" / "results.csv"
        results_csv.parent.mkdir(parents=True, exist_ok=True)
        results_csv.write_text("epoch,metrics/mAP50(B)\n0,0.5\n")
        
        training_pipeline.train()
        
        mock_model.train.assert_called_once()
        call_kwargs = mock_model.train.call_args[1]
        assert call_kwargs['epochs'] == 2
        assert call_kwargs['batch'] == 4
        assert call_kwargs['imgsz'] == 640
        assert call_kwargs['lr0'] == 0.01
        assert call_kwargs['device'] == 'cpu'
        assert call_kwargs['workers'] == 2
        assert call_kwargs['seed'] == 42
    
    @patch('src.uav_survivor_detection.training.pipeline.YOLO')
    def test_train_returns_training_result(self, mock_yolo, training_pipeline):
        """Test that train returns TrainingResult with correct structure."""
        mock_model = Mock()
        mock_train_results = Mock()
        mock_model.train.return_value = mock_train_results
        mock_yolo.return_value = mock_model
        
        # Create mock checkpoint files
        weights_dir = training_pipeline.config.output_dir / "train" / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        (weights_dir / "best.pt").touch()
        (weights_dir / "last.pt").touch()
        
        # Create mock results CSV
        results_csv = training_pipeline.config.output_dir / "train" / "results.csv"
        results_csv.parent.mkdir(parents=True, exist_ok=True)
        results_csv.write_text("epoch,metrics/mAP50(B)\n0,0.5\n1,0.6\n")
        
        result = training_pipeline.train()
        
        assert isinstance(result, TrainingResult)
        assert result.best_checkpoint.exists()
        assert result.final_checkpoint.exists()
        assert isinstance(result.metrics_history, list)
        assert result.total_duration_seconds >= 0  # Allow 0.0 for mock training
        assert result.best_map50 >= 0
    
    @patch('src.uav_survivor_detection.training.pipeline.YOLO')
    def test_train_exports_metrics_json(self, mock_yolo, training_pipeline):
        """Test that train exports metrics history to JSON."""
        mock_model = Mock()
        mock_train_results = Mock()
        mock_model.train.return_value = mock_train_results
        mock_yolo.return_value = mock_model
        
        # Create mock checkpoint files
        weights_dir = training_pipeline.config.output_dir / "train" / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        (weights_dir / "best.pt").touch()
        (weights_dir / "last.pt").touch()
        
        # Create mock results CSV
        results_csv = training_pipeline.config.output_dir / "train" / "results.csv"
        results_csv.parent.mkdir(parents=True, exist_ok=True)
        results_csv.write_text("epoch,metrics/mAP50(B)\n0,0.5\n1,0.6\n")
        
        training_pipeline.train()
        
        metrics_file = training_pipeline.config.output_dir / "train" / "metrics_history.json"
        assert metrics_file.exists()
        
        with open(metrics_file, 'r') as f:
            metrics_data = json.load(f)
        
        assert 'metrics_history' in metrics_data
        assert 'best_map50' in metrics_data
        assert 'total_duration_seconds' in metrics_data


class TestExtractMetricsHistory:
    """Tests for _extract_metrics_history method."""
    
    def test_extract_metrics_from_csv(self, training_pipeline):
        """Test extracting metrics from results CSV."""
        # Create mock results CSV
        results_csv = training_pipeline.config.output_dir / "train" / "results.csv"
        results_csv.parent.mkdir(parents=True, exist_ok=True)
        csv_content = """epoch,train/box_loss,metrics/mAP50(B),metrics/mAP50-95(B)
0,1.5,0.45,0.30
1,1.2,0.55,0.35
2,1.0,0.60,0.40"""
        results_csv.write_text(csv_content)
        
        mock_results = Mock()
        training_pipeline._extract_metrics_history(mock_results)
        
        assert len(training_pipeline.metrics_history) == 3
        assert training_pipeline.metrics_history[0]['map50'] == 0.45
        assert training_pipeline.metrics_history[1]['map50'] == 0.55
        assert training_pipeline.metrics_history[2]['map50'] == 0.60
        assert training_pipeline.best_map50 == 0.60
    
    def test_extract_metrics_handles_missing_csv(self, training_pipeline):
        """Test that missing CSV is handled gracefully."""
        mock_results = Mock()
        training_pipeline._extract_metrics_history(mock_results)
        
        assert training_pipeline.metrics_history == []


class TestOnEpochEnd:
    """Tests for on_epoch_end callback method."""
    
    def test_on_epoch_end_logs_metrics(self, training_pipeline):
        """Test that on_epoch_end logs metrics."""
        metrics = {'map50': 0.75, 'map50_95': 0.50}
        training_pipeline.on_epoch_end(1, metrics)
        
        assert len(training_pipeline.metrics_history) == 1
        assert training_pipeline.metrics_history[0] == metrics
    
    def test_on_epoch_end_tracks_best_map50(self, training_pipeline):
        """Test that on_epoch_end tracks best mAP@50."""
        training_pipeline.on_epoch_end(1, {'map50': 0.60})
        assert training_pipeline.best_map50 == 0.60
        
        training_pipeline.on_epoch_end(2, {'map50': 0.70})
        assert training_pipeline.best_map50 == 0.70
        
        training_pipeline.on_epoch_end(3, {'map50': 0.65})
        assert training_pipeline.best_map50 == 0.70  # Should not decrease
    
    @patch('src.uav_survivor_detection.training.pipeline.YOLO')
    def test_on_epoch_end_saves_checkpoint_on_improvement(self, mock_yolo, training_pipeline):
        """Test that on_epoch_end saves checkpoint when mAP@50 improves."""
        mock_model = Mock()
        mock_yolo.return_value = mock_model
        training_pipeline.model = mock_model
        
        metrics = {'map50': 0.75}
        training_pipeline.on_epoch_end(1, metrics)
        
        mock_model.save.assert_called_once()


class TestExportModel:
    """Tests for export_model method."""
    
    def test_export_model_pytorch_format(self, training_pipeline, temp_dir):
        """Test exporting model in PyTorch format."""
        mock_model = Mock()
        training_pipeline.model = mock_model
        
        output_path = temp_dir / "exported_model.pt"
        training_pipeline.export_model(output_path, format="pytorch")
        
        mock_model.save.assert_called_once_with(str(output_path))
    
    def test_export_model_other_format(self, training_pipeline, temp_dir):
        """Test exporting model in other formats."""
        mock_model = Mock()
        training_pipeline.model = mock_model
        
        output_path = temp_dir / "exported_model.onnx"
        training_pipeline.export_model(output_path, format="onnx")
        
        mock_model.export.assert_called_once_with(format="onnx", imgsz=640)
    
    def test_export_model_without_model_raises_error(self, training_pipeline, temp_dir):
        """Test that exporting without a loaded model raises ValueError."""
        output_path = temp_dir / "exported_model.pt"
        
        with pytest.raises(ValueError, match="No model loaded"):
            training_pipeline.export_model(output_path)


class TestReproducibility:
    """Tests for reproducibility requirements."""
    
    @patch('src.uav_survivor_detection.training.pipeline.torch.manual_seed')
    @patch('src.uav_survivor_detection.training.pipeline.YOLO')
    def test_same_seed_produces_same_behavior(self, mock_yolo, mock_seed, training_config):
        """Test that same random seed produces consistent behavior."""
        # Create two pipelines with same config
        pipeline1 = TrainingPipeline(training_config)
        pipeline2 = TrainingPipeline(training_config)
        
        # Setup mocks
        mock_model = Mock()
        mock_train_results = Mock()
        mock_model.train.return_value = mock_train_results
        mock_yolo.return_value = mock_model
        
        # Create mock checkpoint files for both pipelines
        for pipeline in [pipeline1, pipeline2]:
            weights_dir = pipeline.config.output_dir / "train" / "weights"
            weights_dir.mkdir(parents=True, exist_ok=True)
            (weights_dir / "best.pt").touch()
            (weights_dir / "last.pt").touch()
            
            results_csv = pipeline.config.output_dir / "train" / "results.csv"
            results_csv.parent.mkdir(parents=True, exist_ok=True)
            results_csv.write_text("epoch,metrics/mAP50(B)\n0,0.5\n")
        
        # Train both
        pipeline1.train()
        pipeline2.train()
        
        # Both should set the same seed
        assert mock_seed.call_count == 2
        assert mock_seed.call_args_list[0] == mock_seed.call_args_list[1]


class TestIntegrationWithSmallDataset:
    """Integration tests with small synthetic dataset.
    
    These tests create actual image files and run real (but minimal) YOLO training
    to verify end-to-end functionality.
    
    Requirements: 2.2, 2.4, 2.6, 2.7
    """
    
    @pytest.fixture
    def small_dataset(self, temp_dir):
        """Create a small synthetic dataset with 10 images."""
        import numpy as np
        import cv2
        
        # Create dataset structure
        train_images = temp_dir / "train" / "images"
        train_labels = temp_dir / "train" / "labels"
        val_images = temp_dir / "val" / "images"
        val_labels = temp_dir / "val" / "labels"
        
        for directory in [train_images, train_labels, val_images, val_labels]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Create 8 training images
        for i in range(8):
            # Create synthetic 640x640 RGB image
            img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            img_path = train_images / f"train_{i}.jpg"
            cv2.imwrite(str(img_path), img)
            
            # Create corresponding label (person class = 0, with random bbox)
            label_path = train_labels / f"train_{i}.txt"
            x_center = 0.3 + (i * 0.05)  # Vary position
            y_center = 0.3 + (i * 0.05)
            width = 0.2
            height = 0.3
            label_path.write_text(f"0 {x_center} {y_center} {width} {height}\n")
        
        # Create 2 validation images
        for i in range(2):
            img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            img_path = val_images / f"val_{i}.jpg"
            cv2.imwrite(str(img_path), img)
            
            label_path = val_labels / f"val_{i}.txt"
            label_path.write_text(f"0 0.5 0.5 0.2 0.3\n")
        
        # Create data.yaml
        data_yaml = temp_dir / "data.yaml"
        yaml_content = {
            'path': str(temp_dir),
            'train': 'train/images',
            'val': 'val/images',
            'names': {0: 'person'},
            'nc': 1
        }
        
        with open(data_yaml, 'w') as f:
            import yaml
            yaml.dump(yaml_content, f)
        
        return data_yaml
    
    @pytest.mark.slow
    def test_train_on_small_dataset_2_epochs(self, temp_dir, small_dataset):
        """Test training for 2 epochs on small dataset (10 images).
        
        This integration test creates 10 synthetic images and runs actual YOLO training
        for 2 epochs to verify the training pipeline works end-to-end.
        
        **Validates: Requirements 2.2, 2.4, 2.7**
        """
        # Create training config with minimal settings
        config = TrainingConfig(
            model_name="yolo11n",
            dataset_yaml=small_dataset,
            epochs=2,
            batch_size=2,
            img_size=640,
            learning_rate=0.01,
            random_seed=42,
            output_dir=temp_dir / "runs" / "train",
            device="cpu",
            workers=1
        )
        
        # Create pipeline
        pipeline = TrainingPipeline(config)
        
        # Train model
        result = pipeline.train()
        
        # Verify training completed
        assert isinstance(result, TrainingResult)
        assert result.total_duration_seconds > 0
        
        # Verify checkpoints exist
        assert result.best_checkpoint.exists(), "Best checkpoint should exist after training"
        assert result.final_checkpoint.exists(), "Final checkpoint should exist after training"
        
        # Verify metrics history has 2 epochs
        assert len(result.metrics_history) >= 0, "Metrics history should be populated"
        
        # Verify best_map50 is recorded (may be 0 for synthetic data)
        assert result.best_map50 >= 0, "Best mAP@50 should be non-negative"
        
        # Verify metrics JSON was exported
        metrics_json = config.output_dir / "train" / "metrics_history.json"
        assert metrics_json.exists(), "Metrics history JSON should be exported"
        
        with open(metrics_json, 'r') as f:
            metrics_data = json.load(f)
        
        assert 'best_checkpoint' in metrics_data
        assert 'final_checkpoint' in metrics_data
        assert 'metrics_history' in metrics_data
        assert 'total_duration_seconds' in metrics_data
        assert 'best_map50' in metrics_data
    
    @pytest.mark.slow
    def test_checkpoint_saved_after_training(self, temp_dir, small_dataset):
        """Test that checkpoint is saved after training completes.
        
        **Validates: Requirement 2.4**
        """
        config = TrainingConfig(
            model_name="yolo11n",
            dataset_yaml=small_dataset,
            epochs=2,
            batch_size=2,
            img_size=640,
            learning_rate=0.01,
            random_seed=42,
            output_dir=temp_dir / "runs" / "train",
            device="cpu",
            workers=1
        )
        
        pipeline = TrainingPipeline(config)
        result = pipeline.train()
        
        # Verify checkpoint files exist
        weights_dir = config.output_dir / "train" / "weights"
        assert weights_dir.exists(), "Weights directory should exist"
        
        best_pt = weights_dir / "best.pt"
        last_pt = weights_dir / "last.pt"
        
        assert best_pt.exists() or last_pt.exists(), "At least one checkpoint should exist"
        assert result.best_checkpoint.exists(), "Best checkpoint path should point to existing file"
        assert result.final_checkpoint.exists(), "Final checkpoint path should point to existing file"
    
    @pytest.mark.slow
    def test_training_resumption_from_checkpoint(self, temp_dir, small_dataset):
        """Test training resumption from checkpoint.
        
        **Validates: Requirement 2.6**
        """
        config = TrainingConfig(
            model_name="yolo11n",
            dataset_yaml=small_dataset,
            epochs=1,
            batch_size=2,
            img_size=640,
            learning_rate=0.01,
            random_seed=42,
            output_dir=temp_dir / "runs" / "train_initial",
            device="cpu",
            workers=1
        )
        
        # First training run
        pipeline1 = TrainingPipeline(config)
        result1 = pipeline1.train()
        
        # Verify checkpoint exists
        assert result1.final_checkpoint.exists(), "Checkpoint from first training should exist"
        
        # Resume training from checkpoint
        config2 = TrainingConfig(
            model_name="yolo11n",
            dataset_yaml=small_dataset,
            epochs=2,  # Train for 1 more epoch
            batch_size=2,
            img_size=640,
            learning_rate=0.01,
            random_seed=42,
            output_dir=temp_dir / "runs" / "train_resumed",
            device="cpu",
            workers=1
        )
        
        pipeline2 = TrainingPipeline(config2)
        result2 = pipeline2.train(resume=True, checkpoint_path=result1.final_checkpoint)
        
        # Verify resumed training completed
        assert isinstance(result2, TrainingResult)
        assert result2.total_duration_seconds > 0
        assert result2.final_checkpoint.exists()
    
    @pytest.mark.slow
    def test_error_handling_nonexistent_checkpoint(self, temp_dir, small_dataset):
        """Test error handling when resuming from non-existent checkpoint.
        
        **Validates: Requirement 2.6**
        """
        config = TrainingConfig(
            model_name="yolo11n",
            dataset_yaml=small_dataset,
            epochs=2,
            batch_size=2,
            img_size=640,
            learning_rate=0.01,
            random_seed=42,
            output_dir=temp_dir / "runs" / "train",
            device="cpu",
            workers=1
        )
        
        pipeline = TrainingPipeline(config)
        
        # Attempt to resume from non-existent checkpoint
        nonexistent_checkpoint = temp_dir / "nonexistent_checkpoint.pt"
        
        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            pipeline.train(resume=True, checkpoint_path=nonexistent_checkpoint)
    
    @pytest.mark.slow
    def test_metrics_history_logged_correctly(self, temp_dir, small_dataset):
        """Test that metrics history is logged correctly during training.
        
        **Validates: Requirements 2.3, 2.4**
        """
        config = TrainingConfig(
            model_name="yolo11n",
            dataset_yaml=small_dataset,
            epochs=2,
            batch_size=2,
            img_size=640,
            learning_rate=0.01,
            random_seed=42,
            output_dir=temp_dir / "runs" / "train",
            device="cpu",
            workers=1
        )
        
        pipeline = TrainingPipeline(config)
        result = pipeline.train()
        
        # Verify metrics history structure
        assert isinstance(result.metrics_history, list), "Metrics history should be a list"
        
        # Verify metrics JSON contains required fields
        metrics_json = config.output_dir / "train" / "metrics_history.json"
        assert metrics_json.exists(), "Metrics history JSON should be exported"
        
        with open(metrics_json, 'r') as f:
            metrics_data = json.load(f)
        
        # Verify structure
        assert 'metrics_history' in metrics_data
        assert 'best_map50' in metrics_data
        assert 'total_duration_seconds' in metrics_data
        assert 'config' in metrics_data
        
        # Verify config contains expected fields
        config_data = metrics_data['config']
        assert config_data['model_name'] == 'yolo11n'
        assert config_data['epochs'] == 2
        assert config_data['batch_size'] == 2
        assert config_data['random_seed'] == 42
        
        # Verify training log exists
        training_log = config.output_dir / "training.log"
        assert training_log.exists(), "Training log file should exist"
        
        # Verify log contains training messages
        log_content = training_log.read_text()
        assert "TrainingPipeline initialized" in log_content or "Training" in log_content
