"""Unit tests for ablation runner module."""

import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass

from src.uav_survivor_detection.ablation.runner import (
    AblationRunner,
    VariantMetrics,
    AblationResults
)
from src.uav_survivor_detection.config.models import (
    AblationConfig,
    ModelVariant,
    EnhancementConfig,
    TrainingConfig
)


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary output directory."""
    return tmp_path / "ablation_output"


@pytest.fixture
def mock_dataset_yaml(tmp_path):
    """Create mock dataset YAML file."""
    yaml_path = tmp_path / "data.yaml"
    yaml_path.write_text("train: train\nval: val\ntest: test\nnames: ['person']")
    return yaml_path


@pytest.fixture
def mock_test_subset(tmp_path):
    """Create mock test subset directory."""
    test_dir = tmp_path / "test"
    test_dir.mkdir(parents=True, exist_ok=True)
    return test_dir


@pytest.fixture
def baseline_variant():
    """Create baseline variant with no enhancements."""
    return ModelVariant(
        name="baseline",
        enhancement_config=EnhancementConfig(
            enable_gam=False,
            enable_cbam=False,
            enable_c3k2_arc=False,
            enable_cl_concat=False
        ),
        training_config=TrainingConfig(
            model_name="yolo11n",
            epochs=1,
            batch_size=2,
            random_seed=42,
            device="cpu"
        )
    )


@pytest.fixture
def enhanced_variant():
    """Create enhanced variant with GAM."""
    return ModelVariant(
        name="gam-only",
        enhancement_config=EnhancementConfig(
            enable_gam=True,
            enable_cbam=False,
            enable_c3k2_arc=False,
            enable_cl_concat=False
        ),
        training_config=TrainingConfig(
            model_name="yolo11n",
            epochs=1,
            batch_size=2,
            random_seed=42,
            device="cpu"
        )
    )


@pytest.fixture
def ablation_config(baseline_variant, temp_output_dir, mock_dataset_yaml, mock_test_subset):
    """Create ablation configuration."""
    return AblationConfig(
        variants=[baseline_variant],
        dataset_yaml=mock_dataset_yaml,
        output_dir=temp_output_dir,
        test_subset_path=mock_test_subset
    )


class TestAblationRunner:
    """Test suite for AblationRunner."""
    
    def test_initialization(self, ablation_config):
        """Test AblationRunner initialization."""
        runner = AblationRunner(ablation_config)
        
        assert runner.config == ablation_config
        assert runner.config.output_dir.exists()
        assert len(runner.config.variants) == 1
    
    def test_initialization_invalid_config(self):
        """Test initialization with invalid config type."""
        with pytest.raises(TypeError):
            AblationRunner("not a config")
    
    def test_variant_metrics_dataclass(self):
        """Test VariantMetrics dataclass structure."""
        metrics = VariantMetrics(
            variant_name="test",
            map50=0.85,
            map50_95=0.65,
            small_object_ap=0.55,
            fps_mean=15.0,
            latency_ms_mean=66.7,
            latency_ms_p95=80.0,
            parameters=2500000,
            gflops=3.5,
            training_duration_seconds=120.0,
            status="success",
            error_message=None
        )
        
        assert metrics.variant_name == "test"
        assert metrics.map50 == 0.85
        assert metrics.status == "success"
        assert metrics.error_message is None
    
    def test_ablation_results_dataclass(self):
        """Test AblationResults dataclass structure."""
        metrics = VariantMetrics(
            variant_name="test",
            map50=0.85,
            map50_95=0.65,
            small_object_ap=0.55,
            fps_mean=15.0,
            latency_ms_mean=66.7,
            latency_ms_p95=80.0,
            parameters=2500000,
            gflops=3.5,
            training_duration_seconds=120.0,
            status="success"
        )
        
        results = AblationResults(
            variants=[metrics],
            timestamp="2024-01-01T00:00:00Z"
        )
        
        assert len(results.variants) == 1
        assert results.timestamp == "2024-01-01T00:00:00Z"
    
    @patch('src.uav_survivor_detection.ablation.runner.TrainingPipeline')
    @patch('src.uav_survivor_detection.ablation.runner.Evaluator')
    @patch('src.uav_survivor_detection.ablation.runner.YOLO')
    def test_train_variant_baseline(
        self,
        mock_yolo_class,
        mock_evaluator_class,
        mock_training_pipeline_class,
        ablation_config,
        baseline_variant,
        tmp_path
    ):
        """Test training a baseline variant."""
        # Setup mocks
        mock_model = MagicMock()
        mock_yolo_class.return_value = mock_model
        
        mock_training_result = MagicMock()
        mock_training_result.best_checkpoint = tmp_path / "best.pt"
        mock_training_result.best_map50 = 0.85
        mock_training_result.total_duration_seconds = 100.0
        
        mock_pipeline = MagicMock()
        mock_pipeline.train.return_value = mock_training_result
        mock_training_pipeline_class.return_value = mock_pipeline
        
        # Create runner and train variant
        runner = AblationRunner(ablation_config)
        checkpoint_path = runner.train_variant(baseline_variant)
        
        # Assertions
        assert checkpoint_path == mock_training_result.best_checkpoint
        mock_yolo_class.assert_called_once_with("yolo11n.pt")
        mock_pipeline.train.assert_called_once_with(resume=False)
    
    @patch('src.uav_survivor_detection.ablation.runner.Evaluator')
    def test_evaluate_variant(self, mock_evaluator_class, ablation_config, tmp_path):
        """Test evaluating a variant."""
        # Setup mocks
        checkpoint_path = tmp_path / "best.pt"
        checkpoint_path.touch()  # Create the file
        
        mock_detection_metrics = MagicMock()
        mock_detection_metrics.map50 = 0.85
        mock_detection_metrics.map50_95 = 0.65
        mock_detection_metrics.small_object_ap = 0.55
        
        mock_efficiency_metrics = MagicMock()
        mock_efficiency_metrics.fps_mean = 15.0
        mock_efficiency_metrics.latency_ms_mean = 66.7
        mock_efficiency_metrics.latency_ms_p95 = 80.0
        
        mock_complexity_metrics = MagicMock()
        mock_complexity_metrics.parameters = 2500000
        mock_complexity_metrics.gflops = 3.5
        
        mock_eval_result = MagicMock()
        mock_eval_result.detection_metrics = mock_detection_metrics
        mock_eval_result.efficiency_metrics = mock_efficiency_metrics
        mock_eval_result.complexity_metrics = mock_complexity_metrics
        
        mock_evaluator = MagicMock()
        mock_evaluator.evaluate.return_value = mock_eval_result
        mock_evaluator_class.return_value = mock_evaluator
        
        # Create runner and evaluate variant
        runner = AblationRunner(ablation_config)
        metrics = runner.evaluate_variant("baseline", checkpoint_path)
        
        # Assertions
        assert metrics.variant_name == "baseline"
        assert metrics.map50 == 0.85
        assert metrics.map50_95 == 0.65
        assert metrics.small_object_ap == 0.55
        assert metrics.fps_mean == 15.0
        assert metrics.latency_ms_mean == 66.7
        assert metrics.parameters == 2500000
        assert metrics.gflops == 3.5
        assert metrics.status == "success"
        assert metrics.error_message is None
    
    def test_evaluate_variant_checkpoint_not_found(self, ablation_config, tmp_path):
        """Test evaluate_variant raises error when checkpoint doesn't exist."""
        runner = AblationRunner(ablation_config)
        checkpoint_path = tmp_path / "nonexistent.pt"
        
        with pytest.raises(FileNotFoundError):
            runner.evaluate_variant("test", checkpoint_path)
    
    def test_export_results_csv(self, ablation_config, tmp_path):
        """Test exporting results to CSV."""
        metrics = VariantMetrics(
            variant_name="baseline",
            map50=0.85,
            map50_95=0.65,
            small_object_ap=0.55,
            fps_mean=15.0,
            latency_ms_mean=66.7,
            latency_ms_p95=80.0,
            parameters=2500000,
            gflops=3.5,
            training_duration_seconds=120.0,
            status="success"
        )
        
        results = AblationResults(
            variants=[metrics],
            timestamp="2024-01-01T00:00:00Z"
        )
        
        runner = AblationRunner(ablation_config)
        runner.export_results(results)
        
        # Check CSV file exists and has content
        csv_path = ablation_config.output_dir / "ablation_results.csv"
        assert csv_path.exists()
        
        csv_content = csv_path.read_text()
        assert "variant_name" in csv_content
        assert "baseline" in csv_content
        assert "0.85" in csv_content
    
    def test_export_results_markdown(self, ablation_config):
        """Test exporting results to Markdown."""
        metrics = VariantMetrics(
            variant_name="baseline",
            map50=0.85,
            map50_95=0.65,
            small_object_ap=0.55,
            fps_mean=15.0,
            latency_ms_mean=66.7,
            latency_ms_p95=80.0,
            parameters=2500000,
            gflops=3.5,
            training_duration_seconds=120.0,
            status="success"
        )
        
        results = AblationResults(
            variants=[metrics],
            timestamp="2024-01-01T00:00:00Z"
        )
        
        runner = AblationRunner(ablation_config)
        runner.export_results(results)
        
        # Check Markdown file exists and has content
        md_path = ablation_config.output_dir / "ablation_results.md"
        assert md_path.exists()
        
        md_content = md_path.read_text()
        assert "# Ablation Study Results" in md_content
        assert "baseline" in md_content
        assert "0.85" in md_content
        assert "## Performance Comparison" in md_content
    
    @patch('src.uav_survivor_detection.ablation.runner.TrainingPipeline')
    @patch('src.uav_survivor_detection.ablation.runner.Evaluator')
    @patch('src.uav_survivor_detection.ablation.runner.YOLO')
    def test_run_ablation_study_with_failure(
        self,
        mock_yolo_class,
        mock_evaluator_class,
        mock_training_pipeline_class,
        ablation_config
    ):
        """Test ablation study handles variant failure gracefully."""
        # Setup mocks to raise exception during training
        mock_pipeline = MagicMock()
        mock_pipeline.train.side_effect = Exception("Training failed")
        mock_training_pipeline_class.return_value = mock_pipeline
        
        mock_yolo_class.return_value = MagicMock()
        
        # Run ablation study
        runner = AblationRunner(ablation_config)
        results = runner.run_ablation_study()
        
        # Assertions
        assert len(results.variants) == 1
        assert results.variants[0].status == "failed"
        assert "Training failed" in results.variants[0].error_message
        assert results.variants[0].map50 == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
