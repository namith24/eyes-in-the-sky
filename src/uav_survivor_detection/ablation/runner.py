"""Ablation study runner for systematic model variant evaluation.

This module coordinates the training and evaluation of model variants with different
enhancement combinations to quantify the contribution of each component.

**Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5**
"""

import logging
import time
import csv
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from ..config.models import AblationConfig, ModelVariant, EnhancementConfig, TrainingConfig, EvaluatorConfig
from ..training.pipeline import TrainingPipeline, TrainingResult
from ..evaluation.evaluator import Evaluator, EvaluationResult
from ..models.enhancement import EnhancementModule, ModelComplexity
from ultralytics import YOLO


@dataclass
class VariantMetrics:
    """Metrics for a single model variant.
    
    Attributes:
        variant_name: Human-readable variant name
        map50: Mean Average Precision at IoU threshold 0.50
        map50_95: Mean Average Precision averaged over IoU thresholds 0.50-0.95
        small_object_ap: Average Precision for small objects (<32x32 pixels)
        fps_mean: Mean frames per second
        latency_ms_mean: Mean latency in milliseconds
        latency_ms_p95: 95th percentile latency in milliseconds
        parameters: Total number of model parameters
        gflops: Giga floating-point operations per forward pass
        training_duration_seconds: Total training time in seconds
        status: Training/evaluation status ("success" or "failed")
        error_message: Error details if status is "failed"
    """
    variant_name: str
    map50: float
    map50_95: float
    small_object_ap: float
    fps_mean: float
    latency_ms_mean: float
    latency_ms_p95: float
    parameters: int
    gflops: float
    training_duration_seconds: float
    status: str
    error_message: Optional[str] = None


@dataclass
class AblationResults:
    """Complete ablation study results.
    
    Attributes:
        variants: List of metrics for all evaluated variants
        timestamp: Study completion timestamp (ISO format)
    """
    variants: List[VariantMetrics]
    timestamp: str


class AblationRunner:
    """Ablation study runner for systematic model variant evaluation.
    
    This class orchestrates training and evaluation of multiple model variants
    with different enhancement combinations to isolate the contribution of each
    enhancement component.
    
    **Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5**
    """
    
    def __init__(self, config: AblationConfig):
        """Initialize ablation runner.
        
        Args:
            config: Ablation configuration with variant definitions and shared parameters
            
        **Validates: Requirement 4.1** - Accept configuration file listing model variants
        """
        if not isinstance(config, AblationConfig):
            raise TypeError(f"Expected AblationConfig, got {type(config)}")
        
        self.config = config
        self.logger = logging.getLogger("AblationRunner")
        
        # Create output directory first
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Then setup logging
        self._setup_logging()
        
        self.logger.info(f"AblationRunner initialized with {len(config.variants)} variants")
        
    def _setup_logging(self) -> None:
        """Configure logging to file and console."""
        log_file = self.config.output_dir / "ablation_study.log"
        
        # Clear existing handlers
        self.logger.handlers.clear()
        self.logger.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def run_ablation_study(self) -> AblationResults:
        """Train and evaluate all variants sequentially.
        
        This method:
        1. Iterates through all configured variants
        2. Trains each variant using identical hyperparameters
        3. Evaluates each variant on the test subset
        4. Collects comprehensive metrics for each variant
        5. Handles failures gracefully without terminating the study
        
        Returns:
            AblationResults with metrics for all variants
            
        **Validates: Requirements 4.2, 4.3, 4.5** - Sequential training/evaluation with error handling
        """
        self.logger.info("=" * 80)
        self.logger.info("Starting ablation study")
        self.logger.info(f"Total variants: {len(self.config.variants)}")
        self.logger.info(f"Dataset YAML: {self.config.dataset_yaml}")
        self.logger.info(f"Test subset: {self.config.test_subset_path}")
        self.logger.info(f"Output directory: {self.config.output_dir}")
        self.logger.info("=" * 80)
        
        study_start_time = time.time()
        variant_metrics_list = []
        
        for idx, variant in enumerate(self.config.variants, start=1):
            self.logger.info("")
            self.logger.info(f"[{idx}/{len(self.config.variants)}] Processing variant: {variant.name}")
            self.logger.info("-" * 80)
            
            try:
                # Train variant
                self.logger.info(f"Training variant: {variant.name}")
                checkpoint_path = self.train_variant(variant)
                
                # Evaluate variant
                self.logger.info(f"Evaluating variant: {variant.name}")
                metrics = self.evaluate_variant(variant.name, checkpoint_path)
                
                variant_metrics_list.append(metrics)
                self.logger.info(f"✓ Variant {variant.name} completed successfully")
                
            except Exception as e:
                # Handle variant failure gracefully
                self.logger.error(f"✗ Variant {variant.name} failed: {e}")
                self.logger.exception("Full traceback:")
                
                # Create failed metrics entry
                failed_metrics = VariantMetrics(
                    variant_name=variant.name,
                    map50=0.0,
                    map50_95=0.0,
                    small_object_ap=0.0,
                    fps_mean=0.0,
                    latency_ms_mean=0.0,
                    latency_ms_p95=0.0,
                    parameters=0,
                    gflops=0.0,
                    training_duration_seconds=0.0,
                    status="failed",
                    error_message=str(e)
                )
                variant_metrics_list.append(failed_metrics)
                self.logger.info(f"Continuing with remaining variants...")
        
        study_duration = time.time() - study_start_time
        
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info(f"Ablation study completed in {study_duration:.2f} seconds")
        self.logger.info(f"Successful variants: {sum(1 for m in variant_metrics_list if m.status == 'success')}")
        self.logger.info(f"Failed variants: {sum(1 for m in variant_metrics_list if m.status == 'failed')}")
        self.logger.info("=" * 80)
        
        # Create results object
        results = AblationResults(
            variants=variant_metrics_list,
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
        
        # Export results to CSV and Markdown
        self.export_results(results)
        
        return results
    
    def train_variant(self, variant: ModelVariant) -> Path:
        """Train a single model variant.
        
        This method:
        1. Loads baseline YOLO11n model
        2. Applies enhancements if configured
        3. Trains using TrainingPipeline with variant-specific config
        4. Returns path to best checkpoint
        
        Args:
            variant: ModelVariant with enhancement and training configuration
            
        Returns:
            Path to trained checkpoint
            
        Raises:
            Exception: If training fails
            
        **Validates: Requirement 4.2** - Train variant with consistent hyperparameters
        """
        variant_output_dir = self.config.output_dir / variant.name
        variant_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Update training config with variant-specific output directory
        variant_training_config = TrainingConfig(
            model_name=variant.training_config.model_name,
            dataset_yaml=self.config.dataset_yaml,
            epochs=variant.training_config.epochs,
            batch_size=variant.training_config.batch_size,
            img_size=variant.training_config.img_size,
            learning_rate=variant.training_config.learning_rate,
            random_seed=variant.training_config.random_seed,
            output_dir=variant_output_dir,
            device=variant.training_config.device,
            workers=variant.training_config.workers
        )
        
        # Load baseline model
        base_model = YOLO(f"{variant.training_config.model_name}.pt")
        
        # Apply enhancements if any are enabled
        if any([
            variant.enhancement_config.enable_gam,
            variant.enhancement_config.enable_cbam,
            variant.enhancement_config.enable_c3k2_arc,
            variant.enhancement_config.enable_cl_concat
        ]):
            self.logger.info(f"Applying enhancements to {variant.name}:")
            self.logger.info(f"  GAM: {variant.enhancement_config.enable_gam}")
            self.logger.info(f"  CBAM: {variant.enhancement_config.enable_cbam}")
            self.logger.info(f"  C3k2-ARC: {variant.enhancement_config.enable_c3k2_arc}")
            self.logger.info(f"  CL-Concat: {variant.enhancement_config.enable_cl_concat}")
            
            enhancement_module = EnhancementModule(variant.enhancement_config)
            base_model = enhancement_module.build_enhanced_model(base_model)
            
            # Log complexity
            complexity = enhancement_module.compute_complexity(base_model)
            self.logger.info(f"Enhanced model complexity: {complexity.parameters:,} params, {complexity.gflops:.2f} GFLOPs")
        else:
            self.logger.info(f"Training baseline model (no enhancements)")
        
        # Train using TrainingPipeline
        pipeline = TrainingPipeline(variant_training_config)
        pipeline.model = base_model  # Use our prepared model
        training_result = pipeline.train(resume=False)
        
        self.logger.info(f"Training completed: {training_result.best_checkpoint}")
        self.logger.info(f"Best mAP@50: {training_result.best_map50:.4f}")
        self.logger.info(f"Training duration: {training_result.total_duration_seconds:.2f}s")
        
        return training_result.best_checkpoint
    
    def evaluate_variant(
        self,
        variant_name: str,
        checkpoint: Path
    ) -> VariantMetrics:
        """Evaluate a single variant on test set.
        
        This method:
        1. Loads trained checkpoint
        2. Runs evaluation using Evaluator
        3. Collects all required metrics
        4. Returns consolidated VariantMetrics
        
        Args:
            variant_name: Human-readable variant name
            checkpoint: Path to trained model checkpoint
            
        Returns:
            VariantMetrics with all evaluation results
            
        Raises:
            Exception: If evaluation fails
            
        **Validates: Requirement 4.3** - Collect comprehensive metrics
        """
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        
        # Configure evaluator
        evaluator_config = EvaluatorConfig(
            checkpoint_path=checkpoint,
            test_dataset_path=self.config.test_subset_path,
            device="cpu",  # Use CPU for consistency
            conf_threshold=0.25,
            iou_threshold=0.5,
            img_size=320
        )
        
        # Run evaluation
        evaluator = Evaluator(evaluator_config)
        evaluation_result = evaluator.evaluate()
        
        # Extract metrics
        detection = evaluation_result.detection_metrics
        efficiency = evaluation_result.efficiency_metrics
        complexity = evaluation_result.complexity_metrics
        
        self.logger.info(f"Evaluation results for {variant_name}:")
        self.logger.info(f"  mAP@50: {detection.map50:.4f}")
        self.logger.info(f"  mAP@50:95: {detection.map50_95:.4f}")
        self.logger.info(f"  Small Object AP: {detection.small_object_ap:.4f}")
        self.logger.info(f"  FPS (mean): {efficiency.fps_mean:.2f}")
        self.logger.info(f"  Latency (mean): {efficiency.latency_ms_mean:.2f} ms")
        self.logger.info(f"  Latency (p95): {efficiency.latency_ms_p95:.2f} ms")
        self.logger.info(f"  Parameters: {complexity.parameters:,}")
        self.logger.info(f"  GFLOPs: {complexity.gflops:.2f}")
        
        # Get training duration from checkpoint directory
        training_duration = self._get_training_duration(checkpoint)
        
        # Create VariantMetrics
        metrics = VariantMetrics(
            variant_name=variant_name,
            map50=detection.map50,
            map50_95=detection.map50_95,
            small_object_ap=detection.small_object_ap,
            fps_mean=efficiency.fps_mean,
            latency_ms_mean=efficiency.latency_ms_mean,
            latency_ms_p95=efficiency.latency_ms_p95,
            parameters=complexity.parameters,
            gflops=complexity.gflops,
            training_duration_seconds=training_duration,
            status="success",
            error_message=None
        )
        
        return metrics
    
    def _get_training_duration(self, checkpoint: Path) -> float:
        """Extract training duration from checkpoint directory.
        
        Args:
            checkpoint: Path to model checkpoint
            
        Returns:
            Training duration in seconds (0.0 if not found)
        """
        try:
            # Look for metrics_history.json in the checkpoint's parent directory
            metrics_file = checkpoint.parent.parent / "metrics_history.json"
            if metrics_file.exists():
                import json
                with open(metrics_file, 'r') as f:
                    data = json.load(f)
                    return data.get('total_duration_seconds', 0.0)
        except Exception as e:
            self.logger.warning(f"Could not extract training duration: {e}")
        
        return 0.0
    
    def export_results(self, results: AblationResults) -> None:
        """Export results to CSV and Markdown files.
        
        Args:
            results: AblationResults with metrics for all variants
            
        **Validates: Requirement 4.4** - Export results in CSV and Markdown formats
        """
        # Export CSV
        csv_path = self.config.output_dir / "ablation_results.csv"
        self._export_csv(results, csv_path)
        self.logger.info(f"Results exported to CSV: {csv_path}")
        
        # Export Markdown
        markdown_path = self.config.output_dir / "ablation_results.md"
        self._export_markdown(results, markdown_path)
        self.logger.info(f"Results exported to Markdown: {markdown_path}")
    
    def _export_csv(self, results: AblationResults, output_path: Path) -> None:
        """Export results to CSV file.
        
        Args:
            results: AblationResults to export
            output_path: Path for CSV file
        """
        fieldnames = [
            'variant_name',
            'status',
            'map50',
            'map50_95',
            'small_object_ap',
            'fps_mean',
            'latency_ms_mean',
            'latency_ms_p95',
            'parameters',
            'gflops',
            'training_duration_seconds',
            'error_message'
        ]
        
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for variant_metrics in results.variants:
                row = asdict(variant_metrics)
                writer.writerow(row)
    
    def _export_markdown(self, results: AblationResults, output_path: Path) -> None:
        """Export results to Markdown file.
        
        Args:
            results: AblationResults to export
            output_path: Path for Markdown file
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Ablation Study Results\n\n")
            f.write(f"**Timestamp:** {results.timestamp}\n\n")
            f.write(f"**Total Variants:** {len(results.variants)}\n\n")
            
            # Summary statistics
            successful = [v for v in results.variants if v.status == "success"]
            failed = [v for v in results.variants if v.status == "failed"]
            
            f.write(f"**Successful:** {len(successful)}\n\n")
            f.write(f"**Failed:** {len(failed)}\n\n")
            
            # Results table
            f.write("## Performance Comparison\n\n")
            
            # Header
            f.write("| Variant | Status | mAP@50 | mAP@50:95 | Small Obj AP | FPS | Latency (ms) | Params | GFLOPs |\n")
            f.write("|---------|--------|--------|-----------|--------------|-----|--------------|--------|--------|\n")
            
            # Rows
            for variant in results.variants:
                if variant.status == "success":
                    f.write(
                        f"| {variant.variant_name} | ✓ | "
                        f"{variant.map50:.4f} | {variant.map50_95:.4f} | {variant.small_object_ap:.4f} | "
                        f"{variant.fps_mean:.2f} | {variant.latency_ms_mean:.2f} | "
                        f"{variant.parameters:,} | {variant.gflops:.2f} |\n"
                    )
                else:
                    f.write(
                        f"| {variant.variant_name} | ✗ | "
                        f"- | - | - | - | - | - | - |\n"
                    )
            
            # Failed variants details
            if failed:
                f.write("\n## Failed Variants\n\n")
                for variant in failed:
                    f.write(f"### {variant.variant_name}\n\n")
                    f.write(f"**Error:** {variant.error_message}\n\n")
            
            # Training duration summary
            if successful:
                f.write("\n## Training Duration\n\n")
                f.write("| Variant | Duration (seconds) |\n")
                f.write("|---------|--------------------|\n")
                for variant in successful:
                    f.write(f"| {variant.variant_name} | {variant.training_duration_seconds:.2f} |\n")
