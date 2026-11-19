"""Training pipeline implementation for YOLO11n on C2A dataset."""

import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List, Dict

import torch
from ultralytics import YOLO

from ..config.models import TrainingConfig


@dataclass
class TrainingResult:
    """Results from a training session.
    
    Attributes:
        best_checkpoint: Path to best model checkpoint (highest mAP@50)
        final_checkpoint: Path to final model checkpoint
        metrics_history: Per-epoch metrics dictionary
        total_duration_seconds: Total training time in seconds
        best_map50: Best validation mAP@50 achieved
    """
    best_checkpoint: Path
    final_checkpoint: Path
    metrics_history: List[Dict[str, float]]
    total_duration_seconds: float
    best_map50: float


class TrainingPipeline:
    """Training pipeline for YOLO11n model on C2A dataset.
    
    This pipeline handles:
    - Loading pretrained YOLO11n weights from ultralytics hub
    - Training with configurable hyperparameters
    - Per-epoch metric logging to console and file
    - Best checkpoint saving based on validation mAP@50
    - Training resumption from checkpoint
    - Reproducibility through fixed random seed
    
    Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7
    """
    
    def __init__(self, config: TrainingConfig):
        """Initialize training pipeline.
        
        Args:
            config: Training configuration with model architecture, dataset path,
                   hyperparameters, and random seed
        """
        self.config = config
        self.model: Optional[YOLO] = None
        self.best_map50 = 0.0
        self.metrics_history: List[Dict[str, float]] = []
        
        # Set up logging
        self._setup_logging()
        
        # Create output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"TrainingPipeline initialized with config: {config}")
    
    def _setup_logging(self) -> None:
        """Configure logging to file and console."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        log_file = self.config.output_dir / "training.log"
        
        # Create logger
        self.logger = logging.getLogger("TrainingPipeline")
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers to avoid duplicates
        self.logger.handlers.clear()
        
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
    
    def load_pretrained_weights(self, model_name: str = "yolo11n.pt") -> YOLO:
        """Load YOLO11n pretrained weights from ultralytics hub.
        
        Args:
            model_name: Model architecture name (default: "yolo11n.pt")
        
        Returns:
            YOLO model instance with pretrained weights
        
        Requirements: 2.2
        """
        self.logger.info(f"Loading pretrained weights: {model_name}")
        
        try:
            model = YOLO(model_name)
            self.logger.info(f"Successfully loaded pretrained model: {model_name}")
            return model
        except Exception as e:
            self.logger.error(f"Failed to load pretrained weights: {e}")
            raise
    
    def train(
        self,
        resume: bool = False,
        checkpoint_path: Optional[Path] = None
    ) -> TrainingResult:
        """Train model on C2A dataset.
        
        This method:
        - Sets random seed for reproducibility
        - Loads pretrained or checkpoint weights
        - Trains model with configured hyperparameters
        - Logs per-epoch metrics
        - Saves best checkpoint based on mAP@50
        
        Args:
            resume: If True, resume from checkpoint_path
            checkpoint_path: Path to checkpoint for resuming (required if resume=True)
        
        Returns:
            TrainingResult with metrics history and checkpoint paths
        
        Raises:
            FileNotFoundError: If resume=True and checkpoint_path doesn't exist
            ValueError: If resume=True but checkpoint_path is None
        
        Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7
        """
        # Set random seed for reproducibility (Requirement 2.7)
        torch.manual_seed(self.config.random_seed)
        self.logger.info(f"Set random seed to {self.config.random_seed}")
        
        # Handle resumption (Requirement 2.6)
        if resume:
            if checkpoint_path is None:
                raise ValueError("checkpoint_path must be provided when resume=True")
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
            self.logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
            self.model = YOLO(str(checkpoint_path))
        elif self.model is not None:
            # A pre-built model was injected by the caller (e.g. the ablation
            # runner sets ``pipeline.model = enhanced_model`` before calling
            # train()). Preserve it instead of overwriting with fresh weights,
            # otherwise any applied enhancements (GAM/CBAM/...) are discarded.
            self.logger.info("Using caller-provided model (enhancements preserved)")
        else:
            # Load pretrained weights (Requirement 2.2)
            model_file = f"{self.config.model_name}.pt"
            self.model = self.load_pretrained_weights(model_file)
        
        # Verify dataset YAML exists
        if not self.config.dataset_yaml.exists():
            raise FileNotFoundError(f"Dataset YAML not found: {self.config.dataset_yaml}")
        
        self.logger.info("Starting training...")
        self.logger.info(f"Dataset: {self.config.dataset_yaml}")
        self.logger.info(f"Epochs: {self.config.epochs}")
        self.logger.info(f"Batch size: {self.config.batch_size}")
        self.logger.info(f"Image size: {self.config.img_size}")
        self.logger.info(f"Learning rate: {self.config.learning_rate}")
        self.logger.info(f"Device: {self.config.device}")
        
        # Record start time
        start_time = time.time()
        
        # Train model using ultralytics API (Requirement 2.3)
        try:
            # Resolve output_dir to an absolute path so ultralytics doesn't
            # nest it under its default 'runs/detect/' prefix.
            abs_output_dir = self.config.output_dir.resolve()

            results = self.model.train(
                data=str(self.config.dataset_yaml.resolve()),
                epochs=self.config.epochs,
                batch=self.config.batch_size,
                imgsz=self.config.img_size,
                lr0=self.config.learning_rate,
                device=self.config.device,
                workers=self.config.workers,
                project=str(abs_output_dir),
                name="train",
                exist_ok=True,
                verbose=True,
                seed=self.config.random_seed,
            )
            
            # Calculate training duration
            total_duration = time.time() - start_time
            
            self.logger.info(f"Training completed in {total_duration:.2f} seconds")
            
            # Extract metrics history from results
            # The ultralytics results object contains training metrics
            self._extract_metrics_history(results)
            
            # Determine checkpoint paths
            # Ultralytics saves checkpoints in project/name/weights/
            weights_dir = self.config.output_dir / "train" / "weights"
            best_checkpoint = weights_dir / "best.pt"
            final_checkpoint = weights_dir / "last.pt"
            
            # Verify checkpoints exist
            if not best_checkpoint.exists():
                self.logger.warning(f"Best checkpoint not found at {best_checkpoint}")
                best_checkpoint = final_checkpoint
            
            if not final_checkpoint.exists():
                raise FileNotFoundError(f"Final checkpoint not found at {final_checkpoint}")
            
            self.logger.info(f"Best checkpoint saved: {best_checkpoint}")
            self.logger.info(f"Final checkpoint saved: {final_checkpoint}")
            self.logger.info(f"Best mAP@50: {self.best_map50:.4f}")
            
            # Create training result
            training_result = TrainingResult(
                best_checkpoint=best_checkpoint,
                final_checkpoint=final_checkpoint,
                metrics_history=self.metrics_history,
                total_duration_seconds=total_duration,
                best_map50=self.best_map50
            )
            
            # Export metrics history to JSON
            self._export_metrics_json(training_result)
            
            return training_result
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
    
    def _extract_metrics_history(self, results) -> None:
        """Extract per-epoch metrics from ultralytics results.
        
        Args:
            results: Ultralytics training results object
        
        Requirements: 2.3, 2.4
        """
        # Ultralytics stores metrics in a CSV file
        results_csv = self.config.output_dir / "train" / "results.csv"
        
        if results_csv.exists():
            import csv
            
            with open(results_csv, 'r') as f:
                reader = csv.DictReader(f)
                reader.fieldnames = [name.strip() for name in reader.fieldnames]
                
                for row in reader:
                    # Extract key metrics
                    epoch_metrics = {}
                    
                    # Map CSV columns to our metric names
                    # Common ultralytics column names (may vary by version)
                    column_mapping = {
                        'epoch': 'epoch',
                        'train/box_loss': 'train_box_loss',
                        'train/cls_loss': 'train_cls_loss',
                        'train/dfl_loss': 'train_dfl_loss',
                        'metrics/precision(B)': 'precision',
                        'metrics/recall(B)': 'recall',
                        'metrics/mAP50(B)': 'map50',
                        'metrics/mAP50-95(B)': 'map50_95',
                        'val/box_loss': 'val_box_loss',
                        'val/cls_loss': 'val_cls_loss',
                        'val/dfl_loss': 'val_dfl_loss',
                    }
                    
                    for csv_col, metric_name in column_mapping.items():
                        if csv_col in row and row[csv_col]:
                            try:
                                epoch_metrics[metric_name] = float(row[csv_col])
                            except ValueError:
                                pass
                    
                    if epoch_metrics:
                        self.metrics_history.append(epoch_metrics)
                        
                        # Track best mAP@50
                        if 'map50' in epoch_metrics:
                            if epoch_metrics['map50'] > self.best_map50:
                                self.best_map50 = epoch_metrics['map50']
            
            self.logger.info(f"Extracted {len(self.metrics_history)} epochs of metrics")
        else:
            self.logger.warning(f"Results CSV not found at {results_csv}")
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Callback invoked after each epoch.
        
        Logs metrics and saves checkpoint if validation mAP@50 improved.
        
        Note: This method is provided for potential custom training loop integration.
        The ultralytics API handles epoch callbacks internally, so this is primarily
        for documentation and potential future use.
        
        Args:
            epoch: Current epoch number
            metrics: Dictionary of metrics for this epoch
        
        Requirements: 2.3, 2.4
        """
        # Log metrics to console
        self.logger.info(f"Epoch {epoch} metrics: {metrics}")
        
        # Track metrics history
        self.metrics_history.append(metrics)
        
        # Check if mAP@50 improved (Requirement 2.4)
        if 'map50' in metrics:
            current_map50 = metrics['map50']
            if current_map50 > self.best_map50:
                self.best_map50 = current_map50
                self.logger.info(f"New best mAP@50: {self.best_map50:.4f}")
                
                # Save checkpoint
                if self.model is not None:
                    checkpoint_path = self.config.output_dir / f"best_epoch_{epoch}.pt"
                    self.model.save(str(checkpoint_path))
                    self.logger.info(f"Saved best checkpoint: {checkpoint_path}")
    
    def export_model(self, output_path: Path, format: str = "pytorch") -> None:
        """Export trained model in specified format.
        
        Args:
            output_path: Path for exported model
            format: Export format (default: "pytorch")
                   Supported formats: pytorch, onnx, torchscript, etc.
        
        Requirements: 2.5
        """
        if self.model is None:
            raise ValueError("No model loaded. Train or load a model first.")
        
        self.logger.info(f"Exporting model to {output_path} in {format} format")
        
        try:
            if format == "pytorch":
                # Save as PyTorch .pt file
                self.model.save(str(output_path))
                self.logger.info(f"Model exported successfully: {output_path}")
            else:
                # Use ultralytics export API for other formats
                self.model.export(format=format, imgsz=self.config.img_size)
                self.logger.info(f"Model exported successfully in {format} format")
        except Exception as e:
            self.logger.error(f"Model export failed: {e}")
            raise
    
    def _export_metrics_json(self, training_result: TrainingResult) -> None:
        """Export training metrics to JSON file.
        
        Args:
            training_result: Training result with metrics history
        """
        metrics_file = self.config.output_dir / "train" / "metrics_history.json"
        
        # Convert TrainingResult to dictionary for JSON serialization
        result_dict = {
            'best_checkpoint': str(training_result.best_checkpoint),
            'final_checkpoint': str(training_result.final_checkpoint),
            'metrics_history': training_result.metrics_history,
            'total_duration_seconds': training_result.total_duration_seconds,
            'best_map50': training_result.best_map50,
            'config': {
                'model_name': self.config.model_name,
                'epochs': self.config.epochs,
                'batch_size': self.config.batch_size,
                'img_size': self.config.img_size,
                'learning_rate': self.config.learning_rate,
                'random_seed': self.config.random_seed,
            }
        }
        
        with open(metrics_file, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        self.logger.info(f"Metrics history exported to {metrics_file}")
