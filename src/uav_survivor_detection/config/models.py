"""Configuration data models for UAV Survivor Detection System."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Tuple, List


@dataclass
class InferenceConfig:
    """Configuration for real-time inference mode.
    
    Attributes:
        video_source: Device index, RTSP URL, UDP URL, or file path
        model_checkpoint: Path to trained YOLO model checkpoint
        conf_threshold: Confidence threshold for detections (0.0-1.0)
        iou_threshold: IoU threshold for non-maximum suppression (0.0-1.0)
        img_size: Input image size for inference (pixels)
        device: Compute device ('cpu' or 'cuda')
        display_enabled: Enable real-time display window
        logging_enabled: Enable detection logging
        log_output_dir: Directory for log files
        max_reconnect_attempts: Maximum reconnection attempts on stream failure
        reconnect_interval_seconds: Delay between reconnection attempts
        max_frame_buffer_seconds: Maximum frame buffer duration
    """
    video_source: str
    model_checkpoint: Path
    conf_threshold: float = 0.25
    iou_threshold: float = 0.5
    img_size: int = 640
    device: str = "cpu"
    display_enabled: bool = True
    logging_enabled: bool = True
    log_output_dir: Path = Path("logs")
    max_reconnect_attempts: int = 5
    reconnect_interval_seconds: float = 2.0
    max_frame_buffer_seconds: float = 2.0


@dataclass
class DatasetConfig:
    """Configuration for dataset preparation pipeline.
    
    Attributes:
        dataset_path: Path to raw C2A dataset
        output_dir: Directory for processed YOLO-format output
        train_ratio: Fraction of data for training (0.0-1.0)
        val_ratio: Fraction of data for validation (0.0-1.0)
        test_ratio: Fraction of data for testing (0.0-1.0)
        random_seed: Random seed for reproducible splits
        overwrite_existing: Overwrite existing output directory
    """
    dataset_path: Path
    output_dir: Path
    train_ratio: float = 0.7
    val_ratio: float = 0.2
    test_ratio: float = 0.1
    random_seed: int = 42
    overwrite_existing: bool = False


@dataclass
class TrainingConfig:
    """Configuration for model training pipeline.
    
    Attributes:
        model_name: YOLO model architecture name
        dataset_yaml: Path to YOLO dataset YAML file
        epochs: Number of training epochs
        batch_size: Training batch size
        img_size: Input image size (pixels)
        learning_rate: Initial learning rate
        random_seed: Random seed for reproducibility
        output_dir: Directory for training outputs
        device: Compute device ('cpu' or 'cuda')
        workers: Number of data loading workers
    """
    model_name: str = "yolo11n"
    dataset_yaml: Path = Path("data.yaml")
    epochs: int = 100
    batch_size: int = 16
    img_size: int = 640
    learning_rate: float = 0.01
    random_seed: int = 42
    output_dir: Path = Path("runs/train")
    device: str = "cpu"
    workers: int = 4


@dataclass
class EnhancementConfig:
    """Configuration for model enhancement module.
    
    Attributes:
        enable_gam: Enable Global Attention Mechanism
        enable_cbam: Enable Convolutional Block Attention Module
        enable_c3k2_arc: Enable rotation-aware convolution
        enable_cl_concat: Enable cross-level feature concatenation
        gam_reduction: Channel reduction ratio for GAM
        cbam_reduction: Channel reduction ratio for CBAM
    """
    enable_gam: bool = False
    enable_cbam: bool = False
    enable_c3k2_arc: bool = False
    enable_cl_concat: bool = False
    gam_reduction: int = 16
    cbam_reduction: int = 16


@dataclass
class EvaluatorConfig:
    """Configuration for model evaluation.
    
    Attributes:
        checkpoint_path: Path to model checkpoint to evaluate
        test_dataset_path: Path to test dataset
        device: Compute device ('cpu' or 'cuda')
        conf_threshold: Confidence threshold for detections (0.0-1.0)
        iou_threshold: IoU threshold for NMS (0.0-1.0)
        img_size: Input image size (pixels)
    """
    checkpoint_path: Optional[Path] = None
    test_dataset_path: Optional[Path] = None
    device: str = "cpu"
    conf_threshold: float = 0.25
    iou_threshold: float = 0.5
    img_size: int = 640


@dataclass
class ModelVariant:
    """Definition of a single model variant for ablation study.
    
    Attributes:
        name: Human-readable variant name
        enhancement_config: Enhancement configuration for this variant
        training_config: Training configuration for this variant
    """
    name: str
    enhancement_config: EnhancementConfig
    training_config: TrainingConfig


@dataclass
class AblationConfig:
    """Configuration for ablation study runner.
    
    Attributes:
        variants: List of model variants to train and evaluate
        dataset_yaml: Path to YOLO dataset YAML file
        output_dir: Directory for ablation study outputs
        test_subset_path: Path to test dataset subset
    """
    variants: List[ModelVariant]
    dataset_yaml: Path
    output_dir: Path
    test_subset_path: Path


@dataclass
class DisplayConfig:
    """Configuration for display module.
    
    Attributes:
        window_name: Display window title
        bbox_enabled: Enable bounding box rendering
        bbox_thickness: Bounding box line thickness (pixels)
        font_scale: Font size scale factor
        font_thickness: Font line thickness (pixels)
        color_scheme: Class name to BGR color mapping
        quit_key: Keyboard key to quit
        toggle_bbox_key: Keyboard key to toggle bounding boxes
    """
    window_name: str = "UAV Survivor Detection"
    bbox_enabled: bool = True
    bbox_thickness: int = 2
    font_scale: float = 0.5
    font_thickness: int = 1
    color_scheme: Optional[Dict[str, Tuple[int, int, int]]] = None
    quit_key: str = "q"
    toggle_bbox_key: str = "b"


@dataclass
class LoggerConfig:
    """Configuration for detection logger.
    
    Attributes:
        output_dir: Directory for log files
        flush_interval_seconds: Interval for flushing logs to disk
    """
    output_dir: Path = Path("logs")
    flush_interval_seconds: float = 5.0


@dataclass
class DetectionEngineConfig:
    """Configuration for detection engine.
    
    Attributes:
        checkpoint_path: Path to model checkpoint
        conf_threshold: Confidence threshold for detections (0.0-1.0)
        iou_threshold: IoU threshold for NMS (0.0-1.0)
        img_size: Input image size (pixels)
        device: Compute device ('cpu' or 'cuda')
        num_threads: Number of CPU threads for inference
    """
    checkpoint_path: Optional[Path] = None
    conf_threshold: float = 0.25
    iou_threshold: float = 0.5
    img_size: int = 640
    device: str = "cpu"
    num_threads: int = 4


@dataclass
class SystemConfig:
    """Top-level system configuration.
    
    Attributes:
        inference: Inference mode configuration
        dataset: Dataset preparation configuration
        training: Training pipeline configuration
        enhancement: Model enhancement configuration
        ablation: Ablation study configuration
        evaluation: Evaluation configuration
        display: Display module configuration
        logging: Logger configuration
        detection_engine: Detection engine configuration
    """
    # Inference mode settings
    inference: Optional[InferenceConfig] = None
    
    # Training mode settings
    dataset: Optional[DatasetConfig] = None
    training: Optional[TrainingConfig] = None
    enhancement: Optional[EnhancementConfig] = None
    ablation: Optional[AblationConfig] = None
    evaluation: Optional[EvaluatorConfig] = None
    
    # Shared settings
    display: DisplayConfig = field(default_factory=DisplayConfig)
    logging: LoggerConfig = field(default_factory=LoggerConfig)
    detection_engine: DetectionEngineConfig = field(default_factory=DetectionEngineConfig)
