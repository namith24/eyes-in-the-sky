"""Configuration manager for loading, validating, and applying CLI overrides."""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

from .models import (
    SystemConfig,
    InferenceConfig,
    DatasetConfig,
    TrainingConfig,
    EnhancementConfig,
    EvaluatorConfig,
    AblationConfig,
    DisplayConfig,
    LoggerConfig,
    DetectionEngineConfig,
    ModelVariant,
)


class ConfigurationError(Exception):
    """Exception raised for configuration validation errors."""
    pass


class ConfigurationManager:
    """Manages loading, validation, and CLI overrides for system configuration.
    
    This class is responsible for:
    - Loading YAML configuration files
    - Validating configuration parameters
    - Applying command-line argument overrides
    - Providing typed configuration objects to all modules
    """
    
    def __init__(self, config_path: Path):
        """Initialize configuration manager.
        
        Args:
            config_path: Path to YAML configuration file
            
        Raises:
            ConfigurationError: If config file does not exist
        """
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise ConfigurationError(f"Configuration file not found: {config_path}")
    
    def load_config(self) -> SystemConfig:
        """Load and validate YAML configuration.
        
        Returns:
            SystemConfig object with all settings
            
        Raises:
            ConfigurationError: If validation fails
        """
        # Load YAML file
        with open(self.config_path, 'r') as f:
            raw_config = yaml.safe_load(f)
        
        if raw_config is None:
            raise ConfigurationError("Configuration file is empty")
        
        # Parse configuration sections
        config = self._parse_config(raw_config)
        
        # Validate configuration
        errors = self.validate_config(config)
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            raise ConfigurationError(error_msg)
        
        return config
    
    def _parse_config(self, raw_config: Dict[str, Any]) -> SystemConfig:
        """Parse raw YAML dictionary into SystemConfig object.
        
        Args:
            raw_config: Raw configuration dictionary from YAML
            
        Returns:
            SystemConfig object
        """
        # Parse inference config
        inference = None
        if "inference" in raw_config and raw_config["inference"]:
            inf_data = raw_config["inference"]
            inference = InferenceConfig(
                video_source=inf_data["video_source"],
                model_checkpoint=Path(inf_data["model_checkpoint"]),
                conf_threshold=inf_data.get("conf_threshold", 0.25),
                iou_threshold=inf_data.get("iou_threshold", 0.5),
                img_size=inf_data.get("img_size", 640),
                device=inf_data.get("device", "cpu"),
                display_enabled=inf_data.get("display_enabled", True),
                logging_enabled=inf_data.get("logging_enabled", True),
                log_output_dir=Path(inf_data.get("log_output_dir", "logs")),
                max_reconnect_attempts=inf_data.get("max_reconnect_attempts", 5),
                reconnect_interval_seconds=inf_data.get("reconnect_interval_seconds", 2.0),
                max_frame_buffer_seconds=inf_data.get("max_frame_buffer_seconds", 2.0),
            )
        
        # Parse dataset config
        dataset = None
        if "dataset" in raw_config and raw_config["dataset"]:
            ds_data = raw_config["dataset"]
            dataset = DatasetConfig(
                dataset_path=Path(ds_data["dataset_path"]),
                output_dir=Path(ds_data["output_dir"]),
                train_ratio=ds_data.get("train_ratio", 0.7),
                val_ratio=ds_data.get("val_ratio", 0.2),
                test_ratio=ds_data.get("test_ratio", 0.1),
                random_seed=ds_data.get("random_seed", 42),
                overwrite_existing=ds_data.get("overwrite_existing", False),
            )
        
        # Parse training config
        training = None
        if "training" in raw_config and raw_config["training"]:
            tr_data = raw_config["training"]
            training = TrainingConfig(
                model_name=tr_data.get("model_name", "yolo11n"),
                dataset_yaml=Path(tr_data["dataset_yaml"]),
                epochs=tr_data.get("epochs", 100),
                batch_size=tr_data.get("batch_size", 16),
                img_size=tr_data.get("img_size", 640),
                learning_rate=tr_data.get("learning_rate", 0.01),
                random_seed=tr_data.get("random_seed", 42),
                output_dir=Path(tr_data.get("output_dir", "runs/train")),
                device=tr_data.get("device", "cpu"),
                workers=tr_data.get("workers", 4),
            )
        
        # Parse enhancement config
        enhancement = None
        if "enhancement" in raw_config and raw_config["enhancement"]:
            enh_data = raw_config["enhancement"]
            enhancement = EnhancementConfig(
                enable_gam=enh_data.get("enable_gam", False),
                enable_cbam=enh_data.get("enable_cbam", False),
                enable_c3k2_arc=enh_data.get("enable_c3k2_arc", False),
                enable_cl_concat=enh_data.get("enable_cl_concat", False),
                gam_reduction=enh_data.get("gam_reduction", 16),
                cbam_reduction=enh_data.get("cbam_reduction", 16),
            )
        
        # Parse evaluation config
        evaluation = None
        if "evaluation" in raw_config and raw_config["evaluation"]:
            eval_data = raw_config["evaluation"]
            # checkpoint_path and test_dataset_path are optional (can be set later)
            checkpoint_path = Path(eval_data["checkpoint_path"]) if "checkpoint_path" in eval_data else None
            test_dataset_path = Path(eval_data["test_dataset_path"]) if "test_dataset_path" in eval_data else None
            
            if checkpoint_path or test_dataset_path:
                evaluation = EvaluatorConfig(
                    checkpoint_path=checkpoint_path,
                    test_dataset_path=test_dataset_path,
                    device=eval_data.get("device", "cpu"),
                    conf_threshold=eval_data.get("conf_threshold", 0.25),
                    iou_threshold=eval_data.get("iou_threshold", 0.5),
                    img_size=eval_data.get("img_size", 640),
                )
        
        # Parse ablation config
        ablation = None
        if "ablation" in raw_config and raw_config["ablation"]:
            abl_data = raw_config["ablation"]
            variants = []
            
            # Get shared training config from top-level "training" section
            shared_training = training if training else None
            
            if "variants" in abl_data:
                for var_data in abl_data["variants"]:
                    # Parse enhancement config (may be nested under "enhancement" key)
                    enh_data = var_data.get("enhancement", var_data)
                    enh_cfg = EnhancementConfig(
                        enable_gam=enh_data.get("enable_gam", False),
                        enable_cbam=enh_data.get("enable_cbam", False),
                        enable_c3k2_arc=enh_data.get("enable_c3k2_arc", False),
                        enable_cl_concat=enh_data.get("enable_cl_concat", False),
                        gam_reduction=enh_data.get("gam_reduction", 16),
                        cbam_reduction=enh_data.get("cbam_reduction", 16),
                    )
                    
                    # Use variant-specific training config if provided, otherwise use shared
                    if "dataset_yaml" in var_data:
                        # Variant has its own complete training config
                        tr_cfg = TrainingConfig(
                            model_name=var_data.get("model_name", "yolo11n"),
                            dataset_yaml=Path(var_data["dataset_yaml"]),
                            epochs=var_data.get("epochs", 100),
                            batch_size=var_data.get("batch_size", 16),
                            img_size=var_data.get("img_size", 640),
                            learning_rate=var_data.get("learning_rate", 0.01),
                            random_seed=var_data.get("random_seed", 42),
                            output_dir=Path(var_data.get("output_dir", "runs/train")),
                            device=var_data.get("device", "cpu"),
                            workers=var_data.get("workers", 4),
                        )
                    elif shared_training:
                        # Use shared training config from top level
                        tr_cfg = shared_training
                    else:
                        raise ConfigurationError(f"Ablation variant '{var_data.get('name')}' missing training config")
                    
                    variants.append(ModelVariant(
                        name=var_data["name"],
                        enhancement_config=enh_cfg,
                        training_config=tr_cfg,
                    ))
            
            ablation = AblationConfig(
                variants=variants,
                dataset_yaml=Path(abl_data["dataset_yaml"]) if "dataset_yaml" in abl_data else (training.dataset_yaml if training else Path("data.yaml")),
                output_dir=Path(abl_data["output_dir"]),
                test_subset_path=Path(abl_data["test_subset_path"]),
            )
        
        # Parse display config
        display = DisplayConfig()
        if "display" in raw_config and raw_config["display"]:
            disp_data = raw_config["display"]
            display = DisplayConfig(
                window_name=disp_data.get("window_name", "UAV Survivor Detection"),
                bbox_enabled=disp_data.get("bbox_enabled", True),
                bbox_thickness=disp_data.get("bbox_thickness", 2),
                font_scale=disp_data.get("font_scale", 0.5),
                font_thickness=disp_data.get("font_thickness", 1),
                color_scheme=disp_data.get("color_scheme"),
                quit_key=disp_data.get("quit_key", "q"),
                toggle_bbox_key=disp_data.get("toggle_bbox_key", "b"),
            )
        
        # Parse logging config
        logging = LoggerConfig()
        if "logging" in raw_config and raw_config["logging"]:
            log_data = raw_config["logging"]
            logging = LoggerConfig(
                output_dir=Path(log_data.get("output_dir", "logs")),
                flush_interval_seconds=log_data.get("flush_interval_seconds", 5.0),
            )
        
        # Parse detection engine config
        detection_engine = DetectionEngineConfig()
        if "detection_engine" in raw_config and raw_config["detection_engine"]:
            de_data = raw_config["detection_engine"]
            detection_engine = DetectionEngineConfig(
                checkpoint_path=Path(de_data["checkpoint_path"]) if "checkpoint_path" in de_data else None,
                conf_threshold=de_data.get("conf_threshold", 0.25),
                iou_threshold=de_data.get("iou_threshold", 0.5),
                img_size=de_data.get("img_size", 640),
                device=de_data.get("device", "cpu"),
                num_threads=de_data.get("num_threads", 4),
            )
        
        return SystemConfig(
            inference=inference,
            dataset=dataset,
            training=training,
            enhancement=enhancement,
            ablation=ablation,
            evaluation=evaluation,
            display=display,
            logging=logging,
            detection_engine=detection_engine,
        )
    
    def validate_config(self, config: SystemConfig) -> List[str]:
        """Validate configuration parameters.
        
        Args:
            config: SystemConfig to validate
            
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Validate inference config
        if config.inference:
            errors.extend(self._validate_inference_config(config.inference))
        
        # Validate dataset config
        if config.dataset:
            errors.extend(self._validate_dataset_config(config.dataset))
        
        # Validate training config
        if config.training:
            errors.extend(self._validate_training_config(config.training))
        
        # Validate enhancement config
        if config.enhancement:
            errors.extend(self._validate_enhancement_config(config.enhancement))
        
        # Validate evaluation config
        if config.evaluation:
            errors.extend(self._validate_evaluator_config(config.evaluation))
        
        # Validate ablation config
        if config.ablation:
            errors.extend(self._validate_ablation_config(config.ablation))
        
        # Validate display config
        errors.extend(self._validate_display_config(config.display))
        
        # Validate logging config
        errors.extend(self._validate_logger_config(config.logging))
        
        # Validate detection engine config
        errors.extend(self._validate_detection_engine_config(config.detection_engine))
        
        return errors
    
    def _validate_inference_config(self, config: InferenceConfig) -> List[str]:
        """Validate inference configuration."""
        errors = []
        
        if not config.video_source:
            errors.append("inference.video_source is required")
        
        if not 0.0 <= config.conf_threshold <= 1.0:
            errors.append(f"inference.conf_threshold must be in [0.0, 1.0], got {config.conf_threshold}")
        
        if not 0.0 <= config.iou_threshold <= 1.0:
            errors.append(f"inference.iou_threshold must be in [0.0, 1.0], got {config.iou_threshold}")
        
        if config.img_size <= 0:
            errors.append(f"inference.img_size must be > 0, got {config.img_size}")
        
        if config.max_reconnect_attempts < 0:
            errors.append(f"inference.max_reconnect_attempts must be >= 0, got {config.max_reconnect_attempts}")
        
        if config.reconnect_interval_seconds <= 0:
            errors.append(f"inference.reconnect_interval_seconds must be > 0, got {config.reconnect_interval_seconds}")
        
        return errors
    
    def _validate_dataset_config(self, config: DatasetConfig) -> List[str]:
        """Validate dataset configuration."""
        errors = []
        
        # Validate split ratios sum to ~1.0
        ratio_sum = config.train_ratio + config.val_ratio + config.test_ratio
        if not (0.99 <= ratio_sum <= 1.01):
            errors.append(
                f"dataset split ratios must sum to 1.0 (within 0.01 tolerance), "
                f"got {ratio_sum:.3f} (train={config.train_ratio}, val={config.val_ratio}, test={config.test_ratio})"
            )
        
        # Validate individual ratios
        if not 0.0 <= config.train_ratio <= 1.0:
            errors.append(f"dataset.train_ratio must be in [0.0, 1.0], got {config.train_ratio}")
        
        if not 0.0 <= config.val_ratio <= 1.0:
            errors.append(f"dataset.val_ratio must be in [0.0, 1.0], got {config.val_ratio}")
        
        if not 0.0 <= config.test_ratio <= 1.0:
            errors.append(f"dataset.test_ratio must be in [0.0, 1.0], got {config.test_ratio}")
        
        return errors
    
    def _validate_training_config(self, config: TrainingConfig) -> List[str]:
        """Validate training configuration."""
        errors = []
        
        if config.epochs <= 0:
            errors.append(f"training.epochs must be > 0, got {config.epochs}")
        
        if config.batch_size <= 0:
            errors.append(f"training.batch_size must be > 0, got {config.batch_size}")
        
        if config.img_size <= 0:
            errors.append(f"training.img_size must be > 0, got {config.img_size}")
        
        if config.learning_rate <= 0:
            errors.append(f"training.learning_rate must be > 0, got {config.learning_rate}")
        
        if config.workers < 0:
            errors.append(f"training.workers must be >= 0, got {config.workers}")
        
        return errors
    
    def _validate_enhancement_config(self, config: EnhancementConfig) -> List[str]:
        """Validate enhancement configuration."""
        errors = []
        
        if config.gam_reduction <= 0:
            errors.append(f"enhancement.gam_reduction must be > 0, got {config.gam_reduction}")
        
        if config.cbam_reduction <= 0:
            errors.append(f"enhancement.cbam_reduction must be > 0, got {config.cbam_reduction}")
        
        return errors
    
    def _validate_evaluator_config(self, config: EvaluatorConfig) -> List[str]:
        """Validate evaluator configuration."""
        errors = []
        
        if not 0.0 <= config.conf_threshold <= 1.0:
            errors.append(f"evaluation.conf_threshold must be in [0.0, 1.0], got {config.conf_threshold}")
        
        if not 0.0 <= config.iou_threshold <= 1.0:
            errors.append(f"evaluation.iou_threshold must be in [0.0, 1.0], got {config.iou_threshold}")
        
        if config.img_size <= 0:
            errors.append(f"evaluation.img_size must be > 0, got {config.img_size}")
        
        return errors
    
    def _validate_ablation_config(self, config: AblationConfig) -> List[str]:
        """Validate ablation configuration."""
        errors = []
        
        if not config.variants:
            errors.append("ablation.variants must contain at least one variant")
        
        # Validate each variant
        for i, variant in enumerate(config.variants):
            if not variant.name:
                errors.append(f"ablation.variants[{i}].name is required")
            
            # Validate variant's training config
            errors.extend([f"ablation.variants[{i}].{e}" for e in self._validate_training_config(variant.training_config)])
            
            # Validate variant's enhancement config
            errors.extend([f"ablation.variants[{i}].{e}" for e in self._validate_enhancement_config(variant.enhancement_config)])
        
        return errors
    
    def _validate_display_config(self, config: DisplayConfig) -> List[str]:
        """Validate display configuration."""
        errors = []
        
        if config.bbox_thickness <= 0:
            errors.append(f"display.bbox_thickness must be > 0, got {config.bbox_thickness}")
        
        if config.font_scale <= 0:
            errors.append(f"display.font_scale must be > 0, got {config.font_scale}")
        
        if config.font_thickness <= 0:
            errors.append(f"display.font_thickness must be > 0, got {config.font_thickness}")
        
        return errors
    
    def _validate_logger_config(self, config: LoggerConfig) -> List[str]:
        """Validate logger configuration."""
        errors = []
        
        if config.flush_interval_seconds <= 0:
            errors.append(f"logging.flush_interval_seconds must be > 0, got {config.flush_interval_seconds}")
        
        return errors
    
    def _validate_detection_engine_config(self, config: DetectionEngineConfig) -> List[str]:
        """Validate detection engine configuration."""
        errors = []
        
        if not 0.0 <= config.conf_threshold <= 1.0:
            errors.append(f"detection_engine.conf_threshold must be in [0.0, 1.0], got {config.conf_threshold}")
        
        if not 0.0 <= config.iou_threshold <= 1.0:
            errors.append(f"detection_engine.iou_threshold must be in [0.0, 1.0], got {config.iou_threshold}")
        
        if config.img_size <= 0:
            errors.append(f"detection_engine.img_size must be > 0, got {config.img_size}")
        
        if config.num_threads <= 0:
            errors.append(f"detection_engine.num_threads must be > 0, got {config.num_threads}")
        
        return errors
    
    def apply_cli_overrides(
        self,
        config: SystemConfig,
        cli_args: argparse.Namespace
    ) -> SystemConfig:
        """Apply command-line argument overrides to configuration.
        
        CLI arguments take precedence over YAML values. Only non-None CLI arguments
        are applied as overrides.
        
        Args:
            config: Base configuration from YAML
            cli_args: Parsed command-line arguments
            
        Returns:
            Updated SystemConfig with CLI overrides applied
        """
        # Apply inference overrides
        if config.inference and hasattr(cli_args, 'video_source') and cli_args.video_source is not None:
            config.inference.video_source = cli_args.video_source
        
        if config.inference and hasattr(cli_args, 'model_checkpoint') and cli_args.model_checkpoint is not None:
            config.inference.model_checkpoint = Path(cli_args.model_checkpoint)
        
        if config.inference and hasattr(cli_args, 'conf_threshold') and cli_args.conf_threshold is not None:
            config.inference.conf_threshold = cli_args.conf_threshold
        
        if config.inference and hasattr(cli_args, 'iou_threshold') and cli_args.iou_threshold is not None:
            config.inference.iou_threshold = cli_args.iou_threshold
        
        if config.inference and hasattr(cli_args, 'device') and cli_args.device is not None:
            config.inference.device = cli_args.device
        
        # Apply training overrides
        if config.training and hasattr(cli_args, 'epochs') and cli_args.epochs is not None:
            config.training.epochs = cli_args.epochs
        
        if config.training and hasattr(cli_args, 'batch_size') and cli_args.batch_size is not None:
            config.training.batch_size = cli_args.batch_size
        
        if config.training and hasattr(cli_args, 'learning_rate') and cli_args.learning_rate is not None:
            config.training.learning_rate = cli_args.learning_rate
        
        if config.training and hasattr(cli_args, 'device') and cli_args.device is not None:
            config.training.device = cli_args.device
        
        # Apply dataset overrides
        if config.dataset and hasattr(cli_args, 'dataset_path') and cli_args.dataset_path is not None:
            config.dataset.dataset_path = Path(cli_args.dataset_path)
        
        # Apply detection engine overrides
        if hasattr(cli_args, 'device') and cli_args.device is not None:
            config.detection_engine.device = cli_args.device
        
        if hasattr(cli_args, 'conf_threshold') and cli_args.conf_threshold is not None:
            config.detection_engine.conf_threshold = cli_args.conf_threshold
        
        if hasattr(cli_args, 'iou_threshold') and cli_args.iou_threshold is not None:
            config.detection_engine.iou_threshold = cli_args.iou_threshold
        
        return config
