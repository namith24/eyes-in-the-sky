#!/usr/bin/env python3
"""Training script for UAV Survivor Detection System.

This script serves as the CLI entry point for training YOLO models on the C2A dataset.
It loads configuration from YAML, executes the training pipeline, and exports results.

Usage:
    python scripts/train.py --config configs/training_config.yaml
    python scripts/train.py --config configs/training_config.yaml --resume --checkpoint runs/train/weights/last.pt

Requirements: 2.1, 2.5
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from uav_survivor_detection.config.manager import ConfigurationManager
from uav_survivor_detection.training.pipeline import TrainingPipeline
from uav_survivor_detection.models.enhancement import EnhancementModule

from ultralytics import YOLO


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train YOLO model on C2A dataset"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to training configuration YAML file"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from checkpoint"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to checkpoint for resuming (required if --resume is set)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default=None,
        help="Override compute device"
    )
    return parser.parse_args()


def main():
    """Main training script entry point."""
    args = parse_args()
    
    # Load configuration
    print(f"Loading configuration from {args.config}")
    config_manager = ConfigurationManager(args.config)
    system_config = config_manager.load_config()
    
    if system_config.training is None:
        print("ERROR: Training configuration not found in config file")
        sys.exit(1)
    
    training_config = system_config.training
    
    # Apply CLI overrides
    if args.epochs is not None:
        training_config.epochs = args.epochs
        print(f"Overriding epochs: {args.epochs}")
    
    if args.batch_size is not None:
        training_config.batch_size = args.batch_size
        print(f"Overriding batch size: {args.batch_size}")
    
    if args.device is not None:
        training_config.device = args.device
        print(f"Overriding device: {args.device}")
    
    # Validate resume arguments
    if args.resume and args.checkpoint is None:
        print("ERROR: --checkpoint is required when --resume is set")
        sys.exit(1)
    
    # Initialize training pipeline
    print("\nInitializing training pipeline...")
    pipeline = TrainingPipeline(training_config)

    # Apply enhancement module if configured (Requirement 3.5)
    if system_config.enhancement is not None and any([
        system_config.enhancement.enable_gam,
        system_config.enhancement.enable_cbam,
        system_config.enhancement.enable_c3k2_arc,
        system_config.enhancement.enable_cl_concat
    ]):
        print("\nApplying enhancement module:")
        print(f"  GAM       : {system_config.enhancement.enable_gam}")
        print(f"  CBAM      : {system_config.enhancement.enable_cbam}")
        print(f"  C3k2-ARC  : {system_config.enhancement.enable_c3k2_arc}")
        print(f"  CL-Concat : {system_config.enhancement.enable_cl_concat}")
        base_model = YOLO(f"{training_config.model_name}.pt")
        enhancement_module = EnhancementModule(system_config.enhancement)
        pipeline.model = enhancement_module.build_enhanced_model(base_model)
        complexity = enhancement_module.compute_complexity(pipeline.model)
        print(f"Enhanced model: {complexity.parameters:,} params, {complexity.gflops:.2f} GFLOPs")
    else:
        print("\nTraining baseline model (no enhancements)")
    
    # Execute training
    print("\nStarting training...")
    print(f"  Model: {training_config.model_name}")
    print(f"  Dataset: {training_config.dataset_yaml}")
    print(f"  Epochs: {training_config.epochs}")
    print(f"  Batch size: {training_config.batch_size}")
    print(f"  Image size: {training_config.img_size}")
    print(f"  Learning rate: {training_config.learning_rate}")
    print(f"  Device: {training_config.device}")
    print(f"  Random seed: {training_config.random_seed}")
    
    if args.resume:
        print(f"  Resuming from: {args.checkpoint}")
    
    print()
    
    try:
        result = pipeline.train(resume=args.resume, checkpoint_path=args.checkpoint)
        
        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Best checkpoint: {result.best_checkpoint}")
        print(f"Final checkpoint: {result.final_checkpoint}")
        print(f"Best mAP@50: {result.best_map50:.4f}")
        print(f"Total duration: {result.total_duration_seconds:.2f} seconds")
        print(f"Epochs trained: {len(result.metrics_history)}")
        print()
        
        # Export training result summary
        summary_file = training_config.output_dir / "train" / "training_summary.json"
        summary_data = {
            "best_checkpoint": str(result.best_checkpoint),
            "final_checkpoint": str(result.final_checkpoint),
            "best_map50": result.best_map50,
            "total_duration_seconds": result.total_duration_seconds,
            "epochs_trained": len(result.metrics_history),
            "config": {
                "model_name": training_config.model_name,
                "epochs": training_config.epochs,
                "batch_size": training_config.batch_size,
                "img_size": training_config.img_size,
                "learning_rate": training_config.learning_rate,
                "random_seed": training_config.random_seed,
                "device": training_config.device,
            }
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"Training summary exported to {summary_file}")
        print()
        
        return 0
        
    except Exception as e:
        print(f"\nERROR: Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
