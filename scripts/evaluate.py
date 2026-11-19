#!/usr/bin/env python3
"""Evaluation script for UAV Survivor Detection System.

This script serves as the CLI entry point for evaluating trained YOLO models on the test dataset.
It loads configuration from YAML, executes the evaluator, generates visualizations (PR curves,
confusion matrix), and exports comprehensive evaluation reports in JSON and Markdown formats.

Usage:
    python scripts/evaluate.py --config configs/evaluation_config.yaml
    python scripts/evaluate.py --config configs/evaluation_config.yaml --checkpoint checkpoints/best.pt
    python scripts/evaluate.py --config configs/evaluation_config.yaml --test-dataset data/processed/test

Requirements: 5.4, 5.5
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from uav_survivor_detection.config.manager import ConfigurationManager
from uav_survivor_detection.evaluation.evaluator import Evaluator
from uav_survivor_detection.evaluation.visualization import (
    generate_pr_curves,
    generate_confusion_matrix,
    export_report,
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO model on test dataset"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to evaluation configuration YAML file"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Override checkpoint path from config"
    )
    parser.add_argument(
        "--test-dataset",
        type=Path,
        default=None,
        help="Override test dataset path from config (YAML or directory)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory for reports and visualizations (default: output/evaluation)"
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=None,
        help="Override confidence threshold [0.0-1.0]"
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=None,
        help="Override IoU threshold [0.0-1.0]"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default=None,
        help="Override compute device"
    )
    parser.add_argument(
        "--no-visualizations",
        action="store_true",
        help="Skip generating visualizations (PR curves and confusion matrix)"
    )
    return parser.parse_args()


def main():
    """Main evaluation script entry point."""
    args = parse_args()
    
    # Load configuration
    print(f"Loading configuration from {args.config}")
    config_manager = ConfigurationManager(args.config)
    system_config = config_manager.load_config()
    
    if system_config.evaluation is None:
        print("ERROR: Evaluation configuration not found in config file")
        sys.exit(1)
    
    evaluation_config = system_config.evaluation
    
    # Apply CLI overrides
    if args.checkpoint is not None:
        evaluation_config.checkpoint_path = args.checkpoint
        print(f"Overriding checkpoint path: {args.checkpoint}")
    
    if args.test_dataset is not None:
        evaluation_config.test_dataset_path = args.test_dataset
        print(f"Overriding test dataset path: {args.test_dataset}")
    
    if args.conf_threshold is not None:
        evaluation_config.conf_threshold = args.conf_threshold
        print(f"Overriding confidence threshold: {args.conf_threshold}")
    
    if args.iou_threshold is not None:
        evaluation_config.iou_threshold = args.iou_threshold
        print(f"Overriding IoU threshold: {args.iou_threshold}")
    
    if args.device is not None:
        evaluation_config.device = args.device
        print(f"Overriding device: {args.device}")
    
    # Set output directory
    output_dir = args.output_dir if args.output_dir else Path("output/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate required parameters
    if evaluation_config.checkpoint_path is None:
        print("ERROR: checkpoint_path is required (set in config or via --checkpoint)")
        sys.exit(1)
    
    if evaluation_config.test_dataset_path is None:
        print("ERROR: test_dataset_path is required (set in config or via --test-dataset)")
        sys.exit(1)
    
    if not evaluation_config.checkpoint_path.exists():
        print(f"ERROR: Checkpoint file not found: {evaluation_config.checkpoint_path}")
        sys.exit(1)
    
    if not evaluation_config.test_dataset_path.exists():
        print(f"ERROR: Test dataset path not found: {evaluation_config.test_dataset_path}")
        sys.exit(1)
    
    # Initialize evaluator
    print("\nInitializing evaluator...")
    evaluator = Evaluator(evaluation_config)
    
    # Display evaluation configuration
    print("\nEvaluation Configuration:")
    print(f"  Checkpoint: {evaluation_config.checkpoint_path}")
    print(f"  Test dataset: {evaluation_config.test_dataset_path}")
    print(f"  Confidence threshold: {evaluation_config.conf_threshold}")
    print(f"  IoU threshold: {evaluation_config.iou_threshold}")
    print(f"  Image size: {evaluation_config.img_size}")
    print(f"  Device: {evaluation_config.device}")
    print(f"  Output directory: {output_dir}")
    print()
    
    # Execute evaluation
    try:
        print("Starting evaluation...")
        print("-" * 60)
        
        result = evaluator.evaluate()
        
        print("\n" + "=" * 60)
        print("EVALUATION COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
        # Display detection metrics
        print("\nDetection Performance:")
        print(f"  Precision:            {result.detection_metrics.precision:.4f}")
        print(f"  Recall:               {result.detection_metrics.recall:.4f}")
        print(f"  mAP@50:               {result.detection_metrics.map50:.4f}")
        print(f"  mAP@50:95:            {result.detection_metrics.map50_95:.4f}")
        print(f"  Small-Object AP:      {result.detection_metrics.small_object_ap:.4f}")
        
        if result.detection_metrics.per_class_ap:
            print("\n  Per-Class AP:")
            for class_name, ap_value in sorted(result.detection_metrics.per_class_ap.items()):
                print(f"    {class_name:20s}: {ap_value:.4f}")
        
        # Display efficiency metrics
        print("\nEfficiency Metrics:")
        print(f"  FPS (mean):           {result.efficiency_metrics.fps_mean:.2f}")
        print(f"  FPS (median):         {result.efficiency_metrics.fps_median:.2f}")
        print(f"  Latency mean:         {result.efficiency_metrics.latency_ms_mean:.2f} ms")
        print(f"  Latency median:       {result.efficiency_metrics.latency_ms_median:.2f} ms")
        print(f"  Latency p95:          {result.efficiency_metrics.latency_ms_p95:.2f} ms")
        
        # Display complexity metrics
        print("\nModel Complexity:")
        print(f"  Parameters:           {result.complexity_metrics.parameters:,}")
        print(f"  GFLOPs:               {result.complexity_metrics.gflops:.4f}")
        
        # Display summary stats
        print("\nEvaluation Summary:")
        print(f"  Total images:         {result.total_images}")
        print(f"  Total detections:     {result.total_detections}")
        print(f"  Duration:             {result.evaluation_duration_seconds:.2f} seconds")
        print(f"  Timestamp:            {result.timestamp}")
        print()
        
        # Generate visualizations
        if not args.no_visualizations:
            print("\nGenerating visualizations...")
            
            try:
                pr_curves_path = output_dir / "pr_curves.png"
                generate_pr_curves(result, pr_curves_path)
                print(f"  ✓ PR curves saved to {pr_curves_path}")
            except Exception as e:
                print(f"  ✗ Failed to generate PR curves: {e}")
            
            try:
                confusion_matrix_path = output_dir / "confusion_matrix.png"
                generate_confusion_matrix(result, confusion_matrix_path)
                print(f"  ✓ Confusion matrix saved to {confusion_matrix_path}")
            except Exception as e:
                print(f"  ✗ Failed to generate confusion matrix: {e}")
            
            print()
        
        # Export comprehensive report
        print("Exporting evaluation reports...")
        report_paths = export_report(result, output_dir)
        
        print(f"  ✓ JSON report saved to {report_paths['json']}")
        print(f"  ✓ Markdown report saved to {report_paths['markdown']}")
        print()
        
        print("=" * 60)
        print("All evaluation outputs saved to:", output_dir)
        print("=" * 60)
        print()
        
        return 0
        
    except Exception as e:
        print(f"\nERROR: Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
