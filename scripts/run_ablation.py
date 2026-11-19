#!/usr/bin/env python3
"""Ablation study script for UAV Survivor Detection System.

This script serves as the CLI entry point for running ablation studies to quantify
the contribution of each enhancement component. It loads configuration from YAML,
executes the AblationRunner for all configured variants, and exports consolidated
results showing the impact of each enhancement.

Usage:
    python scripts/run_ablation.py --config configs/ablation_config.yaml
    python scripts/run_ablation.py --config configs/ablation_config.yaml --output-dir results/ablation_study
    python scripts/run_ablation.py --config configs/ablation_config.yaml --device cuda

Requirements: 4.1, 4.4
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from uav_survivor_detection.config.manager import ConfigurationManager
from uav_survivor_detection.ablation.runner import AblationRunner


def parse_args():
    """Parse command-line arguments.
    
    Returns:
        argparse.Namespace with parsed CLI arguments
    """
    parser = argparse.ArgumentParser(
        description="Run ablation study to evaluate contribution of enhancement components",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run ablation study with default config
  python scripts/run_ablation.py --config configs/ablation_config.yaml
  
  # Override output directory
  python scripts/run_ablation.py --config configs/ablation_config.yaml --output-dir results/my_study
  
  # Run on GPU with custom device
  python scripts/run_ablation.py --config configs/ablation_config.yaml --device cuda
        """
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to ablation configuration YAML file"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory for ablation results"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default=None,
        help="Override compute device for training and evaluation"
    )
    
    parser.add_argument(
        "--test-subset",
        type=Path,
        default=None,
        help="Override path to test subset directory"
    )
    
    parser.add_argument(
        "--dataset-yaml",
        type=Path,
        default=None,
        help="Override path to dataset YAML file"
    )
    
    return parser.parse_args()


def main():
    """Main ablation study script entry point.
    
    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    args = parse_args()
    
    try:
        # Load configuration
        print("=" * 80)
        print("UAV SURVIVOR DETECTION - ABLATION STUDY")
        print("=" * 80)
        print(f"\nLoading configuration from {args.config}")
        
        if not args.config.exists():
            print(f"ERROR: Configuration file not found: {args.config}")
            return 1
        
        config_manager = ConfigurationManager(args.config)
        system_config = config_manager.load_config()
        
        if system_config.ablation is None:
            print("ERROR: Ablation configuration not found in config file")
            print("Expected 'ablation' section in YAML with variant definitions")
            return 1
        
        ablation_config = system_config.ablation
        
        # Apply CLI overrides
        if args.output_dir is not None:
            ablation_config.output_dir = args.output_dir
            print(f"Overriding output directory: {args.output_dir}")
        
        if args.test_subset is not None:
            ablation_config.test_subset_path = args.test_subset
            print(f"Overriding test subset path: {args.test_subset}")
        
        if args.dataset_yaml is not None:
            ablation_config.dataset_yaml = args.dataset_yaml
            print(f"Overriding dataset YAML: {args.dataset_yaml}")
        
        # Apply device override to all variants
        if args.device is not None:
            print(f"Overriding device for all variants: {args.device}")
            for variant in ablation_config.variants:
                variant.training_config.device = args.device
        
        # Validate required paths
        if not ablation_config.dataset_yaml.exists():
            print(f"ERROR: Dataset YAML not found: {ablation_config.dataset_yaml}")
            return 1
        
        if not ablation_config.test_subset_path.exists():
            print(f"ERROR: Test subset path not found: {ablation_config.test_subset_path}")
            return 1
        
        # Display configuration summary
        print("\nAblation Study Configuration:")
        print(f"  Variants:             {len(ablation_config.variants)}")
        print(f"  Dataset YAML:         {ablation_config.dataset_yaml}")
        print(f"  Test subset:          {ablation_config.test_subset_path}")
        print(f"  Output directory:     {ablation_config.output_dir}")
        
        # Display variant list
        print("\nVariants to evaluate:")
        for i, variant in enumerate(ablation_config.variants, start=1):
            enh = variant.enhancement_config
            enhancements = []
            if enh.enable_gam:
                enhancements.append("GAM")
            if enh.enable_cbam:
                enhancements.append("CBAM")
            if enh.enable_c3k2_arc:
                enhancements.append("C3k2-ARC")
            if enh.enable_cl_concat:
                enhancements.append("CL-Concat")
            
            enh_str = ", ".join(enhancements) if enhancements else "baseline"
            print(f"  {i:2d}. {variant.name:20s} [{enh_str}]")
        
        # Display training parameters (from first variant as representative)
        if ablation_config.variants:
            train_cfg = ablation_config.variants[0].training_config
            print("\nShared Training Parameters:")
            print(f"  Model:                {train_cfg.model_name}")
            print(f"  Epochs:               {train_cfg.epochs}")
            print(f"  Batch size:           {train_cfg.batch_size}")
            print(f"  Image size:           {train_cfg.img_size}")
            print(f"  Learning rate:        {train_cfg.learning_rate}")
            print(f"  Device:               {train_cfg.device}")
            print(f"  Random seed:          {train_cfg.random_seed}")
        
        print("\n" + "=" * 80)
        print("STARTING ABLATION STUDY")
        print("=" * 80)
        print()
        
        # Initialize and run ablation study
        runner = AblationRunner(ablation_config)
        results = runner.run_ablation_study()
        
        # Display final summary
        print("\n" + "=" * 80)
        print("ABLATION STUDY COMPLETED")
        print("=" * 80)
        
        successful_variants = [v for v in results.variants if v.status == "success"]
        failed_variants = [v for v in results.variants if v.status == "failed"]
        
        print(f"\nTotal variants:       {len(results.variants)}")
        print(f"Successful:           {len(successful_variants)}")
        print(f"Failed:               {len(failed_variants)}")
        
        if successful_variants:
            print("\n" + "-" * 80)
            print("Performance Summary (Successful Variants)")
            print("-" * 80)
            print(f"{'Variant':<20} {'mAP@50':>8} {'mAP@50:95':>10} {'Small AP':>9} {'FPS':>7} {'Params':>10}")
            print("-" * 80)
            
            for variant in successful_variants:
                print(
                    f"{variant.variant_name:<20} "
                    f"{variant.map50:>8.4f} "
                    f"{variant.map50_95:>10.4f} "
                    f"{variant.small_object_ap:>9.4f} "
                    f"{variant.fps_mean:>7.2f} "
                    f"{variant.parameters:>10,}"
                )
            print("-" * 80)
            
            # Highlight best performing variant
            best_variant = max(successful_variants, key=lambda v: v.map50)
            print(f"\nBest performing variant: {best_variant.variant_name} (mAP@50: {best_variant.map50:.4f})")
        
        if failed_variants:
            print("\n" + "-" * 80)
            print("Failed Variants")
            print("-" * 80)
            for variant in failed_variants:
                print(f"  ✗ {variant.variant_name}: {variant.error_message}")
            print("-" * 80)
        
        # Display output file locations
        print("\nResults exported to:")
        print(f"  CSV:                  {ablation_config.output_dir / 'ablation_results.csv'}")
        print(f"  Markdown:             {ablation_config.output_dir / 'ablation_results.md'}")
        print(f"  Log file:             {ablation_config.output_dir / 'ablation_study.log'}")
        
        print("\n" + "=" * 80)
        print()
        
        # Return success if at least one variant succeeded
        if len(successful_variants) > 0:
            return 0
        else:
            print("WARNING: All variants failed. Check log file for details.")
            return 1
        
    except KeyboardInterrupt:
        print("\n\nAblation study interrupted by user (Ctrl+C)")
        print("Partial results may be available in the output directory")
        return 130  # Standard exit code for SIGINT
        
    except Exception as e:
        print(f"\nERROR: Ablation study failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
