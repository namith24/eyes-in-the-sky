#!/usr/bin/env python3
"""Dataset preparation CLI script for UAV Survivor Detection System.

This script provides a command-line interface for preparing the C2A dataset
using the Dataset Pipeline (Requirement 1). It supports:
- Loading configuration from YAML files
- Command-line overrides for key parameters
- Verification, splitting, and YOLO format export
- Manifest generation with dataset statistics

**Validates: Requirements 1.1, 1.4**

Usage:
    python scripts/prepare_dataset.py --config configs/training_config.yaml
    python scripts/prepare_dataset.py --config configs/training_config.yaml --overwrite
    python scripts/prepare_dataset.py --config configs/training_config.yaml --dataset-path data/custom
"""

import argparse
import sys
from pathlib import Path
from typing import NoReturn

# Add src directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from uav_survivor_detection.config.manager import ConfigurationManager, ConfigurationError
from uav_survivor_detection.dataset.pipeline import DatasetPipeline, ValidationError


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Prepare C2A dataset for UAV survivor detection training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with config file
  python scripts/prepare_dataset.py --config configs/training_config.yaml
  
  # Override dataset path and enable overwrite
  python scripts/prepare_dataset.py --config configs/training_config.yaml \\
      --dataset-path data/custom --overwrite
  
  # Skip annotation verification (faster, use with caution)
  python scripts/prepare_dataset.py --config configs/training_config.yaml \\
      --skip-verification
  
  # Custom split ratios
  python scripts/prepare_dataset.py --config configs/training_config.yaml \\
      --train-ratio 0.8 --val-ratio 0.15 --test-ratio 0.05
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )
    
    # Dataset configuration overrides
    parser.add_argument(
        "--dataset-path",
        type=str,
        help="Override dataset path from config (path to raw C2A dataset)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Override output directory from config (where processed dataset will be saved)"
    )
    
    parser.add_argument(
        "--train-ratio",
        type=float,
        help="Override training split ratio (0.0-1.0)"
    )
    
    parser.add_argument(
        "--val-ratio",
        type=float,
        help="Override validation split ratio (0.0-1.0)"
    )
    
    parser.add_argument(
        "--test-ratio",
        type=float,
        help="Override test split ratio (0.0-1.0)"
    )
    
    parser.add_argument(
        "--random-seed",
        type=int,
        help="Override random seed for reproducible splits"
    )
    
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output directory without prompting"
    )
    
    # Pipeline options
    parser.add_argument(
        "--skip-verification",
        action="store_true",
        help="Skip annotation verification step (faster but not recommended)"
    )
    
    return parser.parse_args()


def apply_cli_overrides(config, args: argparse.Namespace) -> None:
    """Apply command-line argument overrides to dataset configuration.
    
    Args:
        config: SystemConfig object to modify
        args: Parsed command-line arguments
    """
    if not config.dataset:
        print("ERROR: Configuration file must contain 'dataset' section", file=sys.stderr)
        sys.exit(1)
    
    # Apply overrides
    if args.dataset_path is not None:
        config.dataset.dataset_path = Path(args.dataset_path)
    
    if args.output_dir is not None:
        config.dataset.output_dir = Path(args.output_dir)
    
    if args.train_ratio is not None:
        config.dataset.train_ratio = args.train_ratio
    
    if args.val_ratio is not None:
        config.dataset.val_ratio = args.val_ratio
    
    if args.test_ratio is not None:
        config.dataset.test_ratio = args.test_ratio
    
    if args.random_seed is not None:
        config.dataset.random_seed = args.random_seed
    
    if args.overwrite:
        config.dataset.overwrite_existing = True


def print_config_summary(config) -> None:
    """Print dataset configuration summary.
    
    Args:
        config: DatasetConfig object
    """
    print("\n" + "=" * 70)
    print("Dataset Preparation Configuration")
    print("=" * 70)
    print(f"Dataset Path:      {config.dataset_path}")
    print(f"Output Directory:  {config.output_dir}")
    print(f"Train Ratio:       {config.train_ratio:.2f}")
    print(f"Validation Ratio:  {config.val_ratio:.2f}")
    print(f"Test Ratio:        {config.test_ratio:.2f}")
    print(f"Random Seed:       {config.random_seed}")
    print(f"Overwrite:         {'Yes' if config.overwrite_existing else 'No'}")
    print("=" * 70 + "\n")


def print_manifest_summary(manifest) -> None:
    """Print dataset manifest summary.
    
    Args:
        manifest: DatasetManifest object
    """
    print("\n" + "=" * 70)
    print("Dataset Preparation Complete!")
    print("=" * 70)
    print(f"Total Images:      {manifest.total_images:,}")
    print(f"Training Set:      {manifest.train_count:,} ({manifest.train_count/manifest.total_images*100:.1f}%)")
    print(f"Validation Set:    {manifest.val_count:,} ({manifest.val_count/manifest.total_images*100:.1f}%)")
    print(f"Test Set:          {manifest.test_count:,} ({manifest.test_count/manifest.total_images*100:.1f}%)")
    print()
    print("Class Distribution:")
    for class_name, count in manifest.class_distribution.items():
        print(f"  {class_name}: {count:,}")
    print()
    print(f"Manifest saved to: manifest.json")
    print(f"YOLO data.yaml:    data.yaml")
    print("=" * 70 + "\n")


def main() -> NoReturn:
    """Main entry point for dataset preparation script.
    
    Exit codes:
        0: Success
        1: Configuration error (missing file, validation failed)
        2: Dataset verification failed
        3: Dataset split failed
        4: Export failed
        5: Unexpected error
    """
    # Parse command-line arguments
    args = parse_arguments()
    
    print("UAV Survivor Detection - Dataset Preparation")
    print("=" * 70)
    
    # Load configuration
    try:
        print(f"Loading configuration from: {args.config}")
        config_manager = ConfigurationManager(Path(args.config))
        config = config_manager.load_config()
        
        # Check that dataset config exists
        if not config.dataset:
            print("\nERROR: Configuration file must contain 'dataset' section", file=sys.stderr)
            print("Example configuration:", file=sys.stderr)
            print("  dataset:", file=sys.stderr)
            print("    dataset_path: 'data/C2A'", file=sys.stderr)
            print("    output_dir: 'data/C2A_processed'", file=sys.stderr)
            sys.exit(1)
        
        # Apply CLI overrides
        apply_cli_overrides(config, args)
        
        # Print configuration summary
        print_config_summary(config.dataset)
        
    except ConfigurationError as e:
        print(f"\nERROR: Configuration failed: {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: Unexpected error loading configuration: {e}", file=sys.stderr)
        sys.exit(5)
    
    # Initialize dataset pipeline
    try:
        print("Initializing dataset pipeline...")
        pipeline = DatasetPipeline(config=config.dataset)
        
    except ValueError as e:
        print(f"\nERROR: Pipeline initialization failed: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: Unexpected error initializing pipeline: {e}", file=sys.stderr)
        sys.exit(5)
    
    # Execute pipeline
    try:
        print("\nExecuting dataset preparation pipeline...")
        print("-" * 70)
        
        # Step 1: Verify annotations (unless skipped)
        if not args.skip_verification:
            print("\n[1/4] Verifying annotations...")
            errors = pipeline.verify_annotations()
            
            if errors:
                print(f"\n✗ Verification failed with {len(errors)} error(s):", file=sys.stderr)
                print("-" * 70, file=sys.stderr)
                
                # Show first 20 errors
                for error in errors[:20]:
                    print(f"  • {error}", file=sys.stderr)
                
                if len(errors) > 20:
                    print(f"\n  ... and {len(errors) - 20} more error(s)", file=sys.stderr)
                
                print("-" * 70, file=sys.stderr)
                print(f"\nDataset path: {config.dataset.dataset_path}", file=sys.stderr)
                print("\nPlease fix annotation errors before proceeding.", file=sys.stderr)
                print("Or use --skip-verification to bypass this check (not recommended).", file=sys.stderr)
                sys.exit(2)
            
            print(f"✓ All annotations verified successfully")
        else:
            print("\n[1/4] Skipping annotation verification (--skip-verification)")
        
        # Step 2: Split dataset
        print("\n[2/4] Splitting dataset...")
        split = pipeline.split_dataset()
        print(f"✓ Dataset split complete:")
        print(f"    Train:      {len(split.train_files):,} images")
        print(f"    Validation: {len(split.val_files):,} images")
        print(f"    Test:       {len(split.test_files):,} images")
        
        # Step 3: Export to YOLO format
        print(f"\n[3/4] Exporting to YOLO format...")
        print(f"    Output: {config.dataset.output_dir}")
        pipeline.export_yolo_format()
        print(f"✓ YOLO format export complete")
        
        # Step 4: Generate manifest
        print(f"\n[4/4] Generating dataset manifest...")
        manifest = pipeline.generate_manifest()
        manifest_path = config.dataset.output_dir / 'manifest.json'
        manifest.save_json(manifest_path)
        print(f"✓ Manifest saved to: {manifest_path}")
        
        # Print final summary
        print_manifest_summary(manifest)
        
        print("✓ Dataset preparation completed successfully!")
        sys.exit(0)
        
    except ValueError as e:
        print(f"\n✗ ERROR: {e}", file=sys.stderr)
        sys.exit(3)
    except FileExistsError as e:
        print(f"\n✗ ERROR: {e}", file=sys.stderr)
        print("\nUse --overwrite flag to overwrite existing output directory.", file=sys.stderr)
        sys.exit(4)
    except Exception as e:
        print(f"\n✗ ERROR: Unexpected error during pipeline execution: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(5)


if __name__ == "__main__":
    main()
