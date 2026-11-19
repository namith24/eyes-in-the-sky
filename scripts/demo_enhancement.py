#!/usr/bin/env python3
"""
Enhancement Demonstration Script

This script demonstrates the model enhancement module by:
1. Loading the baseline YOLO11n model
2. Computing baseline complexity metrics
3. Applying different enhancement configurations
4. Computing enhanced model complexity
5. Validating complexity constraints (≤20% params, ≤25% GFLOPs)
6. Displaying comparison results

Requirements: 3.1, 3.2, 3.3, 3.4

Usage:
    python scripts/demo_enhancement.py
    python scripts/demo_enhancement.py --model yolo11n.pt
    python scripts/demo_enhancement.py --device cuda
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ultralytics import YOLO

from uav_survivor_detection.models.enhancement import (
    EnhancementModule,
    EnhancementConfig,
    ModelComplexity
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("EnhancementDemo")


def print_section_header(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}\n")


def print_complexity_metrics(name: str, complexity: ModelComplexity) -> None:
    """Print complexity metrics in a formatted table."""
    print(f"{name}:")
    print(f"  Parameters: {complexity.parameters:,}")
    print(f"  GFLOPs:     {complexity.gflops:.2f}")


def print_comparison(
    baseline: ModelComplexity,
    enhanced: ModelComplexity,
    variant_name: str,
    constraints_met: bool
) -> None:
    """Print comparison between baseline and enhanced models."""
    # Calculate increases
    param_increase = (
        (enhanced.parameters - baseline.parameters) / baseline.parameters * 100
    )
    gflops_increase = (
        (enhanced.gflops - baseline.gflops) / baseline.gflops * 100
    )
    
    print(f"\n{variant_name}:")
    print(f"  Parameters: {baseline.parameters:,} → {enhanced.parameters:,} "
          f"({param_increase:+.2f}%)")
    print(f"  GFLOPs:     {baseline.gflops:.2f} → {enhanced.gflops:.2f} "
          f"({gflops_increase:+.2f}%)")
    
    # Constraint validation
    param_ok = param_increase <= 20.0
    gflops_ok = gflops_increase <= 25.0
    
    print(f"  Constraints:")
    print(f"    Parameter increase ≤20%: {'✓ PASS' if param_ok else '✗ FAIL'} "
          f"({param_increase:.2f}%)")
    print(f"    GFLOPs increase ≤25%:    {'✓ PASS' if gflops_ok else '✗ FAIL'} "
          f"({gflops_increase:.2f}%)")
    print(f"  Overall: {'✓ CONSTRAINTS MET' if constraints_met else '✗ CONSTRAINTS VIOLATED'}")


def create_enhancement_variants() -> Dict[str, EnhancementConfig]:
    """Create a set of enhancement configuration variants to demonstrate.
    
    Returns:
        Dictionary mapping variant names to EnhancementConfig objects
    """
    variants = {
        "GAM Only": EnhancementConfig(
            enable_gam=True,
            enable_cbam=False,
            enable_c3k2_arc=False,
            enable_cl_concat=False,
            gam_reduction=16
        ),
        "CBAM Only": EnhancementConfig(
            enable_gam=False,
            enable_cbam=True,
            enable_c3k2_arc=False,
            enable_cl_concat=False,
            cbam_reduction=16
        ),
        "C3k2-ARC Only": EnhancementConfig(
            enable_gam=False,
            enable_cbam=False,
            enable_c3k2_arc=True,
            enable_cl_concat=False
        ),
        "GAM + CBAM": EnhancementConfig(
            enable_gam=True,
            enable_cbam=True,
            enable_c3k2_arc=False,
            enable_cl_concat=False,
            gam_reduction=16,
            cbam_reduction=16
        ),
        "GAM + CBAM + C3k2-ARC": EnhancementConfig(
            enable_gam=True,
            enable_cbam=True,
            enable_c3k2_arc=True,
            enable_cl_concat=False,
            gam_reduction=16,
            cbam_reduction=16
        ),
        "Full Enhancement (All Enabled)": EnhancementConfig(
            enable_gam=True,
            enable_cbam=True,
            enable_c3k2_arc=True,
            enable_cl_concat=True,
            gam_reduction=16,
            cbam_reduction=16
        ),
    }
    
    return variants


def main():
    """Main demonstration function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Demonstrate YOLO11n model enhancement with attention mechanisms"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="Path to baseline YOLO11n model weights (default: yolo11n.pt)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Compute device (default: cpu)"
    )
    parser.add_argument(
        "--reduction",
        type=int,
        default=16,
        help="Channel reduction ratio for attention mechanisms (default: 16)"
    )
    
    args = parser.parse_args()
    
    # Print banner
    print_section_header("YOLO11n Enhancement Demonstration")
    print("This script demonstrates model enhancement with attention mechanisms")
    print("and validates complexity constraints (≤20% params, ≤25% GFLOPs).")
    print(f"\nModel: {args.model}")
    print(f"Device: {args.device}")
    
    # Step 1: Load baseline YOLO11n model
    print_section_header("Step 1: Load Baseline YOLO11n Model")
    
    try:
        logger.info(f"Loading baseline model from {args.model}")
        baseline_model = YOLO(args.model)
        logger.info("✓ Baseline model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load baseline model: {e}")
        logger.error("Please ensure YOLO11n weights are available.")
        logger.error("You can download them with: yolo detect predict model=yolo11n.pt")
        sys.exit(1)
    
    # Step 2: Compute baseline complexity
    print_section_header("Step 2: Compute Baseline Complexity")
    
    # Create enhancement module with no enhancements to measure baseline
    baseline_enhancer = EnhancementModule(EnhancementConfig())
    baseline_complexity = baseline_enhancer.compute_complexity(baseline_model)
    
    print_complexity_metrics("Baseline YOLO11n", baseline_complexity)
    
    # Step 3: Create enhancement variants
    print_section_header("Step 3: Define Enhancement Variants")
    
    variants = create_enhancement_variants()
    print(f"Created {len(variants)} enhancement variants:")
    for i, variant_name in enumerate(variants.keys(), 1):
        print(f"  {i}. {variant_name}")
    
    # Step 4: Apply enhancements and compare
    print_section_header("Step 4: Apply Enhancements and Validate Constraints")
    
    results: List[Dict] = []
    
    for variant_name, config in variants.items():
        logger.info(f"\n{'─' * 80}")
        logger.info(f"Testing variant: {variant_name}")
        logger.info(f"{'─' * 80}")
        
        try:
            # Create fresh baseline model for each variant
            model = YOLO(args.model)
            
            # Create enhancement module with this configuration
            enhancer = EnhancementModule(config)
            
            # Build enhanced model
            enhanced_model = enhancer.build_enhanced_model(model)
            
            # Compute enhanced model complexity
            enhanced_complexity = enhancer.compute_complexity(enhanced_model)
            
            # Validate constraints
            constraints_met = enhancer.validate_constraints(
                baseline_complexity,
                enhanced_complexity
            )
            
            # Store results
            results.append({
                "name": variant_name,
                "config": config,
                "complexity": enhanced_complexity,
                "constraints_met": constraints_met
            })
            
            # Print comparison
            print_comparison(
                baseline_complexity,
                enhanced_complexity,
                variant_name,
                constraints_met
            )
            
        except Exception as e:
            logger.error(f"Failed to process variant '{variant_name}': {e}")
            results.append({
                "name": variant_name,
                "config": config,
                "complexity": None,
                "constraints_met": False,
                "error": str(e)
            })
    
    # Step 5: Summary
    print_section_header("Step 5: Summary")
    
    print(f"{'Variant':<40} {'Params':<12} {'GFLOPs':<10} {'Constraints':<15}")
    print(f"{'-' * 80}")
    
    # Baseline row
    print(f"{'Baseline YOLO11n':<40} "
          f"{baseline_complexity.parameters:>11,} "
          f"{baseline_complexity.gflops:>9.2f} "
          f"{'(baseline)':<15}")
    
    # Results rows
    for result in results:
        if result["complexity"] is not None:
            complexity = result["complexity"]
            status = "✓ PASS" if result["constraints_met"] else "✗ FAIL"
            
            print(f"{result['name']:<40} "
                  f"{complexity.parameters:>11,} "
                  f"{complexity.gflops:>9.2f} "
                  f"{status:<15}")
        else:
            error_msg = result.get("error", "Unknown error")
            print(f"{result['name']:<40} {'ERROR':<12} {'ERROR':<10} {error_msg[:15]:<15}")
    
    # Step 6: Recommendations
    print_section_header("Step 6: Recommendations")
    
    # Find variants that meet constraints
    valid_variants = [r for r in results if r["constraints_met"]]
    
    if valid_variants:
        print("✓ The following variants meet complexity constraints:")
        for result in valid_variants:
            config = result["config"]
            enhancements = []
            if config.enable_gam:
                enhancements.append("GAM")
            if config.enable_cbam:
                enhancements.append("CBAM")
            if config.enable_c3k2_arc:
                enhancements.append("C3k2-ARC")
            if config.enable_cl_concat:
                enhancements.append("CL-Concat")
            
            print(f"  • {result['name']}")
            print(f"    Enhancements: {', '.join(enhancements)}")
            print(f"    Parameters: {result['complexity'].parameters:,} "
                  f"({((result['complexity'].parameters - baseline_complexity.parameters) / baseline_complexity.parameters * 100):+.2f}%)")
            print(f"    GFLOPs: {result['complexity'].gflops:.2f} "
                  f"({((result['complexity'].gflops - baseline_complexity.gflops) / baseline_complexity.gflops * 100):+.2f}%)")
            print()
    else:
        print("✗ No variants meet complexity constraints.")
        print("  Consider:")
        print("    - Increasing reduction ratios (--reduction parameter)")
        print("    - Disabling some enhancement mechanisms")
        print("    - Using fewer attention modules")
    
    # Invalid variants
    invalid_variants = [r for r in results if not r["constraints_met"] and r["complexity"] is not None]
    if invalid_variants:
        print("\n✗ The following variants violate complexity constraints:")
        for result in invalid_variants:
            param_increase = (
                (result['complexity'].parameters - baseline_complexity.parameters) 
                / baseline_complexity.parameters * 100
            )
            gflops_increase = (
                (result['complexity'].gflops - baseline_complexity.gflops) 
                / baseline_complexity.gflops * 100
            )
            
            violations = []
            if param_increase > 20.0:
                violations.append(f"params: {param_increase:.2f}% > 20%")
            if gflops_increase > 25.0:
                violations.append(f"GFLOPs: {gflops_increase:.2f}% > 25%")
            
            print(f"  • {result['name']}")
            print(f"    Violations: {', '.join(violations)}")
            print()
    
    print("\n" + "=" * 80)
    print("Enhancement demonstration complete!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
