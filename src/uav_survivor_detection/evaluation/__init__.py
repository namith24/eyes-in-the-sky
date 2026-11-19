"""Evaluation module for UAV Survivor Detection System."""

from .evaluator import (
    Evaluator,
    DetectionMetrics,
    EfficiencyMetrics,
    ModelComplexity,
    EvaluationResult,
)
from .visualization import (
    Visualizer,
    generate_pr_curves,
    generate_confusion_matrix,
    export_report,
    generate_pr_curves_with_model,
    generate_confusion_matrix_with_model,
)

__all__ = [
    "Evaluator",
    "DetectionMetrics",
    "EfficiencyMetrics",
    "ModelComplexity",
    "EvaluationResult",
    "Visualizer",
    "generate_pr_curves",
    "generate_confusion_matrix",
    "export_report",
    "generate_pr_curves_with_model",
    "generate_confusion_matrix_with_model",
]
