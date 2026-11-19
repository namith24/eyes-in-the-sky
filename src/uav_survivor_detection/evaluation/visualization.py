"""Visualization and reporting module for evaluation results.

This module provides visualization capabilities for model evaluation, including
precision-recall curves, confusion matrix heatmaps, and structured report generation
in both JSON and Markdown formats.

Two usage patterns are supported:

1. **EvaluationResult-based** (preferred): Pass a completed
   :class:`~uav_survivor_detection.evaluation.EvaluationResult` to the module-level
   functions :func:`generate_pr_curves`, :func:`generate_confusion_matrix`, and
   :func:`export_report`.  No live model or dataset is required.

2. **Live-model-based**: Use :class:`Visualizer` methods directly, passing a loaded
   ``ultralytics.YOLO`` model and dataset path so that ``model.val()`` is re-executed
   to extract raw curve data.

**Validates: Requirements 5.4, 5.5**
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class Visualizer:
    """Generates visualizations and reports from model evaluation results.

    Produces precision-recall curve plots, confusion matrix heatmaps, and
    structured reports (JSON + Markdown) from EvaluationResult objects or
    directly from ultralytics validation output.

    **Validates: Requirements 5.4, 5.5**

    Example::

        from uav_survivor_detection.evaluation import Visualizer, EvaluationResult

        viz = Visualizer()
        viz.generate_pr_curves(model, "data.yaml", "outputs/pr_curves.png")
        viz.generate_confusion_matrix(model, "data.yaml", "outputs/cm.png")
        paths = viz.export_report(result, Path("outputs/"))
    """

    def __init__(self, dpi: int = 150, figsize: Tuple[int, int] = (10, 8)) -> None:
        """Initialize Visualizer with plot settings.

        Args:
            dpi: Resolution for saved PNG figures.
            figsize: Default figure size as (width, height) in inches.
        """
        self.dpi = dpi
        self.figsize = figsize

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_pr_curves(
        self,
        model,
        dataset_path: str,
        output_path: str,
        conf: float = 0.001,
        iou: float = 0.5,
        imgsz: int = 640,
        device: str = "cpu",
    ) -> Path:
        """Generate per-class precision-recall curves and save as PNG.

        Runs ultralytics ``model.val()`` to obtain validation results, extracts
        per-class precision/recall arrays, and plots one curve per class using
        matplotlib.  Falls back to a placeholder plot when curve data is
        unavailable (e.g. no ground-truth boxes in the dataset).

        Args:
            model: A loaded ``ultralytics.YOLO`` model instance.
            dataset_path: Path to the dataset YAML file or directory accepted
                by ``model.val()``.
            output_path: Destination path for the output PNG (parent directories
                are created automatically).
            conf: Confidence threshold passed to ``model.val()``.
            iou: IoU threshold passed to ``model.val()``.
            imgsz: Inference image size passed to ``model.val()``.
            device: Compute device string, e.g. ``"cpu"`` or ``"cuda:0"``.

        Returns:
            Resolved :class:`pathlib.Path` pointing to the saved PNG.

        **Validates: Requirement 5.5**
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Run validation to obtain metrics object
        val_results = self._run_val(model, dataset_path, conf, iou, imgsz, device)

        fig, ax = plt.subplots(figsize=self.figsize)

        plotted = False
        if val_results is not None:
            plotted = self._plot_pr_curves_from_results(ax, val_results, model)

        if not plotted:
            self._plot_pr_placeholder(ax)

        ax.set_xlabel("Recall", fontsize=13)
        ax.set_ylabel("Precision", fontsize=13)
        ax.set_title("Precision-Recall Curves", fontsize=15)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.legend(loc="lower left", fontsize=9)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(output_path, dpi=self.dpi)
        plt.close(fig)

        logger.info("PR curves saved to %s", output_path)
        return output_path.resolve()

    def generate_confusion_matrix(
        self,
        model,
        dataset_path: str,
        output_path: str,
        conf: float = 0.25,
        iou: float = 0.5,
        imgsz: int = 640,
        device: str = "cpu",
        normalize: bool = True,
    ) -> Path:
        """Generate a confusion matrix heatmap and save as PNG.

        Runs ultralytics ``model.val()`` to obtain the confusion matrix, then
        renders it as a colour-mapped heatmap using pure matplotlib (no seaborn
        dependency).  Cell values are annotated as integers (raw counts) or
        two-decimal floats (normalised).  Falls back to a placeholder when the
        confusion matrix is unavailable.

        Args:
            model: A loaded ``ultralytics.YOLO`` model instance.
            dataset_path: Path to the dataset YAML file or directory.
            output_path: Destination path for the output PNG.
            conf: Confidence threshold passed to ``model.val()``.
            iou: IoU threshold passed to ``model.val()``.
            imgsz: Inference image size passed to ``model.val()``.
            device: Compute device string.
            normalize: If ``True``, normalise each row of the confusion matrix
                so values represent fractions (0–1).

        Returns:
            Resolved :class:`pathlib.Path` pointing to the saved PNG.

        **Validates: Requirement 5.5**
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        val_results = self._run_val(model, dataset_path, conf, iou, imgsz, device)

        cm_array, class_names = self._extract_confusion_matrix(val_results, model)

        fig, ax = plt.subplots(figsize=self.figsize)

        if cm_array is not None:
            if normalize:
                row_sums = cm_array.sum(axis=1, keepdims=True)
                # Avoid division by zero
                row_sums = np.where(row_sums == 0, 1, row_sums)
                cm_display = cm_array.astype(float) / row_sums
                fmt = ".2f"
                cbar_label = "Fraction"
            else:
                cm_display = cm_array.astype(float)
                fmt = "d"
                cbar_label = "Count"

            norm = Normalize(vmin=0.0, vmax=cm_display.max() if cm_display.max() > 0 else 1.0)
            im = ax.imshow(cm_display, interpolation="nearest", cmap="Blues", norm=norm)

            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label(cbar_label, fontsize=11)

            tick_marks = np.arange(len(class_names))
            ax.set_xticks(tick_marks)
            ax.set_yticks(tick_marks)
            ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
            ax.set_yticklabels(class_names, fontsize=9)

            # Annotate cells
            thresh = cm_display.max() / 2.0
            for i in range(cm_display.shape[0]):
                for j in range(cm_display.shape[1]):
                    val = cm_display[i, j]
                    text = f"{int(cm_array[i, j])}" if fmt == "d" else f"{val:.2f}"
                    color = "white" if val > thresh else "black"
                    ax.text(j, i, text, ha="center", va="center",
                            color=color, fontsize=8)
        else:
            self._plot_cm_placeholder(ax)

        ax.set_xlabel("Predicted Label", fontsize=13)
        ax.set_ylabel("True Label", fontsize=13)
        title = "Confusion Matrix (Normalised)" if normalize else "Confusion Matrix"
        ax.set_title(title, fontsize=15)

        fig.tight_layout()
        fig.savefig(output_path, dpi=self.dpi)
        plt.close(fig)

        logger.info("Confusion matrix saved to %s", output_path)
        return output_path.resolve()

    def export_report(
        self,
        result,
        output_dir: Path,
    ) -> Dict[str, Path]:
        """Export evaluation results as JSON and Markdown reports.

        Writes two files into *output_dir*:

        * ``evaluation_report.json`` — full metrics serialised with
          :class:`~uav_survivor_detection.evaluation.Evaluator`-compatible
          ``to_dict`` logic.
        * ``evaluation_report.md`` — human-readable Markdown summary with
          sections for Detection Performance, Efficiency Metrics, and Model
          Complexity.

        Args:
            result: An :class:`~uav_survivor_detection.evaluation.EvaluationResult`
                instance produced by the :class:`~uav_survivor_detection.evaluation.Evaluator`.
            output_dir: Directory in which to write the report files.  Created
                if it does not exist.

        Returns:
            Dictionary with keys ``"json"`` and ``"markdown"`` mapping to the
            absolute :class:`pathlib.Path` of each written file.

        **Validates: Requirement 5.4**
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # ---- JSON report ------------------------------------------------
        result_dict = self._result_to_dict(result)
        json_path = output_dir / "evaluation_report.json"
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(result_dict, fh, indent=2)
        logger.info("JSON report saved to %s", json_path)

        # ---- Markdown report --------------------------------------------
        md_path = output_dir / "evaluation_report.md"
        md_content = self._build_markdown_report(result)
        with open(md_path, "w", encoding="utf-8") as fh:
            fh.write(md_content)
        logger.info("Markdown report saved to %s", md_path)

        return {"json": json_path.resolve(), "markdown": md_path.resolve()}

    # ------------------------------------------------------------------
    # EvaluationResult-based API (no live model required)
    # ------------------------------------------------------------------

    def generate_pr_curves_from_result(
        self,
        result,
        output_path: Path,
    ) -> Path:
        """Generate per-class precision-recall curves from a completed EvaluationResult.

        Plots one bar/point per class using the ``per_class_ap`` mapping stored in
        ``result.detection_metrics``.  When ``per_class_ap`` is empty a placeholder
        message is rendered instead.  The figure is saved as a PNG at *output_path*
        and the matplotlib figure is closed to avoid memory leaks.

        Args:
            result: A completed
                :class:`~uav_survivor_detection.evaluation.EvaluationResult` instance
                produced by the :class:`~uav_survivor_detection.evaluation.Evaluator`.
            output_path: Destination path for the saved PNG.  Parent directories are
                created automatically.

        Returns:
            Resolved :class:`pathlib.Path` pointing to the saved PNG.

        **Validates: Requirement 5.5**
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=self.figsize)

        per_class_ap: Dict[str, float] = result.detection_metrics.per_class_ap or {}

        if per_class_ap:
            cmap = _get_class_cmap(len(per_class_ap))
            classes = sorted(per_class_ap.keys())
            for idx, cls_name in enumerate(classes):
                ap_val = per_class_ap[cls_name]
                colour = cmap(idx / max(len(classes) - 1, 1))
                # Draw a horizontal line from recall=0 to recall=1 at height=AP
                # to represent the single-number AP summary as a degenerate PR curve.
                ax.plot([0.0, 1.0], [ap_val, ap_val],
                        color=colour, linewidth=2,
                        label=f"{cls_name} (AP={ap_val:.3f})")
                ax.scatter([0.5], [ap_val], color=colour, zorder=5, s=40)

            # Also draw an overall mAP@50 reference line
            map50 = result.detection_metrics.map50
            ax.axhline(y=map50, color="black", linewidth=1.5, linestyle="--",
                       label=f"mAP@50={map50:.3f}")
        else:
            self._plot_pr_placeholder(ax)

        ax.set_xlabel("Recall", fontsize=13)
        ax.set_ylabel("Precision / AP", fontsize=13)
        ax.set_title("Per-Class Average Precision (PR Curve Summary)", fontsize=15)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.legend(loc="lower left", fontsize=9)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(output_path, dpi=self.dpi)
        plt.close(fig)

        logger.info("PR curves (from EvaluationResult) saved to %s", output_path)
        return output_path.resolve()

    def generate_confusion_matrix_from_result(
        self,
        result,
        output_path: Path,
    ) -> Path:
        """Generate a confusion matrix placeholder from a completed EvaluationResult.

        A full per-pixel confusion matrix requires raw prediction arrays that are not
        stored in :class:`~uav_survivor_detection.evaluation.EvaluationResult`.  This
        method therefore renders an informative placeholder that displays the evaluation
        summary metrics and documents why raw confusion matrix data is unavailable.

        To obtain a full confusion matrix, use
        :meth:`generate_confusion_matrix` with a live model and dataset path so that
        ``model.val()`` can supply the raw prediction data.

        Args:
            result: A completed
                :class:`~uav_survivor_detection.evaluation.EvaluationResult` instance.
            output_path: Destination path for the saved PNG.  Parent directories are
                created automatically.

        Returns:
            Resolved :class:`pathlib.Path` pointing to the saved PNG.

        **Validates: Requirement 5.5**
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=self.figsize)

        dm = result.detection_metrics

        # Render a summary table as a styled text box inside the axes
        summary_lines = [
            "Confusion Matrix",
            "(Full matrix requires live model.val() — see Visualizer.generate_confusion_matrix)",
            "",
            f"  Precision : {dm.precision:.4f}",
            f"  Recall    : {dm.recall:.4f}",
            f"  mAP@50    : {dm.map50:.4f}",
            f"  mAP@50:95 : {dm.map50_95:.4f}",
            f"  Small-Obj AP : {dm.small_object_ap:.4f}",
        ]

        if dm.per_class_ap:
            summary_lines.append("")
            summary_lines.append("  Per-Class AP:")
            for cls_name, ap_val in sorted(dm.per_class_ap.items()):
                summary_lines.append(f"    {cls_name}: {ap_val:.4f}")

        text = "\n".join(summary_lines)
        ax.text(
            0.5, 0.5, text,
            transform=ax.transAxes,
            ha="center", va="center",
            fontsize=10,
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#f0f4ff", edgecolor="#7090c0"),
        )
        ax.axis("off")
        ax.set_title("Confusion Matrix (Summary)", fontsize=15, pad=12)

        fig.tight_layout()
        fig.savefig(output_path, dpi=self.dpi)
        plt.close(fig)

        logger.info("Confusion matrix placeholder saved to %s", output_path)
        return output_path.resolve()

    # ------------------------------------------------------------------
    # Private helpers — validation runner
    # ------------------------------------------------------------------

    def _run_val(self, model, dataset_path: str, conf: float, iou: float,
                 imgsz: int, device: str):
        """Run ultralytics model.val() and return the results object.

        Returns ``None`` on any error so callers can fall back gracefully.
        """
        try:
            results = model.val(
                data=str(dataset_path),
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                device=device,
                verbose=False,
            )
            return results
        except Exception as exc:
            logger.warning("model.val() failed: %s — falling back to placeholder plots.", exc)
            return None

    # ------------------------------------------------------------------
    # Private helpers — PR curves
    # ------------------------------------------------------------------

    def _plot_pr_curves_from_results(self, ax, val_results, model) -> bool:
        """Extract PR curve data from val_results and draw onto *ax*.

        Returns ``True`` if at least one curve was successfully plotted.
        """
        try:
            box = val_results.box
            # ultralytics stores per-class PR data in box.p (precision array),
            # box.r (recall array), and box.ap_class_index.
            # Shapes: p -> (num_classes, num_thresholds), r -> same
            # ap_class_index -> (num_classes,)
            precision = np.array(box.p) if hasattr(box, "p") else None
            recall = np.array(box.r) if hasattr(box, "r") else None
            ap_class_index = (
                np.array(box.ap_class_index) if hasattr(box, "ap_class_index") else None
            )

            if precision is None or recall is None:
                return False

            class_names = model.names if hasattr(model, "names") else {}
            num_classes = precision.shape[0] if precision.ndim == 2 else 1

            # Colour cycle for multiple classes
            cmap = _get_class_cmap(num_classes)

            if precision.ndim == 1:
                # Single-class or aggregated curve
                ap_val = float(np.trapz(precision, recall)) if len(precision) > 1 else 0.0
                label = f"class_0 (AP={ap_val:.3f})"
                ax.plot(recall, precision, color=cmap(0), linewidth=2, label=label)
                return True

            plotted = False
            for cls_idx in range(num_classes):
                r = recall[cls_idx]
                p = precision[cls_idx]
                if len(r) < 2:
                    continue

                # Average precision via trapezoidal integration
                ap_val = float(np.trapz(p, r))

                if ap_class_index is not None and cls_idx < len(ap_class_index):
                    real_cls = int(ap_class_index[cls_idx])
                    name = class_names.get(real_cls, f"class_{real_cls}")
                else:
                    name = f"class_{cls_idx}"

                ax.plot(r, p, color=cmap(cls_idx / max(num_classes - 1, 1)),
                        linewidth=2, label=f"{name} (AP={ap_val:.3f})")
                plotted = True

            return plotted

        except Exception as exc:
            logger.warning("Could not extract PR curve data: %s", exc)
            return False

    @staticmethod
    def _plot_pr_placeholder(ax) -> None:
        """Draw a 'no data' placeholder on *ax*."""
        ax.text(
            0.5, 0.5,
            "PR curve data not available\n(no ground-truth boxes or val() failed)",
            transform=ax.transAxes,
            ha="center", va="center",
            fontsize=12, color="grey",
            wrap=True,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    # ------------------------------------------------------------------
    # Private helpers — confusion matrix
    # ------------------------------------------------------------------

    def _extract_confusion_matrix(
        self, val_results, model
    ) -> Tuple[Optional[np.ndarray], List[str]]:
        """Extract confusion matrix array and class names from val_results.

        Returns a tuple ``(cm_array, class_names)`` where *cm_array* may be
        ``None`` if extraction fails.
        """
        class_names: List[str] = []
        if model is not None and hasattr(model, "names"):
            class_names = [model.names[k] for k in sorted(model.names.keys())]

        if val_results is None:
            return None, class_names or ["unknown"]

        try:
            # ultralytics stores the confusion matrix in val_results.confusion_matrix
            cm_obj = getattr(val_results, "confusion_matrix", None)
            if cm_obj is None:
                return None, class_names or ["unknown"]

            # The confusion matrix object has a .matrix attribute (numpy array)
            cm_array = None
            if hasattr(cm_obj, "matrix"):
                cm_array = np.array(cm_obj.matrix)
            elif isinstance(cm_obj, np.ndarray):
                cm_array = cm_obj

            if cm_array is None or cm_array.size == 0:
                return None, class_names or ["unknown"]

            # ultralytics includes a 'background' class as the last row/col
            # class_names from model.names may be shorter by 1 — pad if needed
            n = cm_array.shape[0]
            if len(class_names) < n:
                class_names = class_names + ["background"] * (n - len(class_names))
            elif len(class_names) > n:
                class_names = class_names[:n]

            return cm_array, class_names

        except Exception as exc:
            logger.warning("Could not extract confusion matrix data: %s", exc)
            return None, class_names or ["unknown"]

    @staticmethod
    def _plot_cm_placeholder(ax) -> None:
        """Draw a 'no data' placeholder on *ax*."""
        ax.text(
            0.5, 0.5,
            "Confusion matrix data not available\n(val() failed or no predictions)",
            transform=ax.transAxes,
            ha="center", va="center",
            fontsize=12, color="grey",
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    # ------------------------------------------------------------------
    # Private helpers — report serialisation
    # ------------------------------------------------------------------

    @staticmethod
    def _result_to_dict(result) -> dict:
        """Serialise an EvaluationResult to a plain dictionary.

        Mirrors the ``Evaluator.to_dict()`` format so consumers can use either
        source interchangeably.

        Args:
            result: EvaluationResult dataclass instance.

        Returns:
            JSON-serialisable dictionary.
        """
        dm = result.detection_metrics
        em = result.efficiency_metrics
        cm = result.complexity_metrics

        return {
            "detection_metrics": {
                "precision": dm.precision,
                "recall": dm.recall,
                "map50": dm.map50,
                "map50_95": dm.map50_95,
                "small_object_ap": dm.small_object_ap,
                "per_class_ap": dm.per_class_ap,
            },
            "efficiency_metrics": {
                "fps_mean": em.fps_mean,
                "fps_median": em.fps_median,
                "latency_ms_mean": em.latency_ms_mean,
                "latency_ms_median": em.latency_ms_median,
                "latency_ms_p95": em.latency_ms_p95,
            },
            "complexity_metrics": {
                "parameters": cm.parameters,
                "gflops": cm.gflops,
            },
            "total_images": result.total_images,
            "total_detections": result.total_detections,
            "evaluation_duration_seconds": result.evaluation_duration_seconds,
            "timestamp": result.timestamp,
        }

    @staticmethod
    def _build_markdown_report(result) -> str:
        """Build a human-readable Markdown report string.

        Sections:
        * Detection Performance
        * Efficiency Metrics
        * Model Complexity

        Args:
            result: EvaluationResult dataclass instance.

        Returns:
            Markdown-formatted string.
        """
        dm = result.detection_metrics
        em = result.efficiency_metrics
        cm = result.complexity_metrics

        lines = [
            "# UAV Survivor Detection — Evaluation Report",
            "",
            f"**Generated:** {result.timestamp}  ",
            f"**Total Images Evaluated:** {result.total_images}  ",
            f"**Total Detections:** {result.total_detections}  ",
            f"**Evaluation Duration:** {result.evaluation_duration_seconds:.2f} s",
            "",
            "---",
            "",
            "## Detection Performance",
            "",
            "| Metric | Value |",
            "| --- | --- |",
            f"| Precision | {dm.precision:.4f} |",
            f"| Recall | {dm.recall:.4f} |",
            f"| mAP@50 | {dm.map50:.4f} |",
            f"| mAP@50:95 | {dm.map50_95:.4f} |",
            f"| Small-Object AP (<32×32 px) | {dm.small_object_ap:.4f} |",
        ]

        if dm.per_class_ap:
            lines += [
                "",
                "### Per-Class Average Precision",
                "",
                "| Class | AP |",
                "| --- | --- |",
            ]
            for cls_name, ap_val in sorted(dm.per_class_ap.items()):
                lines.append(f"| {cls_name} | {ap_val:.4f} |")

        lines += [
            "",
            "---",
            "",
            "## Efficiency Metrics",
            "",
            "| Metric | Value |",
            "| --- | --- |",
            f"| FPS (mean) | {em.fps_mean:.2f} |",
            f"| FPS (median) | {em.fps_median:.2f} |",
            f"| Latency mean | {em.latency_ms_mean:.2f} ms |",
            f"| Latency median | {em.latency_ms_median:.2f} ms |",
            f"| Latency p95 | {em.latency_ms_p95:.2f} ms |",
            "",
            "---",
            "",
            "## Model Complexity",
            "",
            "| Metric | Value |",
            "| --- | --- |",
            f"| Parameters | {cm.parameters:,} |",
            f"| GFLOPs | {cm.gflops:.4f} |",
            "",
        ]

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Module-level convenience functions  (EvaluationResult-based API)
# ---------------------------------------------------------------------------

def generate_pr_curves(result: Any, output_path: Path) -> Path:
    """Generate per-class PR curves from a completed EvaluationResult and save as PNG.

    Uses ``result.detection_metrics.per_class_ap`` to plot one summary curve per class.
    Saves the figure to *output_path* and closes it to prevent memory leaks.

    Args:
        result: A completed
            :class:`~uav_survivor_detection.evaluation.EvaluationResult` instance.
        output_path: Destination path for the output PNG.  Parent directories
            are created automatically.

    Returns:
        Resolved :class:`pathlib.Path` pointing to the saved PNG.

    **Validates: Requirement 5.5**
    """
    return Visualizer().generate_pr_curves_from_result(result, Path(output_path))


def generate_confusion_matrix(result: Any, output_path: Path) -> Path:
    """Generate a confusion matrix image from a completed EvaluationResult and save as PNG.

    Because raw prediction arrays are not stored in
    :class:`~uav_survivor_detection.evaluation.EvaluationResult`, this function renders
    an informative summary placeholder.  For a full confusion matrix backed by
    ``model.val()`` data, use :class:`Visualizer` directly.

    Saves the figure to *output_path* and closes it to prevent memory leaks.

    Args:
        result: A completed
            :class:`~uav_survivor_detection.evaluation.EvaluationResult` instance.
        output_path: Destination path for the output PNG.  Parent directories
            are created automatically.

    Returns:
        Resolved :class:`pathlib.Path` pointing to the saved PNG.

    **Validates: Requirement 5.5**
    """
    return Visualizer().generate_confusion_matrix_from_result(result, Path(output_path))


def export_report(result: Any, output_dir: Path) -> Dict[str, Path]:
    """Export evaluation results as JSON and Markdown reports.

    Writes two files into *output_dir*:

    * ``evaluation_report.json`` — full metrics serialised with ``indent=2``.
    * ``evaluation_report.md`` — human-readable Markdown with sections for
      Detection Performance, Efficiency Metrics, and Model Complexity.

    Args:
        result: A completed
            :class:`~uav_survivor_detection.evaluation.EvaluationResult` instance.
        output_dir: Directory in which to write the report files.  Created
            if it does not exist.

    Returns:
        Dictionary with keys ``"json"`` and ``"markdown"`` mapping to the
        absolute :class:`pathlib.Path` of each written file.

    **Validates: Requirement 5.4**
    """
    return Visualizer().export_report(result, output_dir)


# ---------------------------------------------------------------------------
# Module-level convenience functions  (live-model-based API, kept for compat)
# ---------------------------------------------------------------------------

def generate_pr_curves_with_model(
    model: Any,
    dataset_path: str,
    output_path: str,
    conf: float = 0.001,
    iou: float = 0.5,
    imgsz: int = 640,
    device: str = "cpu",
) -> Path:
    """Generate per-class PR curves by re-running ``model.val()`` and save as PNG.

    This variant runs a live validation pass to extract raw precision/recall
    arrays.  Use :func:`generate_pr_curves` when you already have an
    :class:`~uav_survivor_detection.evaluation.EvaluationResult`.

    Args:
        model: A loaded ``ultralytics.YOLO`` model instance.
        dataset_path: Path to the dataset YAML file or directory.
        output_path: Destination path for the output PNG.
        conf: Confidence threshold.
        iou: IoU threshold.
        imgsz: Inference image size.
        device: Compute device string.

    Returns:
        Resolved :class:`pathlib.Path` pointing to the saved PNG.
    """
    return Visualizer().generate_pr_curves(
        model, dataset_path, output_path,
        conf=conf, iou=iou, imgsz=imgsz, device=device,
    )


def generate_confusion_matrix_with_model(
    model: Any,
    dataset_path: str,
    output_path: str,
    conf: float = 0.25,
    iou: float = 0.5,
    imgsz: int = 640,
    device: str = "cpu",
    normalize: bool = True,
) -> Path:
    """Generate a confusion matrix heatmap by re-running ``model.val()`` and save as PNG.

    This variant runs a live validation pass to extract the full confusion matrix.
    Use :func:`generate_confusion_matrix` when you already have an
    :class:`~uav_survivor_detection.evaluation.EvaluationResult`.

    Args:
        model: A loaded ``ultralytics.YOLO`` model instance.
        dataset_path: Path to the dataset YAML file or directory.
        output_path: Destination path for the output PNG.
        conf: Confidence threshold.
        iou: IoU threshold.
        imgsz: Inference image size.
        device: Compute device string.
        normalize: If ``True``, normalise the confusion matrix rows.

    Returns:
        Resolved :class:`pathlib.Path` pointing to the saved PNG.
    """
    return Visualizer().generate_confusion_matrix(
        model, dataset_path, output_path,
        conf=conf, iou=iou, imgsz=imgsz, device=device, normalize=normalize,
    )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _get_class_cmap(num_classes: int):
    """Return a matplotlib colour map suitable for *num_classes* curves.

    Uses ``tab10`` for ≤10 classes and ``tab20`` for up to 20, falling back
    to ``hsv`` for larger numbers.
    """
    import matplotlib.pyplot as plt

    if num_classes <= 10:
        return plt.get_cmap("tab10")
    if num_classes <= 20:
        return plt.get_cmap("tab20")
    return plt.get_cmap("hsv")
