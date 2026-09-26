"""Evaluation utilities, metric calculations, and visualization artifact generation."""

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def calculate_classification_metrics(
    y_true: np.ndarray | list[int],
    y_pred: np.ndarray | list[int],
    y_pred_proba: np.ndarray | list[float] | None = None,
) -> dict[str, float]:
    """Calculate key classification evaluation metrics.

    Args:
        y_true: Ground truth binary target labels (0 or 1).
        y_pred: Predicted binary labels (0 or 1).
        y_pred_proba: Predicted probabilities for positive class (1).

    Returns:
        Dictionary containing accuracy, precision, recall, f1, and roc_auc.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "precision": float(precision_score(y_true_arr, y_pred_arr, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred_arr, zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred_arr, zero_division=0)),
    }

    if y_pred_proba is not None:
        y_prob_arr = np.asarray(y_pred_proba, dtype=float)
        metrics["roc_auc"] = float(roc_auc_score(y_true_arr, y_prob_arr))

    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray | list[int],
    y_pred: np.ndarray | list[int],
    output_path: Path | str,
    title: str = "Confusion Matrix",
    display_labels: list[str] | None = None,
) -> Path:
    """Generate and save Confusion Matrix plot as an image artifact.

    Args:
        y_true: Ground truth binary target labels.
        y_pred: Predicted binary labels.
        output_path: Destination image path.
        title: Plot title.
        display_labels: Class label names.

    Returns:
        Path to the saved PNG image.
    """
    labels = display_labels or ["No Churn (0)", "Churn (1)"]
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return out_path


def plot_roc_curve(
    y_true: np.ndarray | list[int],
    y_pred_proba: np.ndarray | list[float],
    output_path: Path | str,
    model_name: str = "Classifier",
) -> Path:
    """Generate and save ROC Curve plot as an image artifact.

    Args:
        y_true: Ground truth binary target labels.
        y_pred_proba: Predicted probabilities for positive class.
        output_path: Destination image path.
        model_name: Model identifier for label and title.

    Returns:
        Path to the saved PNG image.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_pred_proba, dtype=float)

    fpr, tpr, _ = roc_curve(y_true_arr, y_prob_arr)
    score = roc_auc_score(y_true_arr, y_prob_arr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(
        fpr,
        tpr,
        color="#1f77b4",
        lw=2,
        label=f"{model_name} (AUC = {score:.4f})",
    )
    # Add random chance diagonal
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Chance (AUC = 0.50)")
    ax.set_title(f"ROC Curve — {model_name}", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=10)
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return out_path


def generate_classification_report(
    y_true: np.ndarray | list[int],
    y_pred: np.ndarray | list[int],
    target_names: list[str] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Generate classification report in both text and dictionary formats.

    Args:
        y_true: Ground truth binary target labels.
        y_pred: Predicted binary labels.
        target_names: Custom names for classes.

    Returns:
        Tuple of (text_report, dict_report).
    """
    names = target_names or ["No Churn (0)", "Churn (1)"]
    text_rep = classification_report(
        y_true,
        y_pred,
        target_names=names,
        zero_division=0,
    )
    dict_rep = classification_report(
        y_true,
        y_pred,
        target_names=names,
        output_dict=True,
        zero_division=0,
    )
    return str(text_rep), dict_rep


def generate_and_save_artifacts(
    y_true: np.ndarray | list[int],
    y_pred: np.ndarray | list[int],
    y_pred_proba: np.ndarray | list[float],
    output_dir: Path | str,
    model_name: str = "Classifier",
) -> dict[str, Path]:
    """Generate and write all evaluation artifacts to disk.

    Args:
        y_true: Ground truth binary target labels.
        y_pred: Predicted binary labels.
        y_pred_proba: Predicted probabilities for positive class.
        output_dir: Output directory for saving artifacts.
        model_name: Name of the model configuration.

    Returns:
        Dictionary mapping artifact keys to their resolved file paths.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cm_path = out_dir / "confusion_matrix.png"
    roc_path = out_dir / "roc_curve.png"
    report_txt_path = out_dir / "classification_report.txt"
    report_json_path = out_dir / "classification_report.json"
    metrics_json_path = out_dir / "evaluation_metrics.json"

    # 1. Plot Confusion Matrix
    plot_confusion_matrix(y_true, y_pred, cm_path, title=f"Confusion Matrix — {model_name}")

    # 2. Plot ROC Curve
    plot_roc_curve(y_true, y_pred_proba, roc_path, model_name=model_name)

    # 3. Generate Classification Reports
    text_rep, dict_rep = generate_classification_report(y_true, y_pred)
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write(f"Classification Report — {model_name}\n")
        f.write("=" * 60 + "\n\n")
        f.write(text_rep)

    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(dict_rep, f, indent=2)

    # 4. Save Metrics Summary JSON
    metrics = calculate_classification_metrics(y_true, y_pred, y_pred_proba)
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return {
        "confusion_matrix": cm_path,
        "roc_curve": roc_path,
        "classification_report_txt": report_txt_path,
        "classification_report_json": report_json_path,
        "evaluation_metrics": metrics_json_path,
    }
