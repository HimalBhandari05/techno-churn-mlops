"""Tests for evaluation metrics and visualization artifact generation."""

import json
from pathlib import Path

import numpy as np
import pytest

from app.evaluate import (
    calculate_classification_metrics,
    generate_and_save_artifacts,
    generate_classification_report,
    plot_confusion_matrix,
    plot_roc_curve,
)


def test_calculate_classification_metrics_exact():
    """Test metric calculations against known ground truth."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.4, 0.8])

    metrics = calculate_classification_metrics(y_true, y_pred, y_prob)

    assert metrics["accuracy"] == 0.5
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5
    assert "roc_auc" in metrics
    assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_calculate_classification_metrics_no_proba():
    """Test metric calculations when predicted probabilities are omitted."""
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 0])

    metrics = calculate_classification_metrics(y_true, y_pred)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert "roc_auc" not in metrics


def test_plot_confusion_matrix(tmp_path):
    """Test generation and saving of Confusion Matrix plot."""
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 0, 1, 0, 0, 1])
    out_file = tmp_path / "cm.png"

    result_path = plot_confusion_matrix(y_true, y_pred, out_file)

    assert result_path.exists()
    assert result_path.stat().st_size > 0


def test_plot_roc_curve(tmp_path):
    """Test generation and saving of ROC Curve plot."""
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    out_file = tmp_path / "roc.png"

    result_path = plot_roc_curve(y_true, y_prob, out_file, model_name="TestModel")

    assert result_path.exists()
    assert result_path.stat().st_size > 0


def test_generate_classification_report():
    """Test classification report generation in text and dict formats."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])

    text_rep, dict_rep = generate_classification_report(y_true, y_pred)

    assert isinstance(text_rep, str)
    assert "precision" in text_rep
    assert isinstance(dict_rep, dict)
    assert "accuracy" in dict_rep
    assert "macro avg" in dict_rep


def test_generate_and_save_artifacts(tmp_path):
    """Test batch artifact generation creates all expected files."""
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 0, 1, 0, 0, 1])
    y_prob = np.array([0.1, 0.2, 0.7, 0.4, 0.3, 0.8])

    artifacts = generate_and_save_artifacts(
        y_true=y_true,
        y_pred=y_pred,
        y_pred_proba=y_prob,
        output_dir=tmp_path,
        model_name="UnitTestingModel",
    )

    expected_keys = [
        "confusion_matrix",
        "roc_curve",
        "classification_report_txt",
        "classification_report_json",
        "evaluation_metrics",
    ]
    for key in expected_keys:
        assert key in artifacts
        file_path = artifacts[key]
        assert Path(file_path).exists()
        assert Path(file_path).stat().st_size > 0

    # Verify metrics JSON is valid JSON
    with open(artifacts["evaluation_metrics"], "r", encoding="utf-8") as f:
        data = json.load(f)
        assert "accuracy" in data
        assert "roc_auc" in data
