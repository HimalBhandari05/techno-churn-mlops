"""Tests for Evidently AI data drift, target drift, synthetic drift, custom metrics, and MLflow logging."""

import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest

from app.config import Settings, get_settings
from app.data import load_raw_data
from app.drift import (
    CustomDriftMetricResult,
    calculate_churn_rate_shift,
    generate_data_drift_report,
    generate_target_drift_report,
    inject_synthetic_drift,
    split_monitoring_data,
)
from app.monitor import run_drift_monitoring


@pytest.fixture
def settings():
    """Project settings fixture."""
    return get_settings()


@pytest.fixture
def raw_df(settings):
    """Raw cleaned dataset."""
    return load_raw_data(settings=settings)


def test_split_monitoring_data_determinism(raw_df):
    """Test that monitoring data split is deterministic, 70/30 proportioned, and disjoint."""
    ref1, curr1 = split_monitoring_data(raw_df, reference_ratio=0.7, random_state=42)
    ref2, curr2 = split_monitoring_data(raw_df, reference_ratio=0.7, random_state=42)

    pd.testing.assert_frame_equal(ref1, ref2)
    pd.testing.assert_frame_equal(curr1, curr2)

    total_rows = len(raw_df)
    assert len(ref1) == int(total_rows * 0.7)
    assert len(curr1) == total_rows - len(ref1)

    # Verify no overlapping customerIDs between reference and current
    assert len(set(ref1["customerID"]).intersection(set(curr1["customerID"]))) == 0


def test_inject_synthetic_numeric_drift(raw_df):
    """Test that synthetic numeric perturbations alter distributions significantly."""
    _, curr = split_monitoring_data(raw_df, reference_ratio=0.7, random_state=42)
    drifted = inject_synthetic_drift(curr, random_state=42)

    # MonthlyCharges should increase by +30
    assert (drifted["MonthlyCharges"] > curr["MonthlyCharges"]).all()
    assert np.isclose((drifted["MonthlyCharges"] - curr["MonthlyCharges"]).mean(), 30.0)

    # tenure should be approximately halved
    assert (drifted["tenure"] <= curr["tenure"]).all()
    assert drifted["tenure"].mean() < curr["tenure"].mean()


def test_inject_synthetic_categorical_drift(raw_df):
    """Test that synthetic categorical perturbations shift category proportions."""
    _, curr = split_monitoring_data(raw_df, reference_ratio=0.7, random_state=42)
    drifted = inject_synthetic_drift(curr, random_state=42)

    # Month-to-month contracts should increase
    orig_month_ratio = (curr["Contract"] == "Month-to-month").mean()
    drifted_month_ratio = (drifted["Contract"] == "Month-to-month").mean()
    assert drifted_month_ratio > orig_month_ratio

    # Fiber optic internet service should increase
    orig_fiber_ratio = (curr["InternetService"] == "Fiber optic").mean()
    drifted_fiber_ratio = (drifted["InternetService"] == "Fiber optic").mean()
    assert drifted_fiber_ratio > orig_fiber_ratio


def test_inject_synthetic_target_drift(raw_df):
    """Test that synthetic target perturbation increases churn rate."""
    _, curr = split_monitoring_data(raw_df, reference_ratio=0.7, random_state=42)
    drifted = inject_synthetic_drift(curr, random_state=42)

    orig_churn_rate = (curr["Churn"] == "Yes").mean()
    drifted_churn_rate = (drifted["Churn"] == "Yes").mean()

    assert drifted_churn_rate > orig_churn_rate
    assert drifted_churn_rate > 0.40


def test_calculate_churn_rate_shift_pass_and_fail(raw_df):
    """Test custom metric evaluation logic under passing and failing conditions."""
    ref, curr = split_monitoring_data(raw_df, reference_ratio=0.7, random_state=42)

    # 1. Unperturbed (Pass test: churn rates are very close)
    res_pass = calculate_churn_rate_shift(ref, curr, threshold=0.05)
    assert isinstance(res_pass, CustomDriftMetricResult)
    assert res_pass.passed is True
    assert res_pass.status == "PASS"
    assert res_pass.difference <= 0.05

    # 2. Perturbed (Fail test: drift exceeds threshold)
    drifted = inject_synthetic_drift(curr, random_state=42)
    res_fail = calculate_churn_rate_shift(ref, drifted, threshold=0.05)
    assert res_fail.passed is False
    assert res_fail.status == "FAIL (DRIFT DETECTED)"
    assert res_fail.difference > 0.05
    assert "Significant business drift detected" in res_fail.production_interpretation


def test_generate_data_drift_report(raw_df, tmp_path, settings):
    """Test Evidently Data Drift report generation and machine-readable summary."""
    ref, curr = split_monitoring_data(raw_df, reference_ratio=0.7, random_state=42)
    drifted = inject_synthetic_drift(curr, random_state=42)

    html_file = tmp_path / "data_drift.html"
    out_path, summary = generate_data_drift_report(
        df_reference=ref,
        df_current=drifted,
        output_html_path=html_file,
        columns_to_include=settings.feature_columns,
    )

    assert out_path.exists()
    assert out_path.stat().st_size > 1000
    assert summary["dataset_drift_detected"] is True
    assert summary["drifted_columns_count"] >= 6

    # Verify our deliberately perturbed columns are detected
    expected_drifted = ["MonthlyCharges", "tenure", "TotalCharges", "Contract", "InternetService", "PaymentMethod"]
    for col in expected_drifted:
        assert col in summary["drifted_columns"]


def test_generate_target_drift_report(raw_df, tmp_path):
    """Test Evidently Target Drift report generation and distribution extraction."""
    ref, curr = split_monitoring_data(raw_df, reference_ratio=0.7, random_state=42)
    drifted = inject_synthetic_drift(curr, random_state=42)

    html_file = tmp_path / "target_drift.html"
    out_path, summary = generate_target_drift_report(
        df_reference=ref,
        df_current=drifted,
        output_html_path=html_file,
        target_column="Churn",
    )

    assert out_path.exists()
    assert out_path.stat().st_size > 1000
    assert summary["target_drift_detected"] is True
    assert summary["current_distribution"]["Yes"] > summary["reference_distribution"]["Yes"]


def test_run_drift_monitoring_mlflow_pipeline(tmp_path):
    """Test complete monitoring workflow with MLflow run creation and artifact logging."""
    db_file = tmp_path / "monitor_mlflow.db"
    test_settings = Settings(
        mlflow_tracking_uri=f"sqlite:///{db_file}",
        reports_dir=tmp_path / "reports",
        models_dir=tmp_path / "models",
    )

    result = run_drift_monitoring(
        experiment_name="test-monitoring-experiment",
        custom_threshold=0.05,
        settings=test_settings,
    )

    assert "run_id" in result
    run_id = result["run_id"]

    # Verify run recorded in MLflow
    client = mlflow.tracking.MlflowClient(tracking_uri=test_settings.mlflow_tracking_uri)
    run = client.get_run(run_id)

    assert run.info.status == "FINISHED"
    assert "drifted_columns_count" in run.data.metrics
    assert "churn_rate_shift" in run.data.metrics
    assert run.data.metrics["drifted_columns_count"] >= 6

    # Verify artifacts logged
    artifacts = client.list_artifacts(run_id)
    artifact_dirs = [a.path for a in artifacts]
    assert "evidently_reports" in artifact_dirs
    assert "monitoring_summaries" in artifact_dirs
