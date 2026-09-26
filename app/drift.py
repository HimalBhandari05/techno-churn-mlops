"""Evidently AI data drift, target drift, synthetic drift injection, and custom metrics module."""

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from app.config import Settings, get_settings
from app.data import load_raw_data


@dataclass
class CustomDriftMetricResult:
    """Structured result of custom drift metric evaluation."""

    metric_name: str
    formula: str
    threshold: float
    reference_value: float
    current_value: float
    difference: float
    passed: bool
    status: str
    production_interpretation: str

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return asdict(self)


def split_monitoring_data(
    df: pd.DataFrame,
    reference_ratio: float = 0.7,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataset deterministically into reference (historical baseline) and current (production) sets.

    Args:
        df: Input cleaned DataFrame.
        reference_ratio: Proportion for reference dataset (default: 0.7).
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (df_reference, df_current).
    """
    rng = np.random.default_rng(random_state)
    indices = np.arange(len(df))
    rng.shuffle(indices)

    split_idx = int(len(df) * reference_ratio)
    ref_idx = indices[:split_idx]
    curr_idx = indices[split_idx:]

    df_reference = df.iloc[ref_idx].copy().reset_index(drop=True)
    df_current = df.iloc[curr_idx].copy().reset_index(drop=True)

    return df_reference, df_current


def inject_synthetic_drift(
    df_current: pd.DataFrame,
    random_state: int = 42,
) -> pd.DataFrame:
    """Inject controlled, reproducible synthetic numeric, categorical, and target drift.

    Modifications:
    1. Numerical Drift:
       - MonthlyCharges: Add +$30.00 (simulating price hikes)
       - tenure: Scaled down by 0.5 (simulating influx of newer customers)
       - TotalCharges: Recalculated / shifted proportionally
    2. Categorical Drift:
       - Contract: Shift 80% of 'Two year' and 'One year' to 'Month-to-month'
       - InternetService: Shift 65% of 'DSL' to 'Fiber optic'
       - PaymentMethod: Shift 60% of automatic payments to 'Electronic check'
    3. Target Drift:
       - Churn: Shift 30% of 'No' to 'Yes' (simulating churn surge due to pricing/contract changes)

    Args:
        df_current: Original current DataFrame.
        random_state: Random seed for perturbation.

    Returns:
        Perturbed current DataFrame.
    """
    rng = np.random.default_rng(random_state)
    df_drifted = df_current.copy()

    # 1. Numerical Drift Injection
    if "MonthlyCharges" in df_drifted.columns:
        df_drifted["MonthlyCharges"] = df_drifted["MonthlyCharges"] + 30.0

    if "tenure" in df_drifted.columns:
        df_drifted["tenure"] = np.maximum(0, (df_drifted["tenure"] * 0.5).astype(int))

    if "TotalCharges" in df_drifted.columns:
        df_drifted["TotalCharges"] = np.maximum(
            0.0, df_drifted["MonthlyCharges"] * np.maximum(1, df_drifted["tenure"])
        )

    # 2. Categorical Drift Injection
    if "Contract" in df_drifted.columns:
        mask_contract = (df_drifted["Contract"].isin(["One year", "Two year"])) & (
            rng.random(len(df_drifted)) < 0.80
        )
        df_drifted.loc[mask_contract, "Contract"] = "Month-to-month"

    if "InternetService" in df_drifted.columns:
        mask_internet = (df_drifted["InternetService"] == "DSL") & (
            rng.random(len(df_drifted)) < 0.65
        )
        df_drifted.loc[mask_internet, "InternetService"] = "Fiber optic"

    if "PaymentMethod" in df_drifted.columns:
        mask_payment = (
            df_drifted["PaymentMethod"].isin(
                ["Bank transfer (automatic)", "Credit card (automatic)"]
            )
        ) & (rng.random(len(df_drifted)) < 0.60)
        df_drifted.loc[mask_payment, "PaymentMethod"] = "Electronic check"

    # 3. Target Drift Injection
    if "Churn" in df_drifted.columns:
        mask_churn = (df_drifted["Churn"] == "No") & (
            rng.random(len(df_drifted)) < 0.30
        )
        df_drifted.loc[mask_churn, "Churn"] = "Yes"

    return df_drifted


def calculate_churn_rate_shift(
    df_reference: pd.DataFrame,
    df_current: pd.DataFrame,
    target_column: str = "Churn",
    positive_label: str = "Yes",
    threshold: float = 0.05,
) -> CustomDriftMetricResult:
    """Compute custom Churn Rate Shift metric comparing current and reference churn rates.

    Formula:
        churn_rate_shift = abs(current_churn_rate - reference_churn_rate)

    Args:
        df_reference: Reference baseline DataFrame.
        df_current: Current evaluation DataFrame.
        target_column: Churn column name.
        positive_label: Positive class label ('Yes' or 1).
        threshold: Maximum allowed absolute rate shift (default: 0.05).

    Returns:
        CustomDriftMetricResult with evaluation status and interpretation.
    """
    ref_churn = (df_reference[target_column].astype(str) == str(positive_label)).mean()
    curr_churn = (df_current[target_column].astype(str) == str(positive_label)).mean()
    shift = abs(curr_churn - ref_churn)
    passed = bool(shift <= threshold)

    status_str = "PASS" if passed else "FAIL (DRIFT DETECTED)"
    interpretation = (
        f"Churn rate shifted by {shift * 100:.2f}% (from {ref_churn * 100:.2f}% to {curr_churn * 100:.2f}%). "
        f"Tolerance threshold is {threshold * 100:.2f}%. "
        + (
            "System behavior is within normal operating limits."
            if passed
            else "Significant business drift detected! Recommended action: investigate upstream billing/contract adjustments and evaluate model performance."
        )
    )

    return CustomDriftMetricResult(
        metric_name="churn_rate_shift",
        formula="abs(current_churn_rate - reference_churn_rate)",
        threshold=threshold,
        reference_value=round(float(ref_churn), 4),
        current_value=round(float(curr_churn), 4),
        difference=round(float(shift), 4),
        passed=passed,
        status=status_str,
        production_interpretation=interpretation,
    )


def _parse_value_drift_metric(metric_name: str, value: Any) -> tuple[str | None, str, float, float, bool]:
    """Helper to parse ValueDrift metric string, threshold, test method, and drift status."""
    col_match = re.search(r"column=([^,)]+)", metric_name)
    method_match = re.search(r"method=([^,)]+)", metric_name)
    thresh_match = re.search(r"threshold=([^,)]+)", metric_name)

    col = col_match.group(1) if col_match else None
    method = method_match.group(1) if method_match else ""
    thresh = float(thresh_match.group(1)) if thresh_match else 0.05
    score = float(value) if isinstance(value, (int, float)) else 0.0

    if "p_value" in method.lower():
        is_drift = score < thresh
    else:
        is_drift = score > thresh

    return col, method, thresh, score, is_drift


def generate_data_drift_report(
    df_reference: pd.DataFrame,
    df_current: pd.DataFrame,
    output_html_path: Path | str,
    columns_to_include: list[str] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Generate and save Evidently Data Drift HTML report and extract machine-readable summary.

    Args:
        df_reference: Reference baseline DataFrame.
        df_current: Current evaluation DataFrame.
        output_html_path: Destination HTML report path.
        columns_to_include: Specific feature column names to analyze.

    Returns:
        Tuple of (html_path, machine_readable_summary_dict).
    """
    out_path = Path(output_html_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ref_data = df_reference[columns_to_include].copy() if columns_to_include else df_reference.copy()
    curr_data = df_current[columns_to_include].copy() if columns_to_include else df_current.copy()

    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=ref_data, current_data=curr_data)
    snapshot.save_html(str(out_path))

    # Extract machine-readable metrics from snapshot
    metrics_data = snapshot.dict().get("metrics", [])
    drifted_columns = []
    drift_by_col: dict[str, Any] = {}
    drifted_count = 0
    drift_share = 0.0

    for m in metrics_data:
        m_name = m.get("metric_name", "")
        if "DriftedColumnsCount" in m_name:
            val = m.get("value", {})
            if isinstance(val, dict):
                drifted_count = int(val.get("count", 0))
                drift_share = float(val.get("share", 0.0))
        elif "ValueDrift" in m_name:
            col, method, thresh, score, is_drift = _parse_value_drift_metric(m_name, m.get("value", 0.0))
            if col:
                drift_by_col[col] = {
                    "method": method,
                    "score": round(score, 6),
                    "threshold": thresh,
                    "drift_detected": is_drift,
                }
                if is_drift and col not in drifted_columns:
                    drifted_columns.append(col)

    total_cols = len(ref_data.columns)
    summary = {
        "report_type": "DataDrift",
        "total_columns_analyzed": total_cols,
        "drifted_columns_count": len(drifted_columns),
        "drifted_columns_ratio": round(len(drifted_columns) / total_cols if total_cols > 0 else 0.0, 4),
        "dataset_drift_detected": bool(len(drifted_columns) > 0),
        "drifted_columns": sorted(drifted_columns),
        "column_drift_details": drift_by_col,
    }

    return out_path, summary


def generate_target_drift_report(
    df_reference: pd.DataFrame,
    df_current: pd.DataFrame,
    output_html_path: Path | str,
    target_column: str = "Churn",
) -> tuple[Path, dict[str, Any]]:
    """Generate and save Evidently Target Drift HTML report comparing target distributions.

    Args:
        df_reference: Reference baseline DataFrame.
        df_current: Current evaluation DataFrame.
        output_html_path: Destination HTML report path.
        target_column: Name of target column (default: 'Churn').

    Returns:
        Tuple of (html_path, machine_readable_summary_dict).
    """
    out_path = Path(output_html_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ref_target = df_reference[[target_column]].copy()
    curr_target = df_current[[target_column]].copy()

    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=ref_target, current_data=curr_target)
    snapshot.save_html(str(out_path))

    metrics_data = snapshot.dict().get("metrics", [])
    target_drift_detected = False
    score = 0.0
    method = ""
    threshold = 0.1

    for m in metrics_data:
        m_name = m.get("metric_name", "")
        if "ValueDrift" in m_name:
            col, method, threshold, score, target_drift_detected = _parse_value_drift_metric(
                m_name, m.get("value", 0.0)
            )

    ref_counts = df_reference[target_column].value_counts(normalize=True).round(4).to_dict()
    curr_counts = df_current[target_column].value_counts(normalize=True).round(4).to_dict()

    summary = {
        "report_type": "TargetDrift",
        "target_column": target_column,
        "target_drift_detected": target_drift_detected,
        "method": method,
        "drift_score": round(score, 6),
        "threshold": threshold,
        "reference_distribution": {str(k): float(v) for k, v in ref_counts.items()},
        "current_distribution": {str(k): float(v) for k, v in curr_counts.items()},
    }

    return out_path, summary
