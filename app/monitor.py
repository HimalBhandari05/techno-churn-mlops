"""MLflow monitoring runner orchestrating Evidently AI drift reports and custom metrics logging."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import mlflow

from app.config import Settings, get_settings
from app.data import load_raw_data
from app.drift import (
    calculate_churn_rate_shift,
    generate_data_drift_report,
    generate_target_drift_report,
    inject_synthetic_drift,
    split_monitoring_data,
)


def run_drift_monitoring(
    data_path: Path | str | None = None,
    experiment_name: str = "week17-track-a-telco-churn-monitoring",
    custom_threshold: float = 0.05,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Execute end-to-end Evidently AI drift monitoring and log results/reports to MLflow.

    Args:
        data_path: Optional path to raw dataset CSV.
        experiment_name: MLflow experiment name for monitoring.
        custom_threshold: Allowable churn rate shift threshold.
        settings: Settings instance.

    Returns:
        Dictionary containing monitoring results, MLflow run ID, and artifact paths.
    """
    settings = settings or get_settings()

    print("=" * 70)
    print("TELCO CHURN MLOPS — PHASE 4 EVIDENTLY AI DRIFT MONITORING")
    print("=" * 70)

    # 1. Load Cleaned Dataset
    print("\n[Step 1/6] Loading dataset...")
    df = load_raw_data(data_path=data_path, settings=settings)
    print(f"    Loaded dataset shape: {df.shape}")

    # 2. Deterministic Reference / Current Split (70% / 30%)
    print("\n[Step 2/6] Splitting dataset into Reference (70%) and Current (30%)...")
    df_ref, df_curr = split_monitoring_data(
        df,
        reference_ratio=0.7,
        random_state=settings.random_state,
    )
    print(f"    Reference dataset rows: {len(df_ref)} (historical baseline)")
    print(f"    Current dataset rows:   {len(df_curr)} (production-like)")

    # 3. Synthetic Drift Injection on Current Dataset
    print("\n[Step 3/6] Injecting controlled synthetic numeric, categorical, and target drift...")
    df_drifted = inject_synthetic_drift(df_curr, random_state=settings.random_state)
    print("    Injected Numerical Drift: MonthlyCharges (+30), tenure (*0.5), TotalCharges")
    print("    Injected Categorical Drift: Contract (-> Month-to-month), InternetService (-> Fiber optic), PaymentMethod (-> Electronic check)")
    print("    Injected Target Drift: Churn (shifted ~30% 'No' -> 'Yes')")

    # 4. Generate Evidently Data Drift Report
    print("\n[Step 4/6] Generating Evidently Data Drift Report...")
    reports_dir = settings.resolve_path(settings.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    data_drift_html = reports_dir / "evidently_data_drift.html"

    feature_cols = settings.feature_columns
    _, data_drift_summary = generate_data_drift_report(
        df_reference=df_ref,
        df_current=df_drifted,
        output_html_path=data_drift_html,
        columns_to_include=feature_cols,
    )
    print(f"    Data Drift Report saved: {data_drift_html}")
    print(f"    Total Features Analyzed: {data_drift_summary['total_columns_analyzed']}")
    print(f"    Drifted Columns Detected: {data_drift_summary['drifted_columns_count']} ({data_drift_summary['drifted_columns_ratio'] * 100:.1f}%)")
    print(f"    Drifted Feature Names: {data_drift_summary['drifted_columns']}")

    # 5. Generate Evidently Target Drift Report
    print("\n[Step 5/6] Generating Evidently Target Drift Report...")
    target_drift_html = reports_dir / "evidently_target_drift.html"
    _, target_drift_summary = generate_target_drift_report(
        df_reference=df_ref,
        df_current=df_drifted,
        output_html_path=target_drift_html,
        target_column=settings.target_column,
    )
    print(f"    Target Drift Report saved: {target_drift_html}")
    print(f"    Target Drift Detected: {target_drift_summary['target_drift_detected']} (score: {target_drift_summary.get('drift_score', 0.0):.4f})")
    print(f"    Reference Target Distribution: {target_drift_summary['reference_distribution']}")
    print(f"    Current Target Distribution:   {target_drift_summary['current_distribution']}")

    # 6. Calculate Custom Churn Rate Shift Metric
    print("\n[Step 6/6] Evaluating Custom Churn Rate Shift Metric...")
    custom_metric = calculate_churn_rate_shift(
        df_reference=df_ref,
        df_current=df_drifted,
        target_column=settings.target_column,
        positive_label="Yes",
        threshold=custom_threshold,
    )
    print(f"    Metric: {custom_metric.metric_name} = {custom_metric.difference:.4f} (Threshold: {custom_metric.threshold})")
    print(f"    Status: {custom_metric.status}")
    print(f"    Interpretation: {custom_metric.production_interpretation}")

    # Save machine-readable JSON summaries
    summary_json_path = reports_dir / "monitoring_summary.json"
    custom_metric_json_path = reports_dir / "custom_metric.json"

    full_summary = {
        "dataset_name": "Telco-Customer-Churn",
        "reference_rows": len(df_ref),
        "current_rows": len(df_curr),
        "data_drift": data_drift_summary,
        "target_drift": target_drift_summary,
        "custom_metric": custom_metric.to_dict(),
    }

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(full_summary, f, indent=2)

    with open(custom_metric_json_path, "w", encoding="utf-8") as f:
        json.dump(custom_metric.to_dict(), f, indent=2)

    # 7. Log to MLflow
    print("\n[MLflow] Logging monitoring run and artifacts to MLflow...")
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="evidently_drift_monitoring") as run:
        run_id = run.info.run_id
        print(f"    MLflow Monitoring Run ID: {run_id}")

        # Parameters
        mlflow.log_params({
            "reference_ratio": 0.7,
            "reference_rows": len(df_ref),
            "current_rows": len(df_curr),
            "random_seed": settings.random_state,
            "custom_metric_threshold": custom_threshold,
            "features_analyzed_count": len(feature_cols),
        })

        # Metrics
        mlflow.log_metrics({
            "drifted_columns_count": float(data_drift_summary["drifted_columns_count"]),
            "drifted_columns_ratio": float(data_drift_summary["drifted_columns_ratio"]),
            "dataset_drift_detected": 1.0 if data_drift_summary["dataset_drift_detected"] else 0.0,
            "target_drift_detected": 1.0 if target_drift_summary["target_drift_detected"] else 0.0,
            "target_drift_score": float(target_drift_summary.get("drift_score", 0.0)),
            "reference_churn_rate": float(custom_metric.reference_value),
            "current_churn_rate": float(custom_metric.current_value),
            "churn_rate_shift": float(custom_metric.difference),
            "custom_metric_pass": 1.0 if custom_metric.passed else 0.0,
        })

        # Artifacts
        mlflow.log_artifact(str(data_drift_html), artifact_path="evidently_reports")
        mlflow.log_artifact(str(target_drift_html), artifact_path="evidently_reports")
        mlflow.log_artifact(str(summary_json_path), artifact_path="monitoring_summaries")
        mlflow.log_artifact(str(custom_metric_json_path), artifact_path="monitoring_summaries")

    print("\n" + "=" * 70)
    print("MONITORING PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)

    return {
        "run_id": run_id,
        "experiment_name": experiment_name,
        "data_drift": data_drift_summary,
        "target_drift": target_drift_summary,
        "custom_metric": custom_metric.to_dict(),
        "artifacts": {
            "data_drift_html": str(data_drift_html),
            "target_drift_html": str(target_drift_html),
            "summary_json": str(summary_json_path),
            "custom_metric_json": str(custom_metric_json_path),
        },
    }


def main() -> None:
    """CLI entrypoint for running drift monitoring."""
    parser = argparse.ArgumentParser(description="Execute Evidently AI Drift Monitoring")
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to Telco Customer Churn CSV dataset",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Allowable churn rate shift threshold (default: 0.05)",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="week17-track-a-telco-churn-monitoring",
        help="MLflow monitoring experiment name",
    )
    args = parser.parse_args()

    try:
        run_drift_monitoring(
            data_path=args.data_path,
            experiment_name=args.experiment_name,
            custom_threshold=args.threshold,
        )
    except Exception as exc:
        print(f"Error executing drift monitoring: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
