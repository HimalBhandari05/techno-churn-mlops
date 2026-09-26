"""ML training pipeline and MLflow experiment tracking runner."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from app.config import Settings, get_settings
from app.data import load_raw_data, split_data
from app.evaluate import calculate_classification_metrics, generate_and_save_artifacts
from app.models import ModelConfig, get_model_configurations
from app.preprocessing import (
    build_preprocessor,
    encode_target,
    fit_and_transform,
    save_preprocessor,
    transform_data,
)


def train_and_log_model(
    model_config: ModelConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    preprocessor_path: Path,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Train estimator, evaluate metrics, and log parameters/metrics/artifacts to MLflow.

    Args:
        model_config: ModelConfig specifying estimator and hyperparameters.
        X_train: Transformed training feature matrix.
        y_train: Encoded training binary target labels.
        X_test: Transformed test feature matrix.
        y_test: Encoded test binary target labels.
        preprocessor_path: Path to serialized preprocessor joblib file.
        settings: Settings instance.

    Returns:
        Dictionary containing run_id, run_name, model_type, metrics, and artifact directory.
    """
    settings = settings or get_settings()

    # Ensure tracking URI and experiment are configured
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    # Temporary directory for local artifact generation
    local_artifact_dir = settings.resolve_path(settings.reports_dir / "artifacts" / model_config.name)
    local_artifact_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name=model_config.name) as run:
        run_id = run.info.run_id
        print(f"\n---> Starting MLflow Run: {model_config.name} (Run ID: {run_id})")

        # 1. Log Parameters
        params_to_log: dict[str, Any] = {
            "model_type": model_config.model_type,
            "random_seed": settings.random_state,
            "test_size": settings.test_size,
            "stratified_split": settings.stratify,
            "preprocessing_pipeline": "SimpleImputer(median/mode) + StandardScaler + OneHotEncoder",
            "num_features_in": int(X_train.shape[1]),
        }
        for k, v in model_config.hyperparameters.items():
            params_to_log[f"param_{k}"] = str(v) if v is not None else "None"

        mlflow.log_params(params_to_log)
        mlflow.set_tags({
            "dataset": "Telco-Customer-Churn",
            "phase": "phase_2_training",
            "model_family": model_config.model_type,
        })

        # 2. Train Estimator
        estimator = model_config.build_estimator()
        estimator.fit(X_train, y_train)

        # 3. Predict & Calculate Metrics
        y_pred = estimator.predict(X_test)
        if hasattr(estimator, "predict_proba"):
            y_pred_proba = estimator.predict_proba(X_test)[:, 1]
        elif hasattr(estimator, "decision_function"):
            df_vals = estimator.decision_function(X_test)
            # Sigmoid conversion
            y_pred_proba = 1 / (1 + np.exp(-df_vals))
        else:
            y_pred_proba = None

        metrics = calculate_classification_metrics(y_test, y_pred, y_pred_proba)

        # 4. Log Metrics
        mlflow.log_metrics(metrics)

        print(f"     Metrics for {model_config.name}:")
        for m_name, m_val in metrics.items():
            print(f"       - {m_name.upper():<10}: {m_val:.4f}")

        # 5. Generate & Log Evaluation Artifacts (Plots, Reports, Metrics JSON)
        artifacts = generate_and_save_artifacts(
            y_true=y_test,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba if y_pred_proba is not None else y_pred,
            output_dir=local_artifact_dir,
            model_name=model_config.name,
        )

        for artifact_key, artifact_file in artifacts.items():
            mlflow.log_artifact(str(artifact_file), artifact_path="evaluation")

        # 6. Log Preprocessor Artifact
        mlflow.log_artifact(str(preprocessor_path), artifact_path="preprocessing")

        # 7. Save and Log Model Artifacts
        local_model_path = settings.resolve_path(settings.models_dir / f"{model_config.name}.joblib")
        local_model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(estimator, local_model_path)
        mlflow.log_artifact(str(local_model_path), artifact_path="model")

        mlflow.sklearn.log_model(
            sk_model=estimator,
            artifact_path="sklearn_model",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            input_example=X_test[:5],
        )

        print(f"     Artifacts successfully logged to MLflow.")

        return {
            "run_id": run_id,
            "run_name": model_config.name,
            "model_type": model_config.model_type,
            "metrics": metrics,
            "hyperparameters": model_config.hyperparameters,
            "artifacts_dir": str(local_artifact_dir),
            "model_path": str(local_model_path),
        }


def run_experiments(
    model_names: list[str] | None = None,
    data_path: Path | str | None = None,
    settings: Settings | None = None,
) -> pd.DataFrame:
    """Execute training, evaluation, and MLflow logging for multiple model configurations.

    Args:
        model_names: List of configuration keys to run. If None, runs all configured models.
        data_path: Optional custom path to dataset CSV.
        settings: Settings instance.

    Returns:
        pandas DataFrame comparing all runs.
    """
    settings = settings or get_settings()

    print("=" * 75)
    print(f"TELCO CHURN MLOPS — PHASE 2 MLFLOW EXPERIMENT TRACKING")
    print(f"Experiment Name: {settings.mlflow_experiment_name}")
    print(f"Tracking URI:    {settings.mlflow_tracking_uri}")
    print("=" * 75)

    # 1. Load Data
    print("\n[Step 1/3] Loading dataset and performing stratified train/test split...")
    df = load_raw_data(data_path=data_path, settings=settings)
    X_train, X_test, y_train_raw, y_test_raw = split_data(df, settings=settings)
    y_train = encode_target(y_train_raw, settings=settings)
    y_test = encode_target(y_test_raw, settings=settings)

    # 2. Fit Preprocessor strictly on X_train
    print("[Step 2/3] Fitting preprocessing pipeline on training split (no leakage)...")
    preprocessor = build_preprocessor(settings=settings)
    fitted_preprocessor, X_train_trans = fit_and_transform(preprocessor, X_train)
    X_test_trans = transform_data(fitted_preprocessor, X_test)

    # Save preprocessor artifact
    prep_path = settings.resolve_path(settings.models_dir / "preprocessor.joblib")
    save_preprocessor(fitted_preprocessor, prep_path, settings=settings)

    # 3. Model Training & Logging Loop
    configs = get_model_configurations(random_state=settings.random_state)
    target_models = model_names or list(configs.keys())

    print(f"[Step 3/3] Training and logging {len(target_models)} model configurations...")
    results: list[dict[str, Any]] = []

    for name in target_models:
        if name not in configs:
            raise KeyError(f"Unknown model configuration '{name}'. Available: {list(configs.keys())}")
        config = configs[name]
        res = train_and_log_model(
            model_config=config,
            X_train=X_train_trans,
            y_train=y_train,
            X_test=X_test_trans,
            y_test=y_test,
            preprocessor_path=prep_path,
            settings=settings,
        )
        results.append(res)

    # 4. Create Side-by-Side Comparison Table
    comparison_rows = []
    for r in results:
        m = r["metrics"]
        comparison_rows.append({
            "Run ID": r["run_id"],
            "Model Name": r["run_name"],
            "Model Type": r["model_type"],
            "Accuracy": round(m.get("accuracy", 0.0), 4),
            "Precision": round(m.get("precision", 0.0), 4),
            "Recall": round(m.get("recall", 0.0), 4),
            "F1-Score": round(m.get("f1", 0.0), 4),
            "ROC-AUC": round(m.get("roc_auc", 0.0), 4),
        })

    comparison_df = pd.DataFrame(comparison_rows)

    # Save comparison reports
    reports_dir = settings.resolve_path(settings.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    comparison_json_path = reports_dir / "experiment_comparison.json"
    comparison_md_path = reports_dir / "experiment_comparison.md"

    with open(comparison_json_path, "w", encoding="utf-8") as f:
        json.dump(comparison_rows, f, indent=2)

    with open(comparison_md_path, "w", encoding="utf-8") as f:
        f.write("# MLflow Experiment Run Comparison\n\n")
        f.write(f"**Experiment Name**: `{settings.mlflow_experiment_name}`\n\n")
        f.write(comparison_df.to_markdown(index=False))
        f.write("\n")

    print("\n" + "=" * 75)
    print("ALL RUNS COMPLETED — COMPARISON SUMMARY")
    print("=" * 75)
    print(comparison_df.to_string(index=False))
    print("\nSaved comparison report to:")
    print(f" - JSON: {comparison_json_path}")
    print(f" - Markdown: {comparison_md_path}")
    print("=" * 75)

    return comparison_df


def main() -> None:
    """CLI entrypoint for running MLflow training experiments."""
    parser = argparse.ArgumentParser(description="Execute MLflow Model Training Experiments")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Specify model configuration names to train (e.g. logistic_regression_balanced random_forest_balanced)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to the Telco Customer Churn CSV dataset",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available model configurations",
    )

    args = parser.parse_args()

    if args.list_models:
        configs = get_model_configurations()
        print("\nAvailable Model Configurations:")
        for k, v in configs.items():
            print(f" - {k:<30} ({v.model_type}): {v.description}")
        return

    try:
        run_experiments(model_names=args.models, data_path=args.data_path)
    except Exception as e:
        print(f"Error executing experiment: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
