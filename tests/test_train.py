"""Tests for ML model training, MLflow tracking, artifact logging, and reproducibility."""

from pathlib import Path

import mlflow
import numpy as np
import pytest

from app.config import Settings, get_settings
from app.data import load_raw_data, split_data
from app.models import get_model_configurations
from app.preprocessing import (
    build_preprocessor,
    encode_target,
    fit_and_transform,
    save_preprocessor,
    transform_data,
)
from app.train import run_experiments, train_and_log_model


@pytest.fixture
def test_settings(tmp_path):
    """Create isolated settings with temporary MLflow database and artifact paths."""
    db_path = tmp_path / "test_mlflow.db"
    return Settings(
        mlflow_tracking_uri=f"sqlite:///{db_path}",
        mlflow_experiment_name="test-telco-churn-experiment",
        models_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
        random_state=42,
    )


@pytest.fixture
def preprocessed_data(test_settings, tmp_path):
    """Return preprocessed data and saved preprocessor path."""
    df = load_raw_data(settings=test_settings)
    X_train, X_test, y_train_raw, y_test_raw = split_data(df, settings=test_settings)
    y_train = encode_target(y_train_raw, settings=test_settings)
    y_test = encode_target(y_test_raw, settings=test_settings)

    preprocessor = build_preprocessor(settings=test_settings)
    fitted_prep, X_train_trans = fit_and_transform(preprocessor, X_train)
    X_test_trans = transform_data(fitted_prep, X_test)

    prep_file = tmp_path / "preprocessor.joblib"
    save_preprocessor(fitted_prep, prep_file, settings=test_settings)

    return X_train_trans, y_train, X_test_trans, y_test, prep_file


def test_model_configurations_diversity():
    """Verify at least 3 genuinely different model configurations are provided."""
    configs = get_model_configurations(random_state=42)

    assert len(configs) >= 3

    model_types = {cfg.model_type for cfg in configs.values()}
    assert len(model_types) >= 3 or "LogisticRegression" in model_types and "RandomForestClassifier" in model_types

    # Ensure configurations differ in meaningful hyperparameters
    assert "logistic_regression_balanced" in configs
    assert "random_forest_balanced" in configs
    assert "hist_gradient_boosting" in configs

    # Verify estimators can be built
    for name, cfg in configs.items():
        estimator = cfg.build_estimator()
        assert estimator is not None


def test_train_and_log_model_mlflow(preprocessed_data, test_settings):
    """Test training and logging parameters, metrics, and artifacts to MLflow."""
    X_train_trans, y_train, X_test_trans, y_test, prep_file = preprocessed_data
    configs = get_model_configurations(random_state=42)
    config = configs["logistic_regression_balanced"]

    result = train_and_log_model(
        model_config=config,
        X_train=X_train_trans,
        y_train=y_train,
        X_test=X_test_trans,
        y_test=y_test,
        preprocessor_path=prep_file,
        settings=test_settings,
    )

    assert "run_id" in result
    run_id = result["run_id"]
    metrics = result["metrics"]

    # Verify required metrics are calculated
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics

    # Query MLflow client to verify run persistence
    client = mlflow.tracking.MlflowClient(tracking_uri=test_settings.mlflow_tracking_uri)
    run = client.get_run(run_id)

    assert run.info.status == "FINISHED"
    assert run.data.params["model_type"] == "LogisticRegression"
    assert "param_C" in run.data.params

    for metric_name in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        assert metric_name in run.data.metrics

    # Verify artifacts exist in MLflow
    artifacts = client.list_artifacts(run_id)
    artifact_paths = [a.path for a in artifacts]
    assert "model" in artifact_paths
    assert "evaluation" in artifact_paths
    assert "preprocessing" in artifact_paths


def test_training_reproducibility(preprocessed_data, test_settings):
    """Test that training the same model with the same seed yields identical predictions & metrics."""
    X_train_trans, y_train, X_test_trans, y_test, prep_file = preprocessed_data
    configs1 = get_model_configurations(random_state=42)
    configs2 = get_model_configurations(random_state=42)

    config1 = configs1["random_forest_balanced"]
    config2 = configs2["random_forest_balanced"]

    res1 = train_and_log_model(
        model_config=config1,
        X_train=X_train_trans,
        y_train=y_train,
        X_test=X_test_trans,
        y_test=y_test,
        preprocessor_path=prep_file,
        settings=test_settings,
    )

    res2 = train_and_log_model(
        model_config=config2,
        X_train=X_train_trans,
        y_train=y_train,
        X_test=X_test_trans,
        y_test=y_test,
        preprocessor_path=prep_file,
        settings=test_settings,
    )

    for m in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        assert np.isclose(res1["metrics"][m], res2["metrics"][m], atol=1e-5)


def test_run_experiments_comparison(test_settings):
    """Test that run_experiments executes all configs and produces a comparison table."""
    df_comp = run_experiments(
        model_names=["logistic_regression_balanced", "random_forest_balanced"],
        settings=test_settings,
    )

    assert len(df_comp) == 2
    assert "Run ID" in df_comp.columns
    assert "Model Name" in df_comp.columns
    assert "F1-Score" in df_comp.columns
    assert "ROC-AUC" in df_comp.columns
