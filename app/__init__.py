"""Telco Customer Churn MLOps Package."""

from app.config import Settings, get_settings
from app.data import get_dataset_summary, load_raw_data, split_data
from app.drift import (
    CustomDriftMetricResult,
    calculate_churn_rate_shift,
    generate_data_drift_report,
    generate_target_drift_report,
    inject_synthetic_drift,
    split_monitoring_data,
)
from app.evaluate import (
    calculate_classification_metrics,
    generate_and_save_artifacts,
    generate_classification_report,
    plot_confusion_matrix,
    plot_roc_curve,
)
from app.models import ModelConfig, get_model_configurations
from app.monitor import run_drift_monitoring
from app.preprocessing import (
    build_preprocessor,
    decode_target,
    encode_target,
    fit_and_transform,
    fit_preprocessor,
    get_transformed_feature_names,
    load_preprocessor,
    save_preprocessor,
    transform_data,
)
from app.registry import (
    get_mlflow_client,
    get_registered_model_metadata,
    load_production_model,
    register_model_from_run,
    transition_model_stage,
)
from app.serve import CustomerFeatures, HealthResponse, PredictionResponse, app
from app.train import run_experiments, train_and_log_model

__all__ = [
    "Settings",
    "get_settings",
    "load_raw_data",
    "split_data",
    "get_dataset_summary",
    "build_preprocessor",
    "encode_target",
    "decode_target",
    "fit_preprocessor",
    "transform_data",
    "fit_and_transform",
    "get_transformed_feature_names",
    "save_preprocessor",
    "load_preprocessor",
    "ModelConfig",
    "get_model_configurations",
    "calculate_classification_metrics",
    "plot_confusion_matrix",
    "plot_roc_curve",
    "generate_classification_report",
    "generate_and_save_artifacts",
    "train_and_log_model",
    "run_experiments",
    "get_mlflow_client",
    "register_model_from_run",
    "transition_model_stage",
    "get_registered_model_metadata",
    "load_production_model",
    "CustomerFeatures",
    "PredictionResponse",
    "HealthResponse",
    "app",
    "CustomDriftMetricResult",
    "split_monitoring_data",
    "inject_synthetic_drift",
    "calculate_churn_rate_shift",
    "generate_data_drift_report",
    "generate_target_drift_report",
    "run_drift_monitoring",
]
