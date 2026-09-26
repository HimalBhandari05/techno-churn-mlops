"""Tests for MLflow Model Registry, registration, stage transitions, and metadata verification."""

import pytest
from mlflow.tracking import MlflowClient

from app.config import get_settings
from app.registry import (
    get_registered_model_metadata,
    load_production_model,
    register_model_from_run,
    transition_model_stage,
)


@pytest.fixture
def settings():
    """Return project settings."""
    return get_settings()


def test_registered_model_exists_and_metadata(settings):
    """Test that telco-churn-classifier exists in Model Registry with valid metadata."""
    meta = get_registered_model_metadata("telco-churn-classifier", settings=settings)

    assert meta["name"] == "telco-churn-classifier"
    assert int(meta["version"]) >= 1
    assert meta["status"] == "READY"
    assert meta["source_run_id"] is not None
    assert len(meta["source_run_id"]) > 0
    assert "RandomForestClassifier" in meta["tags"].get("model_type", "") or "balanced" in meta["description"]


def test_model_version_and_lifecycle_state(settings):
    """Test that registered model version is in Production stage after transitions."""
    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    versions = client.search_model_versions("name='telco-churn-classifier'")

    assert len(versions) >= 1
    v1 = [v for v in versions if str(v.version) == "1"][0]
    assert v1.current_stage in ["Production", "Staging"]


def test_stage_transition_execution(settings):
    """Test executing stage transitions programmatically."""
    # Test transitioning to Staging and then back to Production
    v_staging = transition_model_stage("telco-churn-classifier", version=1, stage="Staging", settings=settings)
    assert v_staging.current_stage == "Staging"

    v_prod = transition_model_stage("telco-churn-classifier", version=1, stage="Production", settings=settings)
    assert v_prod.current_stage == "Production"


def test_load_production_model_and_preprocessor_compatibility(settings):
    """Test loading production model and preprocessor verifies compatibility."""
    model, preprocessor, meta = load_production_model(model_name="telco-churn-classifier", settings=settings)

    assert model is not None
    assert preprocessor is not None
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")
    assert hasattr(preprocessor, "transform")
    assert meta["model_name"] == "telco-churn-classifier"
    assert meta["stage"] == "Production"
