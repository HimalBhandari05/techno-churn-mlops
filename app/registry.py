"""MLflow Model Registry integration, model registration, stage transitions, and loader."""

from typing import Any

import mlflow
import mlflow.sklearn
from mlflow.entities.model_registry import ModelVersion
from mlflow.tracking import MlflowClient

from app.config import Settings, get_settings
from app.preprocessing import load_preprocessor


def get_mlflow_client(settings: Settings | None = None) -> MlflowClient:
    """Return configured MlflowClient instance."""
    settings = settings or get_settings()
    return MlflowClient(tracking_uri=settings.mlflow_tracking_uri)


def register_model_from_run(
    run_id: str,
    model_name: str = "telco-churn-classifier",
    artifact_path: str = "sklearn_model",
    description: str | None = None,
    tags: dict[str, str] | None = None,
    settings: Settings | None = None,
) -> ModelVersion:
    """Register a trained model from an MLflow run into the Model Registry.

    Args:
        run_id: Source MLflow run ID.
        model_name: Target registered model name.
        artifact_path: Subpath of the model artifact inside the run.
        description: Description of the registered model version.
        tags: Optional metadata tags to attach to the model version.
        settings: Settings instance.

    Returns:
        Registered ModelVersion entity.
    """
    settings = settings or get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = get_mlflow_client(settings=settings)

    model_uri = f"runs:/{run_id}/{artifact_path}"
    model_version = mlflow.register_model(model_uri=model_uri, name=model_name)

    # Set description and tags if provided
    if description:
        client.update_model_version(
            name=model_name,
            version=model_version.version,
            description=description,
        )

    if tags:
        for k, v in tags.items():
            client.set_model_version_tag(
                name=model_name,
                version=model_version.version,
                key=k,
                value=str(v),
            )

    return client.get_model_version(name=model_name, version=model_version.version)


def transition_model_stage(
    model_name: str,
    version: str | int,
    stage: str,
    archive_existing: bool = True,
    settings: Settings | None = None,
) -> ModelVersion:
    """Transition a registered model version to a target stage (and set alias).

    Args:
        model_name: Registered model name.
        version: Model version number.
        stage: Target lifecycle stage (e.g. 'Staging', 'Production', 'Archived').
        archive_existing: Whether to archive existing versions in target stage.
        settings: Settings instance.

    Returns:
        Updated ModelVersion entity.
    """
    settings = settings or get_settings()
    client = get_mlflow_client(settings=settings)
    ver_str = str(version)

    # 1. Transition legacy stage
    client.transition_model_version_stage(
        name=model_name,
        version=ver_str,
        stage=stage,
        archive_existing_versions=archive_existing,
    )

    # 2. Also set matching alias for MLflow 3.x deployment compatibility
    alias_name = stage.lower().replace(" ", "_")
    try:
        client.set_registered_model_alias(
            name=model_name,
            alias=alias_name,
            version=ver_str,
        )
    except Exception:
        pass

    return client.get_model_version(name=model_name, version=ver_str)


def get_registered_model_metadata(
    model_name: str,
    version: str | int | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Retrieve metadata and lifecycle state of a registered model.

    Args:
        model_name: Registered model name.
        version: Specific version (defaults to latest production version).
        settings: Settings instance.

    Returns:
        Dictionary containing model metadata.
    """
    settings = settings or get_settings()
    client = get_mlflow_client(settings=settings)

    if version is not None:
        mv = client.get_model_version(name=model_name, version=str(version))
    else:
        # Get production version or latest version
        versions = client.search_model_versions(f"name='{model_name}'")
        prod_versions = [v for v in versions if v.current_stage == "Production"]
        if prod_versions:
            mv = sorted(prod_versions, key=lambda x: int(x.version), reverse=True)[0]
        elif versions:
            mv = sorted(versions, key=lambda x: int(x.version), reverse=True)[0]
        else:
            raise ValueError(f"No versions found for registered model '{model_name}'")

    return {
        "name": mv.name,
        "version": mv.version,
        "current_stage": mv.current_stage,
        "source_run_id": mv.run_id,
        "source_uri": mv.source,
        "status": mv.status,
        "description": mv.description or "",
        "aliases": list(mv.aliases) if hasattr(mv, "aliases") and mv.aliases else [],
        "tags": dict(mv.tags) if hasattr(mv, "tags") and mv.tags else {},
        "creation_timestamp": mv.creation_timestamp,
    }


def load_production_model(
    model_name: str = "telco-churn-classifier",
    stage: str = "Production",
    settings: Settings | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    """Load the registered production model and its matching preprocessor.

    Args:
        model_name: Registered model name.
        stage: Model stage to load from (default: 'Production').
        settings: Settings instance.

    Returns:
        Tuple of (model, preprocessor, metadata).

    Raises:
        ValueError: If no model version in the specified stage is found.
    """
    settings = settings or get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = get_mlflow_client(settings=settings)

    versions = client.search_model_versions(f"name='{model_name}'")
    stage_versions = [v for v in versions if v.current_stage.lower() == stage.lower()]

    if not stage_versions:
        # Fallback to latest version if no explicit stage match
        if versions:
            latest = sorted(versions, key=lambda x: int(x.version), reverse=True)[0]
            mv = latest
        else:
            raise ValueError(f"No registered model version found for '{model_name}'")
    else:
        mv = sorted(stage_versions, key=lambda x: int(x.version), reverse=True)[0]

    # Load model from MLflow
    model_uri = f"models:/{model_name}/{mv.version}"
    try:
        model = mlflow.sklearn.load_model(model_uri)
    except Exception:
        # Direct run source fallback
        model = mlflow.sklearn.load_model(f"runs:/{mv.run_id}/sklearn_model")

    # Load matching preprocessor artifact
    prep_path = settings.resolve_path(settings.models_dir / "preprocessor.joblib")
    preprocessor = load_preprocessor(prep_path, settings=settings)

    metadata = {
        "model_name": mv.name,
        "model_version": mv.version,
        "stage": mv.current_stage,
        "source_run_id": mv.run_id,
        "model_type": type(model).__name__,
    }

    return model, preprocessor, metadata
