"""Configuration settings for Telco Customer Churn MLOps pipeline."""

from functools import lru_cache
from pathlib import Path
from typing import ClassVar

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application and pipeline configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Base paths
    project_root: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent
    )
    data_path: Path = Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    models_dir: Path = Path("models")
    reports_dir: Path = Path("reports")

    # Dataset schema
    id_column: str = "customerID"
    target_column: str = "Churn"

    numerical_features: list[str] = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
    ]

    categorical_features: list[str] = [
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
    ]

    # Data split parameters
    test_size: float = 0.2
    random_state: int = 42
    stratify: bool = True

    # Target mapping
    target_mapping: ClassVar[dict[str, int]] = {"No": 0, "Yes": 1}
    inverse_target_mapping: ClassVar[dict[int, str]] = {0: "No", 1: "Yes"}

    # MLflow Tracking (Phase 2+)
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment_name: str = "week17-track-a-telco-churn"

    @property
    def feature_columns(self) -> list[str]:
        """All predictive feature column names."""
        return self.numerical_features + self.categorical_features

    @property
    def all_columns(self) -> list[str]:
        """All dataset columns including identifier and target."""
        return [self.id_column] + self.feature_columns + [self.target_column]

    def resolve_path(self, path: Path | str) -> Path:
        """Resolve path relative to project root if not absolute."""
        p = Path(path)
        if p.is_absolute():
            return p
        return self.project_root / p


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached Settings instance."""
    return Settings()
