"""Model configuration definitions and factory methods for Telco Churn prediction."""

from dataclasses import dataclass, field
from typing import Any

from sklearn.base import BaseEstimator
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression


@dataclass
class ModelConfig:
    """Configuration specification for a machine learning model."""

    name: str
    model_type: str
    estimator_class: type[BaseEstimator]
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def build_estimator(self) -> BaseEstimator:
        """Instantiate estimator with configured hyperparameters."""
        return self.estimator_class(**self.hyperparameters)


def get_model_configurations(random_state: int = 42) -> dict[str, ModelConfig]:
    """Return dictionary of diverse model configurations for experiment tracking.

    Args:
        random_state: Random seed for reproducibility.

    Returns:
        Dictionary mapping configuration keys to ModelConfig objects.
    """
    return {
        "logistic_regression_balanced": ModelConfig(
            name="logistic_regression_balanced",
            model_type="LogisticRegression",
            estimator_class=LogisticRegression,
            hyperparameters={
                "C": 0.5,
                "solver": "lbfgs",
                "class_weight": "balanced",
                "max_iter": 1000,
                "random_state": random_state,
            },
            description="Linear baseline with L2 regularization and balanced class weighting",
        ),
        "random_forest_balanced": ModelConfig(
            name="random_forest_balanced",
            model_type="RandomForestClassifier",
            estimator_class=RandomForestClassifier,
            hyperparameters={
                "n_estimators": 200,
                "max_depth": 8,
                "min_samples_split": 10,
                "min_samples_leaf": 4,
                "max_features": "sqrt",
                "class_weight": "balanced",
                "random_state": random_state,
            },
            description="Ensemble bagging with regularized tree depth and balanced class weighting",
        ),
        "hist_gradient_boosting": ModelConfig(
            name="hist_gradient_boosting",
            model_type="HistGradientBoostingClassifier",
            estimator_class=HistGradientBoostingClassifier,
            hyperparameters={
                "learning_rate": 0.05,
                "max_iter": 150,
                "max_depth": 5,
                "min_samples_leaf": 20,
                "l2_regularization": 0.5,
                "class_weight": "balanced",
                "random_state": random_state,
            },
            description="Histogram gradient boosting with learning rate shrinkage and L2 regularization",
        ),
        "random_forest_unweighted": ModelConfig(
            name="random_forest_unweighted",
            model_type="RandomForestClassifier",
            estimator_class=RandomForestClassifier,
            hyperparameters={
                "n_estimators": 100,
                "max_depth": 12,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "max_features": "sqrt",
                "class_weight": None,
                "random_state": random_state,
            },
            description="Unweighted Random Forest baseline highlighting accuracy vs recall trade-offs",
        ),
    }
