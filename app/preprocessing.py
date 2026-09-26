"""Preprocessing pipeline for Telco Customer Churn data."""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.config import Settings, get_settings


def build_preprocessor(
    numerical_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
    settings: Settings | None = None,
) -> ColumnTransformer:
    """Build an scikit-learn ColumnTransformer for numerical and categorical preprocessing.

    - Numerical pipeline: SimpleImputer(strategy='median') -> StandardScaler()
    - Categorical pipeline: SimpleImputer(strategy='most_frequent') -> OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    Args:
        numerical_features: List of numerical column names.
        categorical_features: List of categorical column names.
        settings: Settings instance.

    Returns:
        Configured, unfitted ColumnTransformer instance.
    """
    settings = settings or get_settings()
    num_cols = (
        numerical_features
        if numerical_features is not None
        else settings.numerical_features
    )
    cat_cols = (
        categorical_features
        if categorical_features is not None
        else settings.categorical_features
    )

    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def encode_target(
    y: pd.Series | np.ndarray | list[Any],
    settings: Settings | None = None,
) -> np.ndarray:
    """Encode target labels to binary integers (0 and 1).

    Args:
        y: Target series or array (e.g. ['No', 'Yes', ...]).
        settings: Settings instance.

    Returns:
        1D numpy array of 0 and 1 integer values.

    Raises:
        ValueError: If unsupported target values are encountered.
    """
    settings = settings or get_settings()
    mapping = settings.target_mapping

    if isinstance(y, pd.Series):
        # Check if already numeric
        if pd.api.types.is_numeric_dtype(y):
            return y.astype(int).to_numpy()
        # Clean / strip whitespace strings
        y_cleaned = y.astype(str).str.strip()
        encoded = y_cleaned.map(mapping)
        if encoded.isna().any():
            unmapped = y[encoded.isna()].unique()
            raise ValueError(f"Unmapped target values found: {unmapped}")
        return encoded.to_numpy(dtype=int)

    # If numpy array or list
    arr = np.asarray(y)
    if np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.floating):
        return arr.astype(int)

    encoded_list = []
    for item in arr:
        val = str(item).strip()
        if val in mapping:
            encoded_list.append(mapping[val])
        else:
            raise ValueError(f"Unmapped target value encountered: {item}")

    return np.array(encoded_list, dtype=int)


def decode_target(
    y_encoded: np.ndarray | list[int] | pd.Series,
    settings: Settings | None = None,
) -> np.ndarray:
    """Decode binary integers back to original labels ('No', 'Yes').

    Args:
        y_encoded: 1D array of 0s and 1s.
        settings: Settings instance.

    Returns:
        1D numpy array of label strings.
    """
    settings = settings or get_settings()
    inv_map = settings.inverse_target_mapping
    arr = np.asarray(y_encoded)
    return np.array([inv_map[int(val)] for val in arr], dtype=object)


def fit_preprocessor(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
) -> ColumnTransformer:
    """Fit preprocessor exclusively on training feature matrix to avoid leakage.

    Args:
        preprocessor: Unfitted ColumnTransformer.
        X_train: Training feature DataFrame.

    Returns:
        Fitted ColumnTransformer.
    """
    return preprocessor.fit(X_train)


def transform_data(
    preprocessor: ColumnTransformer,
    X: pd.DataFrame,
    as_dataframe: bool = False,
) -> np.ndarray | pd.DataFrame:
    """Transform feature matrix using a pre-fitted ColumnTransformer.

    Args:
        preprocessor: Fitted ColumnTransformer.
        X: Feature DataFrame to transform.
        as_dataframe: If True, returns transformed data as a pandas DataFrame with feature names.

    Returns:
        Numpy ndarray or DataFrame with transformed features.
    """
    transformed = preprocessor.transform(X)

    if as_dataframe:
        feature_names = get_transformed_feature_names(preprocessor)
        return pd.DataFrame(transformed, columns=feature_names, index=X.index)

    return transformed


def fit_and_transform(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    as_dataframe: bool = False,
) -> tuple[ColumnTransformer, np.ndarray | pd.DataFrame]:
    """Fit preprocessor on X_train and transform X_train.

    Args:
        preprocessor: Unfitted ColumnTransformer.
        X_train: Training feature DataFrame.
        as_dataframe: If True, returns transformed features as a pandas DataFrame.

    Returns:
        Tuple of (fitted_preprocessor, transformed_X_train).
    """
    fitted = fit_preprocessor(preprocessor, X_train)
    transformed = transform_data(fitted, X_train, as_dataframe=as_dataframe)
    return fitted, transformed


def get_transformed_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Retrieve output feature names from a fitted ColumnTransformer.

    Args:
        preprocessor: Fitted ColumnTransformer.

    Returns:
        List of generated feature names.
    """
    if hasattr(preprocessor, "get_feature_names_out"):
        return list(preprocessor.get_feature_names_out())
    raise AttributeError("Preprocessor has not been fitted or does not support get_feature_names_out.")


def save_preprocessor(
    preprocessor: ColumnTransformer,
    filepath: Path | str,
    settings: Settings | None = None,
) -> Path:
    """Serialize fitted preprocessor to disk using joblib.

    Args:
        preprocessor: Fitted ColumnTransformer.
        filepath: Destination file path.
        settings: Settings instance.

    Returns:
        Path to saved file.
    """
    settings = settings or get_settings()
    out_path = settings.resolve_path(filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, out_path)
    return out_path


def load_preprocessor(
    filepath: Path | str,
    settings: Settings | None = None,
) -> ColumnTransformer:
    """Deserialize fitted preprocessor from disk.

    Args:
        filepath: Path to saved preprocessor joblib file.
        settings: Settings instance.

    Returns:
        Fitted ColumnTransformer.

    Raises:
        FileNotFoundError: If the preprocessor file does not exist.
    """
    settings = settings or get_settings()
    in_path = settings.resolve_path(filepath)
    if not in_path.exists():
        raise FileNotFoundError(f"Preprocessor file not found at: {in_path}")
    return joblib.load(in_path)
