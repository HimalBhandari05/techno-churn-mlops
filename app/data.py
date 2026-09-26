"""Data loading, validation, cleaning, and train/test splitting module."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from app.config import Settings, get_settings


def clean_raw_dataframe(df: pd.DataFrame, settings: Settings | None = None) -> pd.DataFrame:
    """Clean raw dataframe: handle blank TotalCharges and strip string whitespaces.

    Args:
        df: Raw pandas DataFrame.
        settings: Settings instance (defaults to get_settings()).

    Returns:
        Cleaned DataFrame with TotalCharges as float64 and stripped strings.
    """
    settings = settings or get_settings()
    df_clean = df.copy()

    # Strip whitespace from object/string columns
    string_cols = df_clean.select_dtypes(include=["object", "string"]).columns
    for col in string_cols:
        df_clean[col] = df_clean[col].astype(str).str.strip()

    # TotalCharges coercion (convert ' ' and invalid strings to NaN, then fill)
    if "TotalCharges" in df_clean.columns:
        df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"], errors="coerce")
        # For customers with 0 tenure, total charges accrued is logically 0.0
        zero_tenure_mask = (df_clean["tenure"] == 0) & (df_clean["TotalCharges"].isna())
        df_clean.loc[zero_tenure_mask, "TotalCharges"] = 0.0
        # If any other TotalCharges are still NaN, impute with 0.0 or column median
        if df_clean["TotalCharges"].isna().any():
            median_val = df_clean["TotalCharges"].median()
            df_clean["TotalCharges"] = df_clean["TotalCharges"].fillna(
                median_val if not np.isnan(median_val) else 0.0
            )

    return df_clean


def load_raw_data(
    data_path: Path | str | None = None,
    settings: Settings | None = None,
) -> pd.DataFrame:
    """Load the raw Telco Customer Churn dataset from disk and apply basic cleaning.

    Args:
        data_path: Path to dataset CSV. If None, uses path from Settings.
        settings: Settings instance.

    Returns:
        Cleaned pandas DataFrame.

    Raises:
        FileNotFoundError: If the data file does not exist.
        ValueError: If required columns are missing from the loaded dataset.
    """
    settings = settings or get_settings()
    resolved_path = (
        settings.resolve_path(data_path)
        if data_path is not None
        else settings.resolve_path(settings.data_path)
    )

    if not resolved_path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {resolved_path}")

    df = pd.read_csv(resolved_path)

    # Validate target column presence
    if settings.target_column not in df.columns:
        raise ValueError(
            f"Target column '{settings.target_column}' not found in dataset columns: {list(df.columns)}"
        )

    return clean_raw_dataframe(df, settings=settings)


def split_data(
    df: pd.DataFrame,
    target_column: str | None = None,
    test_size: float | None = None,
    random_state: int | None = None,
    stratify: bool | None = None,
    drop_id: bool = True,
    settings: Settings | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Deterministically split data into stratified train and test sets.

    Args:
        df: Input DataFrame.
        target_column: Target column name. Defaults to settings.target_column.
        test_size: Proportion of dataset for test split. Defaults to settings.test_size.
        random_state: Random seed for reproducibility. Defaults to settings.random_state.
        stratify: Whether to use stratified sampling based on the target. Defaults to settings.stratify.
        drop_id: Whether to drop the customerID column from feature matrices. Defaults to True.
        settings: Settings instance.

    Returns:
        Tuple containing (X_train, X_test, y_train, y_test).

    Raises:
        ValueError: If the target column is missing.
    """
    settings = settings or get_settings()
    target_col = target_column or settings.target_column
    t_size = test_size if test_size is not None else settings.test_size
    r_state = random_state if random_state is not None else settings.random_state
    use_stratify = stratify if stratify is not None else settings.stratify

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    y = df[target_col].copy()
    X = df.drop(columns=[target_col]).copy()

    if drop_id and settings.id_column in X.columns:
        X = X.drop(columns=[settings.id_column])

    stratify_target = y if use_stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=t_size,
        random_state=r_state,
        stratify=stratify_target,
    )

    return X_train, X_test, y_train, y_test


def get_dataset_summary(df: pd.DataFrame, settings: Settings | None = None) -> dict[str, Any]:
    """Compute summary statistics and metadata of the dataset.

    Args:
        df: Input DataFrame.
        settings: Settings instance.

    Returns:
        Dictionary with dataset metrics and distributions.
    """
    settings = settings or get_settings()
    target_col = settings.target_column

    summary: dict[str, Any] = {
        "num_rows": int(len(df)),
        "num_columns": int(df.shape[1]),
        "columns": list(df.columns),
        "missing_values": {k: int(v) for k, v in df.isna().sum().to_dict().items() if v > 0},
        "duplicate_rows": int(df.duplicated().sum()),
    }

    if target_col in df.columns:
        counts = df[target_col].value_counts().to_dict()
        proportions = (df[target_col].value_counts(normalize=True) * 100).round(2).to_dict()
        summary["target_distribution"] = {
            "counts": {str(k): int(v) for k, v in counts.items()},
            "percentages": {str(k): float(v) for k, v in proportions.items()},
        }

    return summary
