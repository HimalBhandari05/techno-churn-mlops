"""Tests for data loading, cleaning, validation, and splitting."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.config import get_settings
from app.data import (
    clean_raw_dataframe,
    get_dataset_summary,
    load_raw_data,
    split_data,
)


@pytest.fixture
def settings():
    """Return application settings."""
    return get_settings()


@pytest.fixture
def raw_dataframe(settings):
    """Load raw dataset DataFrame."""
    return load_raw_data(settings=settings)


def test_load_raw_data_success(raw_dataframe, settings):
    """Test that dataset loads correctly with expected row/col count."""
    assert isinstance(raw_dataframe, pd.DataFrame)
    assert len(raw_dataframe) == 7043
    assert raw_dataframe.shape[1] == 21
    assert settings.target_column in raw_dataframe.columns
    assert settings.id_column in raw_dataframe.columns


def test_load_raw_data_file_not_found(settings):
    """Test that FileNotFoundError is raised when file does not exist."""
    with pytest.raises(FileNotFoundError):
        load_raw_data(data_path=Path("non_existent_file.csv"), settings=settings)


def test_clean_raw_dataframe_handles_total_charges():
    """Test that whitespace strings in TotalCharges are coerced to numeric floats."""
    sample_df = pd.DataFrame({
        "customerID": ["1", "2", "3"],
        "tenure": [0, 5, 0],
        "MonthlyCharges": [20.0, 50.0, 30.0],
        "TotalCharges": [" ", "250.0", " "],
        "Churn": ["No", "No", "Yes"],
    })
    cleaned = clean_raw_dataframe(sample_df)

    assert pd.api.types.is_float_dtype(cleaned["TotalCharges"])
    assert cleaned["TotalCharges"].iloc[0] == 0.0
    assert cleaned["TotalCharges"].iloc[1] == 250.0
    assert cleaned["TotalCharges"].iloc[2] == 0.0
    assert not cleaned["TotalCharges"].isna().any()


def test_target_existence_and_values(raw_dataframe, settings):
    """Test that target column exists and contains valid binary classes."""
    assert settings.target_column in raw_dataframe.columns
    unique_targets = set(raw_dataframe[settings.target_column].unique())
    assert unique_targets == {"No", "Yes"}


def test_split_data_shapes_and_stratification(raw_dataframe, settings):
    """Test that split_data produces deterministic, correctly sized, stratified splits."""
    test_size = 0.2
    X_train, X_test, y_train, y_test = split_data(
        raw_dataframe,
        test_size=test_size,
        random_state=42,
        stratify=True,
        drop_id=True,
        settings=settings,
    )

    expected_test_rows = int(np.round(len(raw_dataframe) * test_size))
    expected_train_rows = len(raw_dataframe) - expected_test_rows

    assert len(X_train) == expected_train_rows
    assert len(X_test) == expected_test_rows
    assert len(y_train) == expected_train_rows
    assert len(y_test) == expected_test_rows

    # Verify ID column is dropped from features
    assert settings.id_column not in X_train.columns
    assert settings.id_column not in X_test.columns

    # Verify target column is dropped from features
    assert settings.target_column not in X_train.columns
    assert settings.target_column not in X_test.columns

    # Verify class stratification ratio
    train_pos_ratio = (y_train == "Yes").mean()
    test_pos_ratio = (y_test == "Yes").mean()
    overall_pos_ratio = (raw_dataframe[settings.target_column] == "Yes").mean()

    assert np.isclose(train_pos_ratio, overall_pos_ratio, atol=0.005)
    assert np.isclose(test_pos_ratio, overall_pos_ratio, atol=0.005)


def test_split_data_reproducibility(raw_dataframe, settings):
    """Test that identical seeds produce identical splits."""
    X_train1, X_test1, y_train1, y_test1 = split_data(
        raw_dataframe, random_state=42, settings=settings
    )
    X_train2, X_test2, y_train2, y_test2 = split_data(
        raw_dataframe, random_state=42, settings=settings
    )

    pd.testing.assert_frame_equal(X_train1, X_train2)
    pd.testing.assert_frame_equal(X_test1, X_test2)
    pd.testing.assert_series_equal(y_train1, y_train2)
    pd.testing.assert_series_equal(y_test1, y_test2)


def test_get_dataset_summary(raw_dataframe, settings):
    """Test dataset summary dictionary computation."""
    summary = get_dataset_summary(raw_dataframe, settings=settings)

    assert summary["num_rows"] == 7043
    assert summary["num_columns"] == 21
    assert summary["duplicate_rows"] == 0
    assert "target_distribution" in summary
    assert summary["target_distribution"]["counts"]["No"] == 5174
    assert summary["target_distribution"]["counts"]["Yes"] == 1869
