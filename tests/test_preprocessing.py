"""Tests for preprocessing pipeline, target encoding, serialization, and leakage prevention."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from app.config import get_settings
from app.data import load_raw_data, split_data
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


@pytest.fixture
def settings():
    """Return application settings."""
    return get_settings()


@pytest.fixture
def dataset_splits(settings):
    """Return stratified train and test splits."""
    df = load_raw_data(settings=settings)
    return split_data(df, settings=settings)


def test_encode_target_series(settings):
    """Test target encoding from pandas Series."""
    s = pd.Series(["No", "Yes", "No", "Yes"])
    encoded = encode_target(s, settings=settings)
    expected = np.array([0, 1, 0, 1])
    np.testing.assert_array_equal(encoded, expected)


def test_encode_target_invalid_value(settings):
    """Test that unknown target values raise ValueError."""
    with pytest.raises(ValueError):
        encode_target(["No", "Unknown", "Yes"], settings=settings)


def test_decode_target(settings):
    """Test target decoding from binary integers back to strings."""
    encoded = np.array([0, 1, 1, 0])
    decoded = decode_target(encoded, settings=settings)
    expected = np.array(["No", "Yes", "Yes", "No"], dtype=object)
    np.testing.assert_array_equal(decoded, expected)


def test_build_preprocessor(settings):
    """Test preprocessor construction and transformer names."""
    preprocessor = build_preprocessor(settings=settings)
    assert isinstance(preprocessor, ColumnTransformer)
    transformer_names = [name for name, _, _ in preprocessor.transformers]
    assert "num" in transformer_names
    assert "cat" in transformer_names


def test_fit_and_transform_pipeline(dataset_splits, settings):
    """Test fitting and transforming training data."""
    X_train, X_test, _, _ = dataset_splits
    preprocessor = build_preprocessor(settings=settings)

    fitted_prep, X_train_trans = fit_and_transform(preprocessor, X_train)

    assert isinstance(X_train_trans, np.ndarray)
    assert X_train_trans.shape[0] == len(X_train)
    # 3 numerical features + 43 one-hot encoded categories = 46 features
    assert X_train_trans.shape[1] == 46
    assert not np.isnan(X_train_trans).any()

    # Transform test set with fitted preprocessor
    X_test_trans = transform_data(fitted_prep, X_test)
    assert isinstance(X_test_trans, np.ndarray)
    assert X_test_trans.shape[0] == len(X_test)
    assert X_test_trans.shape[1] == 46
    assert not np.isnan(X_test_trans).any()


def test_transform_data_as_dataframe(dataset_splits, settings):
    """Test transforming data to pandas DataFrame with named columns."""
    X_train, _, _, _ = dataset_splits
    preprocessor = build_preprocessor(settings=settings)
    fitted_prep, df_trans = fit_and_transform(preprocessor, X_train, as_dataframe=True)

    assert isinstance(df_trans, pd.DataFrame)
    assert df_trans.shape == (len(X_train), 46)
    feature_names = get_transformed_feature_names(fitted_prep)
    assert list(df_trans.columns) == feature_names
    assert all(isinstance(col, str) for col in df_trans.columns)


def test_no_data_leakage(dataset_splits, settings):
    """Test that fitting preprocessor on train set does not leak test set statistics."""
    X_train, X_test, _, _ = dataset_splits

    preprocessor = build_preprocessor(settings=settings)
    fitted_prep = fit_preprocessor(preprocessor, X_train)

    # Extract fitted scaler from numerical pipeline
    num_pipeline = fitted_prep.named_transformers_["num"]
    scaler = num_pipeline.named_steps["scaler"]

    # Calculate expected means strictly on X_train for numerical features
    expected_means = X_train[settings.numerical_features].mean().to_numpy()
    fitted_means = scaler.mean_

    # Assert scaler mean equals train mean, NOT combined train+test mean
    np.testing.assert_allclose(fitted_means, expected_means, rtol=1e-3)

    # Combined mean would be different
    combined_df = pd.concat([X_train, X_test])
    combined_means = combined_df[settings.numerical_features].mean().to_numpy()
    assert not np.allclose(fitted_means, combined_means)


def test_handling_unseen_categories(dataset_splits, settings):
    """Test that preprocessor gracefully handles previously unseen categorical values."""
    X_train, X_test, _, _ = dataset_splits

    preprocessor = build_preprocessor(settings=settings)
    fitted_prep = fit_preprocessor(preprocessor, X_train)

    # Create unseen categories in test set
    X_test_unseen = X_test.copy()
    X_test_unseen.loc[X_test_unseen.index[0], "PaymentMethod"] = "Crypto/Bitcoin"
    X_test_unseen.loc[X_test_unseen.index[0], "Contract"] = "Decade-to-Decade"

    # Should transform without raising exceptions due to handle_unknown='ignore'
    transformed = transform_data(fitted_prep, X_test_unseen)
    assert transformed.shape == (len(X_test_unseen), 46)
    assert not np.isnan(transformed).any()


def test_save_and_load_preprocessor(dataset_splits, tmp_path, settings):
    """Test serialization and deserialization of fitted preprocessor."""
    X_train, X_test, _, _ = dataset_splits

    preprocessor = build_preprocessor(settings=settings)
    fitted_prep = fit_preprocessor(preprocessor, X_train)
    expected_output = transform_data(fitted_prep, X_test)

    # Save to temporary path
    save_path = tmp_path / "preprocessor.joblib"
    save_preprocessor(fitted_prep, save_path, settings=settings)
    assert save_path.exists()

    # Load and test equivalence
    loaded_prep = load_preprocessor(save_path, settings=settings)
    loaded_output = transform_data(loaded_prep, X_test)

    np.testing.assert_allclose(expected_output, loaded_output)
