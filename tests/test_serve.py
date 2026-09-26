"""Tests for FastAPI model serving API endpoints, schemas, validation, and inference."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.data import clean_raw_dataframe
from app.preprocessing import decode_target, transform_data
from app.registry import load_production_model
from app.serve import app


@pytest.fixture
def client():
    """FastAPI TestClient fixture."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def valid_customer_payload():
    """Valid Telco customer record dictionary."""
    return {
        "customerID": "7590-VHVEG",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85,
    }


def test_health_endpoint(client):
    """Test GET /health returns 200 and valid service metadata."""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_name"] == "telco-churn-classifier"
    assert "model_version" in data
    assert data["stage"] in ["Production", "Staging"]
    assert len(data["source_run_id"]) > 0
    assert "RandomForest" in data["model_type"]


def test_predict_valid_request(client, valid_customer_payload):
    """Test POST /predict with a valid payload returns 200 and schema compliant response."""
    response = client.post("/predict", json=valid_customer_payload)
    assert response.status_code == 200

    data = response.json()
    assert "prediction" in data
    assert data["prediction"] in ["Yes", "No"]
    assert "churn_probability" in data
    assert isinstance(data["churn_probability"], float)
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["model_name"] == "telco-churn-classifier"
    assert data["model_version"] == "1"


def test_predict_tenure_zero_with_space_total_charges(client, valid_customer_payload):
    """Test POST /predict gracefully handles tenure=0 new customer with string whitespace TotalCharges."""
    payload = valid_customer_payload.copy()
    payload["tenure"] = 0
    payload["TotalCharges"] = " "

    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["prediction"] in ["Yes", "No"]
    assert 0.0 <= data["churn_probability"] <= 1.0


def test_predict_malformed_incomplete_request(client):
    """Test POST /predict rejects incomplete payload with 422 Unprocessable Entity."""
    incomplete_payload = {
        "gender": "Female",
        "tenure": 5,
        # Missing other required features
    }
    response = client.post("/predict", json=incomplete_payload)
    assert response.status_code == 422


def test_predict_invalid_data_types(client, valid_customer_payload):
    """Test POST /predict rejects invalid data types with 422 Unprocessable Entity."""
    invalid_payload = valid_customer_payload.copy()
    invalid_payload["SeniorCitizen"] = 99  # Valid is 0 or 1
    invalid_payload["MonthlyCharges"] = -50.0  # Invalid negative charge

    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422


def test_serving_matches_direct_model_inference(client, valid_customer_payload):
    """Test that API predictions exactly match direct registered model predictions."""
    # 1. API prediction
    api_response = client.post("/predict", json=valid_customer_payload)
    assert api_response.status_code == 200
    api_data = api_response.json()

    # 2. Direct model prediction
    model, preprocessor, _ = load_production_model(model_name="telco-churn-classifier")
    df_raw = pd.DataFrame([valid_customer_payload])
    df_clean = clean_raw_dataframe(df_raw)
    if "customerID" in df_clean.columns:
        df_clean = df_clean.drop(columns=["customerID"])
    X_trans = transform_data(preprocessor, df_clean)

    direct_pred = str(decode_target(model.predict(X_trans))[0])
    direct_prob = float(model.predict_proba(X_trans)[0, 1])

    assert api_data["prediction"] == direct_pred
    assert np.isclose(api_data["churn_probability"], direct_prob, atol=1e-3)
