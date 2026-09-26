"""FastAPI model serving application for Telco Customer Churn predictions."""

from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.data import clean_raw_dataframe
from app.preprocessing import decode_target, transform_data
from app.registry import get_registered_model_metadata, load_production_model


class CustomerFeatures(BaseModel):
    """Schema for individual Telco Customer feature input."""

    customerID: str | None = Field(default=None, description="Customer unique ID")
    gender: str = Field(..., description="Gender ('Male', 'Female')")
    SeniorCitizen: int = Field(..., ge=0, le=1, description="Whether customer is a senior citizen (0 or 1)")
    Partner: str = Field(..., description="Whether customer has a partner ('Yes', 'No')")
    Dependents: str = Field(..., description="Whether customer has dependents ('Yes', 'No')")
    tenure: int = Field(..., ge=0, description="Number of months customer has stayed with company")
    PhoneService: str = Field(..., description="Whether customer has phone service ('Yes', 'No')")
    MultipleLines: str = Field(..., description="Multiple lines status ('No', 'Yes', 'No phone service')")
    InternetService: str = Field(..., description="Internet service provider ('DSL', 'Fiber optic', 'No')")
    OnlineSecurity: str = Field(..., description="Online security status ('No', 'Yes', 'No internet service')")
    OnlineBackup: str = Field(..., description="Online backup status ('Yes', 'No', 'No internet service')")
    DeviceProtection: str = Field(..., description="Device protection status ('No', 'Yes', 'No internet service')")
    TechSupport: str = Field(..., description="Tech support status ('No', 'Yes', 'No internet service')")
    StreamingTV: str = Field(..., description="Streaming TV status ('No', 'Yes', 'No internet service')")
    StreamingMovies: str = Field(..., description="Streaming movies status ('No', 'Yes', 'No internet service')")
    Contract: str = Field(..., description="Contract term ('Month-to-month', 'One year', 'Two year')")
    PaperlessBilling: str = Field(..., description="Paperless billing status ('Yes', 'No')")
    PaymentMethod: str = Field(..., description="Payment method name")
    MonthlyCharges: float = Field(..., ge=0.0, description="The amount charged to customer monthly")
    TotalCharges: float | str = Field(..., description="Total amount charged to customer (numeric or string)")

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


class PredictionResponse(BaseModel):
    """Schema for prediction response."""

    prediction: str = Field(..., description="Predicted Churn class ('Yes' or 'No')")
    churn_probability: float = Field(..., ge=0.0, le=1.0, description="Probability of churn (class 1)")
    model_name: str = Field(..., description="Registered model name")
    model_version: str = Field(..., description="Registered model version")
    stage: str = Field(..., description="Model stage")


class HealthResponse(BaseModel):
    """Schema for API health status check."""

    status: str = Field(..., description="API operational status")
    model_name: str = Field(..., description="Loaded registered model name")
    model_version: str = Field(..., description="Loaded registered model version")
    stage: str = Field(..., description="Current lifecycle stage")
    source_run_id: str = Field(..., description="Source MLflow run ID")
    model_type: str = Field(..., description="Model estimator type")


# App state container
model_store: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown lifecycle."""
    settings = get_settings()
    try:
        model, preprocessor, metadata = load_production_model(settings=settings)
        model_store["model"] = model
        model_store["preprocessor"] = preprocessor
        model_store["metadata"] = metadata
    except Exception as exc:
        print(f"Warning: Could not load production model on startup: {exc}")
    yield
    model_store.clear()


app = FastAPI(
    title="Telco Customer Churn Prediction Service",
    description="Production MLOps inference API serving the registered MLflow production model.",
    version="1.0.0",
    lifespan=lifespan,
)


def get_model_and_preprocessor():
    """Retrieve model and preprocessor, loading if not already in store."""
    if "model" not in model_store or "preprocessor" not in model_store:
        settings = get_settings()
        model, preprocessor, metadata = load_production_model(settings=settings)
        model_store["model"] = model
        model_store["preprocessor"] = preprocessor
        model_store["metadata"] = metadata
    return model_store["model"], model_store["preprocessor"], model_store["metadata"]


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check() -> HealthResponse:
    """Check API service health and metadata of currently loaded model."""
    try:
        _, _, metadata = get_model_and_preprocessor()
        return HealthResponse(
            status="healthy",
            model_name=metadata["model_name"],
            model_version=str(metadata["model_version"]),
            stage=metadata["stage"],
            source_run_id=metadata["source_run_id"],
            model_type=metadata["model_type"],
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model serving unhealthy: {exc}",
        )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_churn(customer: CustomerFeatures) -> PredictionResponse:
    """Predict customer churn and probability for a single customer payload.

    Follows exact Phase 1 preprocessing pipeline:
    1. Converts Pydantic model to DataFrame.
    2. Runs clean_raw_dataframe (handles TotalCharges float coercion & string stripping).
    3. Transforms features with fitted ColumnTransformer.
    4. Evaluates prediction and probability with registered Production model.
    """
    try:
        model, preprocessor, metadata = get_model_and_preprocessor()

        # Convert to single-row DataFrame
        input_dict = customer.model_dump()
        df_raw = pd.DataFrame([input_dict])

        # Step 1: Phase 1 Data Cleaning & Type Coercion
        df_clean = clean_raw_dataframe(df_raw)

        # Drop ID if present in features
        if "customerID" in df_clean.columns:
            df_features = df_clean.drop(columns=["customerID"])
        else:
            df_features = df_clean

        # Step 2: Phase 1 ColumnTransformer Preprocessing
        X_trans = transform_data(preprocessor, df_features)

        # Step 3: Model Inference
        raw_pred = model.predict(X_trans)[0]
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(X_trans)[0, 1])
        else:
            proba = float(raw_pred)

        pred_label = str(decode_target([raw_pred])[0])

        return PredictionResponse(
            prediction=pred_label,
            churn_probability=round(proba, 4),
            model_name=metadata["model_name"],
            model_version=str(metadata["model_version"]),
            stage=metadata["stage"],
        )

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {exc}",
        )
