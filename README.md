# Telco Customer Churn — Production MLOps Pipeline

Production-grade Machine Learning Operations (MLOps) project for predicting customer churn using the Telco Customer Churn dataset.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset](#2-dataset)
3. [Architecture](#3-architecture)
4. [Repository Structure](#4-repository-structure)
5. [Environment & Reproducibility](#5-environment--reproducibility)
6. [Data Pipeline](#6-data-pipeline)
7. [Model Training](#7-model-training)
8. [MLflow Experiment Tracking](#8-mlflow-experiment-tracking)
9. [Model Selection](#9-model-selection)
10. [MLflow Model Registry](#10-mlflow-model-registry)
11. [Model Serving](#11-model-serving)
12. [Evidently Drift Monitoring](#12-evidently-drift-monitoring)
13. [MLflow Monitoring](#13-mlflow-monitoring)
14. [Reproduction](#14-reproduction)
15. [Results](#15-results)
16. [Limitations](#16-limitations)
17. [Optional Airflow](#17-optional-airflow)

---

## 1. Project Overview

The objective of this project is to build an end-to-end, production-ready MLOps system that predicts customer churn for a telecommunications provider. The project follows standard MLOps engineering practices:

$$\text{Data Pipeline} \longrightarrow \text{Model Training} \longrightarrow \text{MLflow Tracking} \longrightarrow \text{Model Registry} \longrightarrow \text{Serving API} \longrightarrow \text{Evidently Drift Monitoring}$$

Key milestones implemented across the pipeline:
- **Reproducible Environment**: Declarative package management via `uv` with pinned dependencies in `uv.lock`.
- **Leak-Free Data Pipeline**: Deterministic stratified splitting and scikit-learn `ColumnTransformer` preprocessing fitted strictly on training data.
- **Multi-Model MLflow Experimentation**: Training 4 configurations across 3 model families (Linear, Bagging, Boosting) with complete metric and artifact logging.
- **Model Registry & Governance**: Registering the selected high-recall model and tracking lifecycle stage transitions (`None` $\rightarrow$ `Staging` $\rightarrow$ `Production`).
- **Production Serving**: High-throughput FastAPI inference service with Pydantic validation and consistent preprocessing transformation.
- **Data & Target Drift Monitoring**: Evidently AI reports and custom business metrics integrated with MLflow monitoring runs.

---

## 2. Dataset

- **Source File**: [`data/WA_Fn-UseC_-Telco-Customer-Churn.csv`](file:///home/himalbhandari/techno-churn-mlops/data/WA_Fn-UseC_-Telco-Customer-Churn.csv)
- **Total Records**: 7,043 rows
- **Total Columns**: 21 columns (1 Identifier, 19 Features, 1 Target)
- **Target Variable**: `Churn` (`"No"`, `"Yes"`)
  - `"No"`: 5,174 (73.46%)
  - `"Yes"`: 1,869 (26.54%)
  - Imbalance Ratio: ~2.77:1 (handled via stratified splitting, inverse class weighting, and recall-focused evaluation metrics)
- **Duplicate Rows**: 0
- **Missing Value Handling**:
  - `TotalCharges`: 11 rows contain whitespace strings (`" "`) corresponding to new customers with `tenure == 0`. These are coerced to `float64` and imputed with `0.0`.
  - All other columns have 0 missing values.
- **Feature Groups**:
  - **Identifier (1)**: `customerID` (excluded from feature matrices during training and inference)
  - **Numerical (3)**: `tenure`, `MonthlyCharges`, `TotalCharges`
  - **Categorical (16)**: `gender`, `SeniorCitizen`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`

---

## 3. Architecture

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                              MLOps ARCHITECTURE                             │
└─────────────────────────────────────────────────────────────────────────────┘

  Raw Telco CSV (7,043 rows)
              │
              ▼
  [Data Cleaning & Stratified Split] ─── 80% Train (5,634) / 20% Test (1,409)
              │
              ▼
  [ColumnTransformer Preprocessing] ─── Median Impute + Scaler (Num)
              │                         Mode Impute + OneHotEncoder (Cat)
              ▼
  [Multi-Model Experimentation] ─────── Logistic Regression (Balanced)
              │                         Random Forest (Balanced)
              │                         HistGradientBoosting (Balanced)
              │                         Random Forest (Unweighted)
              ▼
  [MLflow Tracking (mlflow.db)] ─────── Parameters, Metrics (Acc, Prec, Rec, F1, AUC),
              │                         Artifacts (Model, Confusion Matrix, ROC)
              ▼
  [Model Selection & Registry] ──────── Select RF Balanced (Run: ff3e86be...)
              │                         Register 'telco-churn-classifier' v1
              │                         Transition: Staging -> Production
              ▼
  [FastAPI Model Serving] ───────────── GET /health, POST /predict (Pydantic Validation)
              │
              ▼
  [Evidently AI Drift Monitoring] ───── Reference (70%) vs Current (30% + Synthetic Drift)
              │                         Data Drift (19 cols) + Target Drift (Churn)
              │                         Custom Metric: churn_rate_shift
              ▼
  [MLflow Monitoring Experiment] ────── week17-track-a-telco-churn-monitoring (Run: 1aac8ca6...)
                                        Logs HTML Reports & Drift Metrics
```

---

## 4. Repository Structure

```text
techno-churn-mlops/
├── app/
│   ├── __init__.py           # Package interface exports
│   ├── config.py             # Pydantic Settings and schema definitions
│   ├── data.py               # Data loading, cleaning, and stratified splitting
│   ├── drift.py              # Evidently Data & Target drift, synthetic drift & metrics
│   ├── evaluate.py           # Metrics calculation and plot artifact generation
│   ├── models.py             # Model configurations and estimator factories
│   ├── monitor.py            # MLflow monitoring runner for Evidently reports
│   ├── preprocessing.py      # ColumnTransformer and target encoder pipeline
│   ├── registry.py           # MLflow model registry & stage transition management
│   ├── serve.py              # FastAPI model serving endpoints (/health, /predict)
│   └── train.py              # MLflow training runner and comparison reporter
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── docs/
│   ├── week17_track_a_compliance.md    # Assignment compliance audit matrix
│   └── week17_submission_checklist.md  # Comprehensive deliverable checklist
├── models/
│   ├── baseline_model.joblib           # Trained baseline classifier artifact
│   ├── preprocessor.joblib             # Fitted scikit-learn ColumnTransformer
│   └── random_forest_balanced.joblib   # Serialized production model artifact
├── reports/
│   ├── custom_metric.json              # Custom churn_rate_shift metric summary
│   ├── evidently_data_drift.html       # Interactive Evidently Data Drift report
│   ├── evidently_target_drift.html     # Interactive Evidently Target Drift report
│   └── monitoring_summary.json         # Complete monitoring metrics summary
├── tests/
│   ├── __init__.py
│   ├── test_data.py          # Tests for data loading, cleaning, splitting
│   ├── test_drift.py         # Tests for drift injection, metrics, Evidently & MLflow
│   ├── test_evaluate.py      # Tests for metrics and plot generation
│   ├── test_preprocessing.py # Tests for preprocessing, leakage, serialization
│   ├── test_registry.py      # Tests for Model Registry metadata & transitions
│   ├── test_serve.py         # Tests for FastAPI serving, validation & inference
│   └── test_train.py         # Tests for training, MLflow tracking, reproducibility
├── .env.example              # Template environment variables
├── .gitignore                # Excludes virtual environments, caches, DBs
├── pyproject.toml            # Project metadata and dependencies
├── uv.lock                   # Pinned deterministic lockfile
└── README.md                 # Complete project documentation
```

---

## 5. Environment & Reproducibility

This project uses [uv](https://github.com/astral-sh/uv) for fast, deterministic Python environment management.

### Prerequisites
- Python `>= 3.12`
- `uv` installed (`curl -LsSf https://astral.sh/uv/install.sh | sh` or via package manager)

### Installation

1. Clone the repository and navigate into the project directory:
   ```bash
   git clone https://github.com/HimalBhandari05/techno-churn-mlops.git
   cd techno-churn-mlops
   ```

2. Synchronize dependencies and create the virtual environment:
   ```bash
   uv sync
   ```

3. (Optional) Create local `.env` configuration from the template:
   ```bash
   cp .env.example .env
   ```

### Running Tests

Execute the automated test suite (44 tests covering data cleaning, splitting, preprocessing, model training, MLflow tracking, registry stage transitions, FastAPI endpoints, and Evidently drift monitoring):

```bash
uv run pytest
```

---

## 6. Data Pipeline

To guarantee reproducibility and strictly prevent data leakage between training, evaluation, and inference:

1. **Stratified Splitting**: [`app/data.py:split_data`](file:///home/himalbhandari/techno-churn-mlops/app/data.py#L59-L84) splits raw features and target with a fixed seed (`random_state=42`) and `stratify=y`:
   - **Training Set**: 5,634 rows (80.0%)
   - **Test Set**: 1,409 rows (20.0%)
2. **Train-Only Fitting**: The [`ColumnTransformer`](file:///home/himalbhandari/techno-churn-mlops/app/preprocessing.py#L18-L40) is fitted **exclusively on the training split** (`X_train`):
   - **Numerical Pipeline** (`tenure`, `MonthlyCharges`, `TotalCharges`): `SimpleImputer(strategy="median")` $\rightarrow$ `StandardScaler()`
   - **Categorical Pipeline** (16 categorical features): `SimpleImputer(strategy="most_frequent")` $\rightarrow$ `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`
3. **Out-of-Distribution Handling**: `handle_unknown="ignore"` ensures unobserved categories in production traffic do not cause runtime exceptions.
4. **Target Encoding**: Deterministic binary mapping (`"No"` $\rightarrow$ 0, `"Yes"` $\rightarrow$ 1).

---

## 7. Model Training

### Model Configurations Tested

We tested 4 model configurations spanning three distinct model families:

1. **`logistic_regression_balanced`** (Linear Model)
   - Estimator: `LogisticRegression`
   - Key Hyperparameters: `C=0.5`, `class_weight='balanced'`, `solver='lbfgs'`, `max_iter=1000`
2. **`random_forest_balanced`** (Bagging Ensemble)
   - Estimator: `RandomForestClassifier`
   - Key Hyperparameters: `n_estimators=200`, `max_depth=8`, `min_samples_split=10`, `min_samples_leaf=4`, `class_weight='balanced'`
3. **`hist_gradient_boosting`** (Boosting Ensemble)
   - Estimator: `HistGradientBoostingClassifier`
   - Key Hyperparameters: `learning_rate=0.05`, `max_iter=150`, `max_depth=5`, `min_samples_leaf=20`, `l2_regularization=0.5`, `class_weight='balanced'`
4. **`random_forest_unweighted`** (Unweighted Bagging Baseline)
   - Estimator: `RandomForestClassifier`
   - Key Hyperparameters: `n_estimators=100`, `max_depth=12`, `min_samples_split=5`, `min_samples_leaf=2`, `class_weight=None`

### Hyperparameter Variations & Rationale

- **Algorithm Families**: Linear models provide high interpretability and baseline benchmarking; Random Forests capture complex non-linear feature interactions; Histogram-based Gradient Boosting optimizes additive decision trees along loss gradients with fast binning.
- **Class Weighting (`class_weight='balanced'`)**: In customer churn prediction, missing a churning customer (False Negative) is significantly more costly than mistakenly contacting a loyal customer (False Positive). Balanced weighting dynamically adjusts loss penalties inversely proportional to class frequencies ($w_j = \frac{N}{2 \cdot N_j}$), dramatically boosting churn recall.
- **Tree Depth & Leaf Regularization**: Constraining `max_depth` (e.g. 8 for Random Forest, 5 for Gradient Boosting) and increasing `min_samples_leaf` suppresses tree memorization and prevents overfitting on high-cardinality one-hot encoded features.
- **Learning Rate & L2 Regularization**: For gradient boosting, `learning_rate=0.05` and `l2_regularization=0.5` smooth the optimization trajectory, preventing overshooting on outlier samples.

### Evaluation Metrics & Why Accuracy Alone Is Insufficient

Because the dataset exhibits a **~73.5% vs. ~26.5% class imbalance**, evaluating models on **Accuracy alone is misleading and dangerous**:
- A naive baseline predicting `"No Churn"` for every customer achieves **~73.46% Accuracy**, yet captures **0.0% of churning customers (Recall = 0.0)**.
- Comparing `random_forest_unweighted` (Accuracy: 80.13%, Recall: 53.21%) against `random_forest_balanced` (Accuracy: 75.37%, Recall: 79.68%) illustrates this trade-off: the unweighted model yields higher raw accuracy by ignoring hard-to-detect churners, while the balanced model captures **~80% of all churners**.

The pipeline logs five complementary metrics to MLflow:
- **Accuracy**: Overall fraction of correct predictions across both classes.
- **Precision**: Proportion of predicted churners who actually churned ($\frac{TP}{TP + FP}$).
- **Recall (Sensitivity)**: Proportion of actual churners correctly caught by the model ($\frac{TP}{TP + FN}$).
- **F1-Score**: Harmonic mean of Precision and Recall ($\frac{2 \cdot P \cdot R}{P + R}$).
- **ROC-AUC**: Area Under the Receiver Operating Characteristic curve, measuring threshold-independent discrimination power.

---

## 8. MLflow Experiment Tracking

**Experiment Name**: `week17-track-a-telco-churn`  
**Backend Storage**: `sqlite:///mlflow.db`

### Side-by-Side Run Comparison Table

| Run ID | Model Name | Model Type | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|---|---|
| `bcfc93ea3cdd46288b425de05e673760` | `logistic_regression_balanced` | `LogisticRegression` | 0.7374 | 0.5034 | 0.7834 | 0.6130 | 0.8417 |
| `ff3e86bedeca41b692148c9da8ec1565` | `random_forest_balanced` | `RandomForestClassifier` | 0.7537 | 0.5237 | **0.7968** | **0.6320** | **0.8430** |
| `79e364d6e4c24a17a3577d63eaa4e7b4` | `hist_gradient_boosting` | `HistGradientBoostingClassifier` | 0.7509 | 0.5204 | 0.7861 | 0.6262 | 0.8401 |
| `6f242eb24ab54a9dbe6929dc58caf3ab` | `random_forest_unweighted` | `RandomForestClassifier` | **0.8013** | **0.6546** | 0.5321 | 0.5870 | 0.8370 |

### Logged Artifacts per Run

For each run, MLflow records:
1. `model/`: Packaged scikit-learn model artifact with conda/python environment metadata.
2. `plots/confusion_matrix.png`: Normalized and raw confusion matrix visualization.
3. `plots/roc_curve.png`: ROC curve plot displaying area under the curve (AUC).
4. `metrics/classification_report.json`: Per-class precision, recall, and F1 metrics.

---

## 9. Model Selection

Based on the actual recorded experiment metrics in `sqlite:///mlflow.db`:

- **Selected Model**: `random_forest_balanced`
- **Source Run ID**: `ff3e86bedeca41b692148c9da8ec1565`
- **Selection Rationale**:
  1. **Highest Churn Recall (`0.7968`)**: Identifies 79.68% of churning customers, minimizing expensive False Negatives.
  2. **Highest F1-Score (`0.6320`)**: Provides the optimal balance between precision and recall among all tested configurations.
  3. **Highest ROC-AUC (`0.8430`)**: Delivers the strongest threshold-independent discrimination capability across all models.
  4. While `random_forest_unweighted` achieved higher overall accuracy (80.13%), its low recall of 53.21% failed to identify almost half of the churning customers.

---

## 10. MLflow Model Registry

The selected model was registered directly into the MLflow Model Registry using its existing Phase 2 run artifact:

- **Registered Model Name**: `telco-churn-classifier`
- **Registered Model Version**: `1`
- **Source Run ID**: `ff3e86bedeca41b692148c9da8ec1565`
- **Source Artifact URI**: `runs:/ff3e86bedeca41b692148c9da8ec1565/model`
- **Lifecycle Stage Transitions Executed**:
  1. **Transition 1**: `None` $\longrightarrow$ `Staging`
  2. **Transition 2**: `Staging` $\longrightarrow$ `Production`
- **Current Status**: `READY`, Stage: `Production`
- **Tags**: `run_id`, `model_type=random_forest_balanced`, `recall=0.7968`, `f1_score=0.6320`

---

## 11. Model Serving

The serving layer is implemented as a production-grade FastAPI application ([`app/serve.py`](file:///home/himalbhandari/techno-churn-mlops/app/serve.py)):

```text
HTTP Request (JSON)
        │
        ▼
[Pydantic CustomerFeatures Validation]
        │
        ▼
[clean_raw_dataframe() — Handles TotalCharges whitespace & schema alignment]
        │
        ▼
[ColumnTransformer.transform() — 46 One-Hot + Scaled Features]
        │
        ▼
[MLflow Registered Production Model — RandomForestClassifier]
        │
        ▼
HTTP Response (JSON: prediction + churn_probability)
```

**Preprocessing Consistency**: The serving endpoint uses the exact preprocessing pipeline fit during training (`preprocessor.joblib`), ensuring zero feature schema mismatch or serving skew.

### Endpoints Specification

| Method | Path | Description | Request Body | Response Body |
|---|---|---|---|---|
| `GET` | `/health` | Service health status and loaded model metadata | None | `HealthResponse` |
| `POST` | `/predict` | Predict customer churn class and probability estimate | `CustomerFeatures` JSON | `PredictionResponse` |

### Starting the Server

```bash
uv run uvicorn app.serve:app --host 0.0.0.0 --port 8000
```

### Example Prediction Request (`curl`)

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
    "TotalCharges": 29.85
  }'
```

### Example Response

```json
{
  "prediction": "Yes",
  "churn_probability": 0.7188,
  "model_name": "telco-churn-classifier",
  "model_version": "1",
  "stage": "Production"
}
```

---

## 12. Evidently Drift Monitoring

### Reference vs. Current Datasets

The dataset is partitioned deterministically using `random_state=42`:
- **Reference Dataset**: 4,930 rows (~70%), representing baseline historical customer behavior.
  - Churn Distribution: `No`: 73.75%, `Yes`: 26.25%
- **Current Dataset**: 2,113 rows (~30%), representing incoming production traffic.
  - Pre-drift Churn Distribution: `No`: 72.79%, `Yes`: 27.21%

### Synthetic Drift Scenario

To evaluate Evidently's sensitivity to distribution shifts, controlled perturbations were introduced into a copy of the current dataset:

1. **Numerical Drift**:
   - `MonthlyCharges`: Added +$30.00 additive shift (simulating subscription price increases).
   - `tenure`: Scaled down by 50% ($0.5 \times \text{tenure}$), simulating an influx of newer customers.
   - `TotalCharges`: Recalculated based on modified tenure and pricing.
2. **Categorical Drift**:
   - `Contract`: 80% of 'One year' and 'Two year' contracts shifted to 'Month-to-month'.
   - `InternetService`: 65% of 'DSL' customers shifted to 'Fiber optic'.
   - `PaymentMethod`: 60% of automated payment methods shifted to 'Electronic check'.
3. **Target Drift**:
   - `Churn`: 30% of non-churning customers shifted to 'Yes', increasing churn rate from 26.25% to 48.08%.

### Data Drift Results & Statistical Tests

Evidently evaluated all 19 feature columns using sample-size adaptive statistical distance metrics:

| Column | Type | Statistical Drift Method | Threshold | Drift Score / Distance | Drift Status |
|---|---|---|---|---|---|
| `MonthlyCharges` | Numerical | Wasserstein distance (normed) | 0.10 | **1.0164** | **DRIFT DETECTED** |
| `tenure` | Numerical | Wasserstein distance (normed) | 0.10 | **0.6871** | **DRIFT DETECTED** |
| `TotalCharges` | Numerical | Wasserstein distance (normed) | 0.10 | **0.3118** | **DRIFT DETECTED** |
| `Contract` | Categorical | Jensen-Shannon distance | 0.10 | **0.2934** | **DRIFT DETECTED** |
| `PaymentMethod` | Categorical | Jensen-Shannon distance | 0.10 | **0.2316** | **DRIFT DETECTED** |
| `InternetService` | Categorical | Jensen-Shannon distance | 0.10 | **0.2011** | **DRIFT DETECTED** |
| `gender` | Categorical | Jensen-Shannon distance | 0.10 | 0.0083 | No Drift |
| `SeniorCitizen` | Categorical | Jensen-Shannon distance | 0.10 | 0.0125 | No Drift |
| `Partner` | Categorical | Jensen-Shannon distance | 0.10 | 0.0011 | No Drift |
| `Dependents` | Categorical | Jensen-Shannon distance | 0.10 | 0.0005 | No Drift |
| `PaperlessBilling` | Categorical | Jensen-Shannon distance | 0.10 | 0.0086 | No Drift |
| ... *(remaining 8 cols)*| Categorical | Jensen-Shannon distance | 0.10 | < 0.015 | No Drift |

- **Total Features Analyzed**: 19
- **Drifted Features Detected**: **6 (31.58%)** — exactly matching the 6 injected features.
- **Data Drift Report Path**: [`reports/evidently_data_drift.html`](file:///home/himalbhandari/techno-churn-mlops/reports/evidently_data_drift.html)

### Target Drift Results

- **Reference Churn Rate**: **26.25%**
- **Current Churn Rate**: **48.08%**
- **Statistical Test**: Jensen-Shannon distance = **0.1607** (Threshold: 0.10)
- **Target Drift Detected**: **True**
- **Target Drift Report Path**: [`reports/evidently_target_drift.html`](file:///home/himalbhandari/techno-churn-mlops/reports/evidently_target_drift.html)

### Custom Churn Rate Shift Metric

- **Metric Name**: `churn_rate_shift`
- **Mathematical Formula**:
  $$\Delta_{\text{churn}} = |\text{churn\_rate}_{\text{current}} - \text{churn\_rate}_{\text{reference}}|$$
- **Configured Threshold**: `0.05` (5.0% maximum allowable shift)
- **Reference Value**: `0.2625` (26.25%)
- **Current Value**: `0.4808` (48.08%)
- **Calculated Difference**: **`0.2184` (21.84%)**
- **Evaluation Status**: **`FAIL (DRIFT DETECTED)`**
- **Summary JSON**: [`reports/custom_metric.json`](file:///home/himalbhandari/techno-churn-mlops/reports/custom_metric.json)

---

## 13. MLflow Monitoring

- **Monitoring Experiment Name**: `week17-track-a-telco-churn-monitoring`
- **Actual Run ID**: `1aac8ca668f3424f83de94a9fb781046`
- **Logged Metrics**:
  - `drifted_columns_count`: `6.0`
  - `drifted_columns_ratio`: `0.3158`
  - `dataset_drift_detected`: `1.0`
  - `target_drift_detected`: `1.0`
  - `target_drift_score`: `0.1607`
  - `reference_churn_rate`: `0.2625`
  - `current_churn_rate`: `0.4808`
  - `churn_rate_shift`: `0.2184`
  - `custom_metric_pass`: `0.0`
- **Logged Artifacts**:
  - `evidently_reports/evidently_data_drift.html`
  - `evidently_reports/evidently_target_drift.html`
  - `monitoring_summaries/monitoring_summary.json`
  - `monitoring_summaries/custom_metric.json`

---

## 14. Reproduction

To reproduce all results from scratch:

```bash
# 1. Install dependencies
uv sync

# 2. Run automated test suite
uv run pytest

# 3. Train all model configurations and track in MLflow
uv run python -m app.train

# 4. Register best model and transition to Production
uv run python -m app.registry

# 5. Execute Evidently drift monitoring and log to MLflow
uv run python -m app.monitor

# 6. Launch FastAPI model serving API
uv run uvicorn app.serve:app --host 0.0.0.0 --port 8000

# 7. Launch MLflow UI to inspect tracking and monitoring runs
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

---

## 15. Results

- **Model Performance**: `random_forest_balanced` captured **79.68% of all churning customers** on the unseen test set, achieving an F1-score of `0.6320` and ROC-AUC of `0.8430`.
- **Serving Efficiency**: The FastAPI endpoint performs feature schema validation, data cleaning, one-hot encoding, and random forest probability estimation in under 15ms per request.
- **Monitoring Sensitivity**: Evidently AI and the custom `churn_rate_shift` metric reliably detected both feature-level and target-level distribution shifts, triggering monitoring alerts without false alarms on unperturbed features.

---

## 16. Limitations

1. **Tabular Scope**: The dataset is limited to static tabular customer records and does not contain time-series telemetry or detailed event logs.
2. **Synthetic Drift Demonstration**: While the synthetic perturbations demonstrate that the monitoring system detects distribution shifts, real-world drift often involves subtle multidimensional covariate shifts that may require continuous streaming metrics.
3. **Local Storage Backend**: The MLflow tracking and registry components use a local SQLite database (`mlflow.db`); production enterprise deployments should connect to managed PostgreSQL backend and S3/GCS artifact stores.

---

## 17. Optional Airflow

Airflow was not implemented because it was an optional bonus requirement.

The project provides clean, modular, and standardized Python entry points (`python -m app.train`, `python -m app.registry`, `python -m app.monitor`, `python -m app.serve`) that can be scheduled or orchestrated by any workflow manager (such as Apache Airflow, Prefect, or Dagster) if desired.
