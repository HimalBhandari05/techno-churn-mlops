# Week 17 Track A Submission Checklist

This checklist verifies all deliverables and requirements for the **Week 17 Track A — Data Science MLOps: Customer Churn Pipeline** assignment.

---

## 1. Environment & Package Management (uv)

- [x] **Declarative Project Configuration**: [`pyproject.toml`](file:///home/himalbhandari/techno-churn-mlops/pyproject.toml) specifies project metadata, Python `>=3.12`, dependencies, and tool settings.
- [x] **Deterministic Lockfile**: Committed [`uv.lock`](file:///home/himalbhandari/techno-churn-mlops/uv.lock) locks exact dependency versions and platform hashes.
- [x] **Single-Command Synchronization**: `uv sync` installs all required packages and creates `.venv` reliably.
- [x] **Automated Test Suite**: 44 tests pass with `uv run pytest`.

---

## 2. Dataset & Data Preprocessing Pipeline

- [x] **Dataset Ingestion**: Telco Customer Churn dataset located at [`data/WA_Fn-UseC_-Telco-Customer-Churn.csv`](file:///home/himalbhandari/techno-churn-mlops/data/WA_Fn-UseC_-Telco-Customer-Churn.csv) (7,043 rows, 21 columns).
- [x] **Data Cleaning**: `TotalCharges` whitespace coercion and zero-imputation; `customerID` dropped from feature matrix ([`app/data.py`](file:///home/himalbhandari/techno-churn-mlops/app/data.py)).
- [x] **Stratified Splitting**: 80/20 train/test split with `random_state=42` preserving ~73.46% / ~26.54% class balance.
- [x] **Data Leakage Prevention**: `ColumnTransformer` fitted exclusively on `X_train` with `handle_unknown="ignore"` for categorical features ([`app/preprocessing.py`](file:///home/himalbhandari/techno-churn-mlops/app/preprocessing.py)).
- [x] **Target Encoding**: Deterministic binary mapping (`"No"` $\rightarrow$ 0, `"Yes"` $\rightarrow$ 1).

---

## 3. Model Training & MLflow Experiment Tracking

- [x] **Dedicated MLflow Experiment**: `week17-track-a-telco-churn` recorded in SQLite backend (`sqlite:///mlflow.db`).
- [x] **Multiple Distinct Model Families**: 4 configurations trained across 3 distinct families:
  1. `logistic_regression_balanced` (`LogisticRegression`)
  2. `random_forest_balanced` (`RandomForestClassifier`)
  3. `hist_gradient_boosting` (`HistGradientBoostingClassifier`)
  4. `random_forest_unweighted` (`RandomForestClassifier`)
- [x] **Hyperparameter Variations**: Meaningful structural differences (`class_weight='balanced'`, tree depth limits, leaf sample regularization, learning rates).
- [x] **Complete Metric Logging**: Accuracy, Precision, Recall, F1-Score, and ROC-AUC logged for all runs.
- [x] **Artifact Logging**: Trained model, confusion matrix plot (`confusion_matrix.png`), ROC curve plot (`roc_curve.png`), and classification report dictionary (`classification_report.json`) logged to MLflow.
- [x] **Comparison Table**: Comprehensive side-by-side comparison table generated and documented.

---

## 4. Model Selection & MLflow Model Registry

- [x] **Multi-Metric Selection Analysis**: Documented why accuracy alone is insufficient for class-imbalanced churn and selected `random_forest_balanced` based on highest Recall (0.7968), F1 (0.6320), and ROC-AUC (0.8430).
- [x] **MLflow Model Registry Registration**: Model registered as `telco-churn-classifier` (Version 1) from run `ff3e86bedeca41b692148c9da8ec1565`.
- [x] **Lifecycle Stage Transitions**: Executed transitions `None` $\rightarrow$ `Staging` $\rightarrow$ `Production`.
- [x] **Production Status**: Registered version 1 is in active `Production` stage with metadata and tags.

---

## 5. Production Model Serving API (FastAPI)

- [x] **FastAPI Application**: High-performance ASGI app in [`app/serve.py`](file:///home/himalbhandari/techno-churn-mlops/app/serve.py).
- [x] **Health Endpoint (`GET /health`)**: Returns service status, model name, version, and stage.
- [x] **Inference Endpoint (`POST /predict`)**: Accepts raw customer JSON, applies fitted preprocessing transformer, and outputs prediction with estimated churn probability.
- [x] **Schema Validation**: Pydantic `CustomerFeatures` enforces strict validation on all 19 feature columns with HTTP 422 error handling.
- [x] **Zero Preprocessing Skew**: Uses the exact fitted `preprocessor.joblib` pipeline from training.

---

## 6. Evidently AI Drift Monitoring

- [x] **Reference / Current Dataset Split**: Deterministic 70/30 partition (`random_state=42`) into reference (4,930 rows) and current (2,113 rows).
- [x] **Synthetic Drift Scenario**: Controlled perturbation of numerical features (`MonthlyCharges`, `tenure`), categorical features (`Contract`, `InternetService`, `PaymentMethod`), and target (`Churn`).
- [x] **Data Drift Report**: HTML report generated using Evidently 0.7.x (`DataDriftPreset`). Accurately detects 6/19 drifted features. Saved to [`reports/evidently_data_drift.html`](file:///home/himalbhandari/techno-churn-mlops/reports/evidently_data_drift.html).
- [x] **Target Drift Report**: HTML report generated using Evidently (`TargetDriftPreset`). Accurately detects shift in churn rate (26.25% $\rightarrow$ 48.08%, JS distance = 0.1607). Saved to [`reports/evidently_target_drift.html`](file:///home/himalbhandari/techno-churn-mlops/reports/evidently_target_drift.html).
- [x] **Custom Drift Metric**: Implemented `churn_rate_shift` ($|\Delta| = 0.2184$, threshold = 0.05). Output saved to [`reports/custom_metric.json`](file:///home/himalbhandari/techno-churn-mlops/reports/custom_metric.json).
- [x] **MLflow Monitoring Experiment**: Dedicated experiment `week17-track-a-telco-churn-monitoring` with logged run `1aac8ca668f3424f83de94a9fb781046` containing all drift metrics and HTML/JSON artifacts.

---

## 7. Quality, Security & Documentation

- [x] **Code Style & Modularity**: Clean separation of concerns into dedicated modules under `app/`.
- [x] **Comprehensive Documentation**: Updated [`README.md`](file:///home/himalbhandari/techno-churn-mlops/README.md) organized into 17 numbered sections covering the complete lifecycle.
- [x] **Terminology Precision**: Accurate probability estimation terminology and non-causal drift reporting language.
- [x] **Airflow Status Statement**: Explicitly stated in Section 17 that Airflow was not implemented because it is an optional bonus requirement.
- [x] **Compliance Matrix**: Completed [`docs/week17_track_a_compliance.md`](file:///home/himalbhandari/techno-churn-mlops/docs/week17_track_a_compliance.md).
- [x] **Repository Cleanliness**: `.gitignore` properly excludes virtual environments, SQLite DBs, and temporary caches. Safe `.env.example` provided without secrets.
