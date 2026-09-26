# MLflow Experiment Run Comparison

**Experiment Name**: `week17-track-a-telco-churn`

| Run ID                           | Model Name                   | Model Type                     |   Accuracy |   Precision |   Recall |   F1-Score |   ROC-AUC |
|:---------------------------------|:-----------------------------|:-------------------------------|-----------:|------------:|---------:|-----------:|----------:|
| bcfc93ea3cdd46288b425de05e673760 | logistic_regression_balanced | LogisticRegression             |     0.7374 |      0.5034 |   0.7834 |     0.613  |    0.8417 |
| ff3e86bedeca41b692148c9da8ec1565 | random_forest_balanced       | RandomForestClassifier         |     0.7537 |      0.5237 |   0.7968 |     0.632  |    0.843  |
| 79e364d6e4c24a17a3577d63eaa4e7b4 | hist_gradient_boosting       | HistGradientBoostingClassifier |     0.7509 |      0.5204 |   0.7861 |     0.6262 |    0.8401 |
| 6f242eb24ab54a9dbe6929dc58caf3ab | random_forest_unweighted     | RandomForestClassifier         |     0.8013 |      0.6546 |   0.5321 |     0.587  |    0.837  |
