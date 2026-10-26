
# P1 — Industrial IoT Telemetry Anomaly Detection & Failure Prediction

## 1. Project Overview

This project demonstrates a machine-learning pipeline for industrial IoT telemetry analysis. It uses sensor readings to estimate the remaining useful life (RUL) of equipment and classify whether a failure may occur within the next 24 hours.

**Important:** The project uses synthetic data for educational purposes. Its results are not validated for real industrial maintenance decisions.

## 2. Objectives

- Generate and clean machine sensor telemetry.
- Perform exploratory data analysis and visualize sensor distributions and relationships.
- Apply polynomial feature engineering and standardization.
- Predict remaining useful life using Linear Regression.
- Classify failure risk using Logistic Regression.
- Evaluate regression performance using MAE and RMSE.
- Evaluate classification performance using precision, recall, F1-score, and ROC-AUC.
- Select a classification threshold using validation data to prioritize failure recall.

## 3. Technology Stack

- Python 3.12
- pandas and NumPy
- scikit-learn
- Matplotlib and Seaborn
- Joblib
- Jupyter (available for notebook-based exploration)

## 4. Project Structure

```text
P1_IoT_Failure_Prediction/
├── data/
│   ├── raw/
│   │   └── sensor_telemetry.csv
│   └── processed/
│       ├── sensor_telemetry_clean.csv
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
├── models/
│   ├── polynomial_features.joblib
│   ├── scaler.joblib
│   ├── rul_linear_regression.joblib
│   └── failure_logistic_regression.joblib
├── notebooks/
├── reports/
│   ├── correlation_matrix.png
│   ├── machine_sensor_trends.png
│   ├── sensor_distributions.png
│   ├── rul_actual_vs_predicted.png
│   ├── rul_predictions.csv
│   ├── failure_predictions.csv
│   ├── model_metrics.csv
│   ├── threshold_evaluation.csv
│   ├── threshold_evaluation.png
│   └── validation_threshold_evaluation.csv
├── src/
│   ├── generate_data.py
│   ├── data_analysis.py
│   ├── prepare_data.py
│   ├── train_models.py
│   └── predict_machine.py
├── requirements.txt
└── README.md
```

## 5. Dataset

The project generates a reproducible synthetic dataset with 6,000 telemetry records across 30 simulated machines.

### Sensor features

- `vibration`
- `spindle_temperature`
- `motor_voltage`
- `acoustic_noise`

### Prediction targets

- `rul_hours`: simulated remaining useful life in hours.
- `failure_within_24h`: binary target indicating whether failure is expected within 24 hours.

The target labels are generated from the synthetic data-generation logic. They do not represent verified real-world failures.

## 6. Setup

From this project directory, activate the virtual environment:

```bash
source .venv/bin/activate
```

If the environment has not been created yet:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 7. Run the Pipeline

Run these commands in order:

### Step 1 — Generate telemetry

```bash
python src/generate_data.py
```

### Step 2 — Clean and analyze the data

```bash
python src/data_analysis.py
```

### Step 3 — Prepare features and splits

```bash
python src/prepare_data.py
```

### Step 4 — Train and evaluate models

```bash
python src/train_models.py
```

### Step 5 — Run a sample prediction

```bash
python src/predict_machine.py
```

The prediction script loads the saved polynomial transformer, scaler, RUL model, and failure classifier. It prints predicted RUL, estimated failure probability, the selected threshold, and the resulting risk classification.

## 8. Evaluation

The latest run produced the following results:

| Metric | Result |
|---|---:|
| RUL MAE | 11.208 hours |
| RUL RMSE | 14.108 hours |
| Failure accuracy at threshold 0.50 | 0.963 |
| Failure precision at threshold 0.50 | 0.930 |
| Failure recall at threshold 0.50 | 0.943 |
| Failure F1-score at threshold 0.50 | 0.936 |
| Failure ROC-AUC | 0.994 |
| Selected decision threshold | 0.10 |
| Test precision at selected threshold | 0.768 |
| Test recall at selected threshold | 0.997 |

The selected threshold is chosen using validation data, not the test data. The test metrics above come from the separate test set.

A lower decision threshold generally identifies more potential failures, improving recall at the cost of more false positives. Threshold selection should reflect the operational cost of missed failures and unnecessary inspections.

## 9. Generated Deliverables

- Cleaned telemetry dataset and prepared train/test files.
- Saved polynomial feature transformer and scaler.
- Saved Linear Regression and Logistic Regression models.
- RUL prediction output and actual-versus-predicted plot.
- Classification predictions and model metrics.
- Validation threshold evaluation and test threshold report.
- Sensor distribution, trend, and correlation visualizations.
- Script for running a sample machine-health prediction.

## 10. Limitations

- Data is synthetic and cannot establish real industrial model performance.
- The target labels are derived from the simulation logic.
- The sample prediction is a demonstration, not a maintenance recommendation.
- Real deployment would require validated historical telemetry, careful time-aware evaluation, operational monitoring, and domain-expert review.
- The current threshold policy uses a minimum validation precision of 0.30 when selecting the highest-recall threshold. This is an illustrative policy, not a production-approved operating point.

## 11. Future Improvements

- Validate on real industrial telemetry.
- Add time-window and temporal sensor features.
- Compare additional regression and classification algorithms.
- Tune thresholds according to maintenance costs and failure severity.
- Add drift monitoring and automated model retraining.
- Develop a dashboard for live sensor monitoring.

## 12. Conclusion

This project demonstrates an end-to-end educational workflow for synthetic industrial IoT telemetry, including preprocessing, feature engineering, model training, threshold selection, evaluation, artifact saving, and sample inference.
