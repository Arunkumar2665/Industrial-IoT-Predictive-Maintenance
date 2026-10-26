
from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

# Load the saved feature engineering, scaler, and models
poly = joblib.load(MODEL_DIR / "polynomial_features.joblib")
scaler = joblib.load(MODEL_DIR / "scaler.joblib")
rul_model = joblib.load(MODEL_DIR / "rul_linear_regression.joblib")
failure_model = joblib.load(
    MODEL_DIR / "failure_logistic_regression.joblib"
)

# Load the threshold selected by the training script
metrics = pd.read_csv(
    REPORT_DIR / "model_metrics.csv",
    index_col=0,
)
threshold = float(metrics.loc["selected_threshold", "value"])

# Example sensor readings; replace these with new measurements
sensor_readings = pd.DataFrame([{
    "vibration": 6.5,
    "spindle_temperature": 85.0,
    "motor_voltage": 222.0,
    "acoustic_noise": 62.0,
}])

# Apply exactly the same feature transformations used in training
X_poly = poly.transform(sensor_readings)

# The scaler was fitted on NumPy arrays, so pass it an array.
X_scaled = scaler.transform(X_poly)


# Restore feature names for the trained prediction models.
feature_names = poly.get_feature_names_out(sensor_readings.columns)
X_scaled_df = pd.DataFrame(
    X_scaled,
    columns=feature_names,
    index=sensor_readings.index
)

# Predict remaining useful life and failure probability
predicted_rul = float(rul_model.predict(X_scaled_df)[0])
failure_probability = float(failure_model.predict_proba(X_scaled_df)[0, 1])


failure_flag = int(failure_probability >= threshold)

print("=== MACHINE HEALTH PREDICTION ===")
print("\nSensor readings:")
print(sensor_readings.to_string(index=False))

print(f"\nPredicted remaining useful life: {predicted_rul:.2f} hours")
print(f"Estimated failure probability: {failure_probability:.2%}")
print(f"Decision threshold: {threshold:.2f}")
print(f"High-risk classification: {'YES' if failure_flag else 'NO'}")

print(
    "\nNote: This is an experimental prediction using synthetic training "
    "data. It is not validated for real-world maintenance decisions."
)
