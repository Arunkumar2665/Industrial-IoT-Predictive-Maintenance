
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = BASE_DIR / "data" / "processed" / "sensor_telemetry_clean.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_FILE)

sensor_columns = [
    "vibration",
    "spindle_temperature",
    "motor_voltage",
    "acoustic_noise",
]

# Use a machine-level split to prevent machine overlap.
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42,
)

train_idx, test_idx = next(
    splitter.split(df, groups=df["machine_id"])
)

train_df = df.iloc[train_idx].copy()
test_df = df.iloc[test_idx].copy()

# Fit feature engineering only on training data.
poly = PolynomialFeatures(
    degree=2,
    include_bias=False,
)

X_train_poly = poly.fit_transform(train_df[sensor_columns])
X_test_poly = poly.transform(test_df[sensor_columns])

feature_names = poly.get_feature_names_out(sensor_columns)

# Fit scaling only on training data.
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_poly)
X_test_scaled = scaler.transform(X_test_poly)

X_train = pd.DataFrame(
    X_train_scaled,
    columns=feature_names,
    index=train_df.index,
)
X_test = pd.DataFrame(
    X_test_scaled,
    columns=feature_names,
    index=test_df.index,
)

# Preserve targets and machine IDs separately.
y_rul_train = train_df["rul_hours"]
y_rul_test = test_df["rul_hours"]

y_failure_train = train_df["failure_within_24h"]
y_failure_test = test_df["failure_within_24h"]

X_train.to_csv(OUTPUT_DIR / "X_train.csv", index=False)
X_test.to_csv(OUTPUT_DIR / "X_test.csv", index=False)

pd.DataFrame({
    "rul_hours": y_rul_train,
    "failure_within_24h": y_failure_train,
    "machine_id": train_df["machine_id"],
}).to_csv(OUTPUT_DIR / "y_train.csv", index=False)

pd.DataFrame({
    "rul_hours": y_rul_test,
    "failure_within_24h": y_failure_test,
    "machine_id": test_df["machine_id"],
}).to_csv(OUTPUT_DIR / "y_test.csv", index=False)

import joblib

joblib.dump(poly, MODEL_DIR / "polynomial_features.joblib")
joblib.dump(scaler, MODEL_DIR / "scaler.joblib")

print("=== DATA PREPARATION COMPLETE ===")
print(f"Training rows: {len(X_train)}")
print(f"Testing rows: {len(X_test)}")
print(f"Polynomial features: {len(feature_names)}")
print(f"Training machines: {train_df['machine_id'].nunique()}")
print(f"Testing machines: {test_df['machine_id'].nunique()}")
print(f"Machine overlap: {set(train_df['machine_id']) & set(test_df['machine_id'])}")
print("\nTraining failure-label distribution:")
print(y_failure_train.value_counts().sort_index())
print("\nTesting failure-label distribution:")
print(y_failure_test.value_counts().sort_index())
print("\nPrepared files saved in:", OUTPUT_DIR)
