
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = BASE_DIR / "data" / "raw" / "sensor_telemetry.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORT_DIR = BASE_DIR / "reports"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_FILE, parse_dates=["timestamp"])

print("=== DATASET OVERVIEW ===")
print(f"Shape: {df.shape}")
print(f"Duplicate rows: {df.duplicated().sum()}")
print(f"Duplicate machine/timestamp pairs: "
      f"{df.duplicated(['machine_id', 'timestamp']).sum()}")
print("\nData types:")
print(df.dtypes)
print("\nStatistical summary:")
print(df.select_dtypes(include="number").describe().round(2))

# Remove exact duplicate records
df = df.drop_duplicates()

# Remove duplicate readings for the same machine and timestamp
df = df.drop_duplicates(
    subset=["machine_id", "timestamp"],
    keep="first"
)

# Sort records chronologically within each machine
df = df.sort_values(
    ["machine_id", "timestamp"]
).reset_index(drop=True)

print("\n=== CLEANING RESULTS ===")
print(f"Rows after cleaning: {len(df)}")
print(f"Missing values after cleaning: {df.isnull().sum().sum()}")

# Save the cleaned dataset
clean_file = PROCESSED_DIR / "sensor_telemetry_clean.csv"
df.to_csv(clean_file, index=False)
print(f"Cleaned data saved to: {clean_file}")

# Sensor distributions
sensor_columns = [
    "vibration",
    "spindle_temperature",
    "motor_voltage",
    "acoustic_noise",
]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for column, ax in zip(sensor_columns, axes.flatten()):
    sns.histplot(df[column], kde=True, ax=ax)
    ax.set_title(f"Distribution of {column.replace('_', ' ').title()}")
    ax.set_xlabel(column)

fig.tight_layout()
fig.savefig(REPORT_DIR / "sensor_distributions.png", dpi=150)
plt.close(fig)

# Sensor trends for one example machine
machine = df[df["machine_id"] == "M001"]

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

axes[0].plot(machine["timestamp"], machine["vibration"])
axes[0].set_title("Vibration Trend — Machine M001")
axes[0].set_ylabel("Vibration")

axes[1].plot(machine["timestamp"], machine["spindle_temperature"])
axes[1].set_title("Spindle Temperature Trend — Machine M001")
axes[1].set_ylabel("Temperature")
axes[1].set_xlabel("Timestamp")

fig.tight_layout()
fig.savefig(REPORT_DIR / "machine_sensor_trends.png", dpi=150)
plt.close(fig)

# Correlation matrix
numeric_columns = sensor_columns + ["rul_hours", "failure_within_24h"]

plt.figure(figsize=(10, 7))
sns.heatmap(
    df[numeric_columns].corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    center=0
)
plt.title("Sensor and Target Correlation Matrix")
plt.tight_layout()
plt.savefig(REPORT_DIR / "correlation_matrix.png", dpi=150)
plt.close()

print("\nEDA charts saved:")
for name in [
    "sensor_distributions.png",
    "machine_sensor_trends.png",
    "correlation_matrix.png",
]:
    print(REPORT_DIR / name)
