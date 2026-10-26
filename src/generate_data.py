
from pathlib import Path

import numpy as np
import pandas as pd

# Reproducible synthetic sensor data
rng = np.random.default_rng(42)

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

n_machines = 30
readings_per_machine = 200
rows = []

start_time = pd.Timestamp("2026-01-01 00:00:00")

for machine_number in range(1, n_machines + 1):
    machine_id = f"M{machine_number:03d}"
    age = np.linspace(0, 1, readings_per_machine)

    machine_health = rng.uniform(0.85, 1.15)

    vibration = (
        2.0 + 5.0 * age * machine_health
        + rng.normal(0, 0.35, readings_per_machine)
    )
    temperature = (
        55.0 + 35.0 * age * machine_health
        + rng.normal(0, 2.0, readings_per_machine)
    )
    voltage = (
        230.0 - 8.0 * age
        + rng.normal(0, 2.0, readings_per_machine)
    )
    acoustic_noise = (
        40.0 + 25.0 * age * machine_health
        + rng.normal(0, 1.5, readings_per_machine)
    )

    # Synthetic target: remaining useful operating hours
    rul = np.maximum(
        1.0,
        240.0 * (1.0 - age)
        - 8.0 * (vibration - 2.0)
        - 0.8 * (temperature - 55.0)
        + rng.normal(0, 5.0, readings_per_machine)
    )

    # Positive label means failure risk within the next 24 hours
    failure_within_24h = (rul <= 24).astype(int)

    for i in range(readings_per_machine):
        rows.append({
            "timestamp": start_time + pd.Timedelta(minutes=5 * i),
            "machine_id": machine_id,
            "vibration": round(vibration[i], 3),
            "spindle_temperature": round(temperature[i], 3),
            "motor_voltage": round(voltage[i], 3),
            "acoustic_noise": round(acoustic_noise[i], 3),
            "rul_hours": round(rul[i], 3),
            "failure_within_24h": int(failure_within_24h[i]),
        })

df = pd.DataFrame(rows)
output_file = OUTPUT_DIR / "sensor_telemetry.csv"
df.to_csv(output_file, index=False)

print(f"Dataset saved to: {output_file}")
print(f"Total rows: {len(df)}")
print(f"Total machines: {df['machine_id'].nunique()}")
print("\nFirst five records:")
print(df.head().to_string(index=False))
print("\nFailure label distribution:")
print(df["failure_within_24h"].value_counts().sort_index())
print("\nMissing values:")
print(df.isnull().sum())
