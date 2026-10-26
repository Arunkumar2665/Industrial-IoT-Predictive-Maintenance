
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

X_train = pd.read_csv(DATA_DIR / "X_train.csv")
X_test = pd.read_csv(DATA_DIR / "X_test.csv")
y_train = pd.read_csv(DATA_DIR / "y_train.csv")
y_test = pd.read_csv(DATA_DIR / "y_test.csv")

y_rul_train = y_train["rul_hours"]
y_rul_test = y_test["rul_hours"]
y_fail_train = y_train["failure_within_24h"]
y_fail_test = y_test["failure_within_24h"]

results = {}

# -------------------------------------------------
# 1. Linear Regression: Remaining Useful Life (RUL)
# -------------------------------------------------
print("\n=== LINEAR REGRESSION: RUL PREDICTION ===")

rul_model = LinearRegression()
rul_model.fit(X_train, y_rul_train)
rul_predictions = rul_model.predict(X_test)

mae = mean_absolute_error(y_rul_test, rul_predictions)
rmse = np.sqrt(mean_squared_error(y_rul_test, rul_predictions))

results["rul_mae_hours"] = float(mae)
results["rul_rmse_hours"] = float(rmse)

print(f"MAE:  {mae:.3f} hours")
print(f"RMSE: {rmse:.3f} hours")

joblib.dump(rul_model, MODEL_DIR / "rul_linear_regression.joblib")

pd.DataFrame({
    "actual_rul_hours": y_rul_test,
    "predicted_rul_hours": rul_predictions,
}).to_csv(REPORT_DIR / "rul_predictions.csv", index=False)

plt.figure(figsize=(8, 6))
plt.scatter(y_rul_test, rul_predictions, alpha=0.5)
lower = min(y_rul_test.min(), rul_predictions.min())
upper = max(y_rul_test.max(), rul_predictions.max())
plt.plot([lower, upper], [lower, upper], linestyle="--")
plt.xlabel("Actual RUL (hours)")
plt.ylabel("Predicted RUL (hours)")
plt.title("Linear Regression: Actual vs Predicted RUL")
plt.tight_layout()
plt.savefig(REPORT_DIR / "rul_actual_vs_predicted.png", dpi=150)
plt.close()

# -------------------------------------------------
# 2. Group-based validation split for threshold choice
# -------------------------------------------------
print("\n=== VALIDATION: THRESHOLD SELECTION ===")

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.25,
    random_state=21,
)

subtrain_idx, validation_idx = next(
    splitter.split(X_train, y_fail_train, groups=y_train["machine_id"])
)

X_subtrain = X_train.iloc[subtrain_idx]
X_validation = X_train.iloc[validation_idx]
y_subtrain = y_fail_train.iloc[subtrain_idx]
y_validation = y_fail_train.iloc[validation_idx]

validation_model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42,
)
validation_model.fit(X_subtrain, y_subtrain)

validation_probabilities = validation_model.predict_proba(X_validation)[:, 1]

threshold_rows = []
for threshold in np.round(np.arange(0.10, 0.91, 0.05), 2):
    predictions = (validation_probabilities >= threshold).astype(int)

    threshold_rows.append({
        "threshold": float(threshold),
        "precision": precision_score(
            y_validation, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_validation, predictions, zero_division=0
        ),
        "f1": f1_score(y_validation, predictions, zero_division=0),
    })

validation_threshold_df = pd.DataFrame(threshold_rows)

eligible = validation_threshold_df[
    validation_threshold_df["precision"] >= 0.30
]

if not eligible.empty:
    selected = eligible.sort_values(
        ["recall", "precision", "threshold"],
        ascending=[False, False, False],
    ).iloc[0]
else:
    selected = validation_threshold_df.sort_values(
        ["recall", "precision", "threshold"],
        ascending=[False, False, False],
    ).iloc[0]

selected_threshold = float(selected["threshold"])

validation_threshold_df.to_csv(
    REPORT_DIR / "validation_threshold_evaluation.csv",
    index=False,
)

print(f"Validation machines: {y_train.iloc[validation_idx]['machine_id'].nunique()}")
print(f"Selected threshold on validation data: {selected_threshold:.2f}")
print(
    "Validation precision:",
    f"{selected['precision']:.3f}",
)
print(
    "Validation recall:",
    f"{selected['recall']:.3f}",
)

# -------------------------------------------------
# 3. Train final classifier on all training data
# -------------------------------------------------
print("\n=== FINAL LOGISTIC REGRESSION: TEST EVALUATION ===")

failure_model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42,
)
failure_model.fit(X_train, y_fail_train)

test_probabilities = failure_model.predict_proba(X_test)[:, 1]
test_predictions_05 = (test_probabilities >= 0.50).astype(int)
test_predictions_selected = (
    test_probabilities >= selected_threshold
).astype(int)

results["failure_accuracy"] = float(
    accuracy_score(y_fail_test, test_predictions_05)
)
results["failure_precision"] = float(
    precision_score(y_fail_test, test_predictions_05, zero_division=0)
)
results["failure_recall_at_0_5"] = float(
    recall_score(y_fail_test, test_predictions_05, zero_division=0)
)
results["failure_f1_at_0_5"] = float(
    f1_score(y_fail_test, test_predictions_05, zero_division=0)
)
results["failure_roc_auc"] = float(
    roc_auc_score(y_fail_test, test_probabilities)
)

results["selected_threshold"] = selected_threshold
results["selected_threshold_precision"] = float(
    precision_score(y_fail_test, test_predictions_selected, zero_division=0)
)
results["selected_threshold_recall"] = float(
    recall_score(y_fail_test, test_predictions_selected, zero_division=0)
)
results["selected_threshold_f1"] = float(
    f1_score(y_fail_test, test_predictions_selected, zero_division=0)
)

print(f"Accuracy at 0.50:  {results['failure_accuracy']:.3f}")
print(f"Precision at 0.50: {results['failure_precision']:.3f}")
print(f"Recall at 0.50:    {results['failure_recall_at_0_5']:.3f}")
print(f"F1 at 0.50:        {results['failure_f1_at_0_5']:.3f}")
print(f"ROC-AUC:           {results['failure_roc_auc']:.3f}")

print(f"\nTest results at validation-selected threshold {selected_threshold:.2f}:")
print(f"Precision: {results['selected_threshold_precision']:.3f}")
print(f"Recall:    {results['selected_threshold_recall']:.3f}")
print(f"F1-score:  {results['selected_threshold_f1']:.3f}")

print("\nClassification report at threshold 0.50:")
print(classification_report(
    y_fail_test, test_predictions_05, zero_division=0
))

joblib.dump(failure_model, MODEL_DIR / "failure_logistic_regression.joblib")

pd.DataFrame({
    "actual_failure": y_fail_test,
    "failure_probability": test_probabilities,
    "predicted_failure_at_0_5": test_predictions_05,
    "predicted_failure_at_selected_threshold": test_predictions_selected,
}).to_csv(REPORT_DIR / "failure_predictions.csv", index=False)

# -------------------------------------------------
# 4. Test-set threshold report (evaluation only)
# Threshold is chosen on validation data, not this report.
# -------------------------------------------------
test_threshold_rows = []

for threshold in np.round(np.arange(0.10, 0.91, 0.05), 2):
    predictions = (test_probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_fail_test, predictions, labels=[0, 1]
    ).ravel()

    test_threshold_rows.append({
        "threshold": float(threshold),
        "precision": precision_score(
            y_fail_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_fail_test, predictions, zero_division=0
        ),
        "f1": f1_score(y_fail_test, predictions, zero_division=0),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "selected_on_validation": bool(
            np.isclose(threshold, selected_threshold)
        ),
    })

test_threshold_df = pd.DataFrame(test_threshold_rows)
test_threshold_df.to_csv(
    REPORT_DIR / "threshold_evaluation.csv",
    index=False,
)

plt.figure(figsize=(9, 5))
plt.plot(
    test_threshold_df["threshold"],
    test_threshold_df["precision"],
    marker="o",
    label="Test precision",
)
plt.plot(
    test_threshold_df["threshold"],
    test_threshold_df["recall"],
    marker="o",
    label="Test recall",
)
plt.axvline(
    selected_threshold,
    linestyle="--",
    label=f"Validation-selected threshold = {selected_threshold:.2f}",
)
plt.xlabel("Decision threshold")
plt.ylabel("Score")
plt.title("Failure Classifier Threshold Evaluation")
plt.legend()
plt.tight_layout()
plt.savefig(REPORT_DIR / "threshold_evaluation.png", dpi=150)
plt.close()

pd.Series(results, name="value").to_csv(
    REPORT_DIR / "model_metrics.csv",
    header=True,
)

print("\n=== TRAINING COMPLETE ===")
print("Models saved in:", MODEL_DIR)
print("Reports saved in:", REPORT_DIR)
print("Metrics saved to:", REPORT_DIR / "model_metrics.csv")
