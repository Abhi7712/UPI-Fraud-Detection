
from pathlib import Path
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
TEST_PATH = BASE_DIR / "data" / "processed" / "test_data.csv"
REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


MODEL_NAMES = [
    "logistic_regression",
    "random_forest",
    "isolation_forest_0_02",
    "isolation_forest_0_03",
    "isolation_forest_0_04",
    "isolation_forest_0_05",
    "isolation_forest_0_08"
]


def evaluate_model(model_name, test_df):

    model_path = MODEL_DIR / f"{model_name}.joblib"

    saved_data = joblib.load(model_path)

    model = saved_data["model"]
    feature_columns = saved_data["feature_columns"]

    X_test = test_df[feature_columns]
    y_test = test_df["is_fraud"].astype(int)

    if model_name.startswith("isolation_forest"):

        raw_predictions = model.predict(X_test)

        predictions = (raw_predictions == -1).astype(int)

        anomaly_scores = -model.decision_function(X_test)

        pr_auc = average_precision_score(
            y_test,
            anomaly_scores
        )

    else:

        predictions = model.predict(X_test)

        probabilities = model.predict_proba(X_test)[:, 1]

        pr_auc = average_precision_score(
            y_test,
            probabilities
        )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )

    results = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "PR-AUC": pr_auc,
        "Fraud Detected": int(matrix[1, 1]),
        "False Positives": int(matrix[0, 1]),
        "Missed Fraud": int(matrix[1, 0])
    }

    print(f"\n{'=' * 45}")
    print(model_name.upper())
    print("=" * 45)

    for metric, value in results.items():

        if metric != "Model":
            print(f"{metric}: {value:.4f}" if isinstance(value, float)
                  else f"{metric}: {value}")

    print("\nConfusion Matrix:")
    print(matrix)

    return results, matrix


def main():

    print("Loading test dataset...")

    test_df = pd.read_csv(TEST_PATH)

    all_results = []
    all_matrices = []

    for model_name in MODEL_NAMES:

        results, matrix = evaluate_model(
            model_name,
            test_df
        )

        all_results.append(results)

        all_matrices.append({
            "Model": model_name,
            "TN": matrix[0, 0],
            "FP": matrix[0, 1],
            "FN": matrix[1, 0],
            "TP": matrix[1, 1]
        })

    comparison_df = pd.DataFrame(all_results)

    comparison_df.to_csv(
        REPORT_DIR / "model_comparison.csv",
        index=False
    )

    matrices_df = pd.DataFrame(all_matrices)

    matrices_df.to_csv(
        REPORT_DIR / "confusion_matrices.csv",
        index=False
    )

    print("\n\nMODEL COMPARISON")
    print("=" * 100)

    print(
        comparison_df.round(4).to_string(index=False)
    )

    print("\nReports saved successfully:")
    print(REPORT_DIR / "model_comparison.csv")
    print(REPORT_DIR / "confusion_matrices.csv")


if __name__ == "__main__":
    main()
