
from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"
TEST_PATH = BASE_DIR / "data" / "processed" / "test_data.csv"
REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_model(model_name, test_df):

    model_path = MODEL_DIR / f"{model_name}.joblib"

    saved_data = joblib.load(model_path)

    model = saved_data["model"]
    feature_columns = saved_data["feature_columns"]

    X_test = test_df[feature_columns]
    y_test = test_df["is_fraud"].astype(int)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

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
        "PR-AUC": average_precision_score(
            y_test, probabilities
        )
    }

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )

    print("\n" + "=" * 50)
    print("MODEL:", model_name.upper())
    print("=" * 50)

    print("\nPerformance Metrics:")
    for metric, value in results.items():
        if metric != "Model":
            print(f"{metric}: {value:.4f}")

    print("\nConfusion Matrix:")
    print(matrix)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Legitimate", "Fraud"],
            zero_division=0
        )
    )

    return results, matrix


def main():

    print("Loading test dataset...")

    test_df = pd.read_csv(TEST_PATH)

    model_names = [
        "logistic_regression",
        "random_forest"
    ]

    all_results = []
    confusion_matrices = {}

    for model_name in model_names:

        results, matrix = evaluate_model(
            model_name,
            test_df
        )

        all_results.append(results)
        confusion_matrices[model_name] = matrix.tolist()

    comparison_df = pd.DataFrame(all_results)

    comparison_path = REPORT_DIR / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)

    confusion_path = REPORT_DIR / "confusion_matrices.csv"

    confusion_rows = []

    for model_name, matrix in confusion_matrices.items():

        confusion_rows.append({
            "Model": model_name,
            "True Negative": matrix[0][0],
            "False Positive": matrix[0][1],
            "False Negative": matrix[1][0],
            "True Positive": matrix[1][1]
        })

    pd.DataFrame(confusion_rows).to_csv(
        confusion_path,
        index=False
    )

    print("\n" + "=" * 50)
    print("MODEL COMPARISON")
    print("=" * 50)

    print(comparison_df.to_string(index=False))

    print("\nReports saved successfully:")
    print(comparison_path)
    print(confusion_path)


if __name__ == "__main__":
    main()
