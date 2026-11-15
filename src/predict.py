
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

SUPPORTED_MODELS = [
    "random_forest",
    "logistic_regression"
]


def load_model(model_name="random_forest"):

    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported prediction model: {model_name}. "
            f"Choose from {SUPPORTED_MODELS}"
        )

    model_path = MODEL_DIR / f"{model_name}.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    saved_data = joblib.load(model_path)

    return (
        saved_data["model"],
        saved_data["feature_columns"]
    )


def validate_numeric_inputs(
    amount,
    hour,
    user_avg_amount_prior,
    user_transaction_count_prior,
    transactions_last_hour,
    is_weekend
):

    values = {
        "amount": amount,
        "hour": hour,
        "user_avg_amount_prior": user_avg_amount_prior,
        "user_transaction_count_prior": user_transaction_count_prior,
        "transactions_last_hour": transactions_last_hour
    }

    for name, value in values.items():

        if not np.isfinite(value):
            raise ValueError(f"{name} must be a valid finite number.")

    if amount < 0:
        raise ValueError("Amount cannot be negative.")

    if not 0 <= hour <= 23 or int(hour) != hour:
        raise ValueError("Hour must be an integer between 0 and 23.")

    if user_avg_amount_prior < 0:
        raise ValueError("Historical average cannot be negative.")

    if user_transaction_count_prior < 0:
        raise ValueError("Transaction count cannot be negative.")

    if transactions_last_hour < 0:
        raise ValueError("Last-hour count cannot be negative.")

    if int(user_transaction_count_prior) != user_transaction_count_prior:
        raise ValueError("Transaction count must be an integer.")

    if int(transactions_last_hour) != transactions_last_hour:
        raise ValueError("Last-hour count must be an integer.")

    if is_weekend not in [0, 1]:
        raise ValueError("is_weekend must be 0 or 1.")


def generate_transaction_signals(transaction):

    signals = []

    amount = transaction["amount"]
    average = transaction["user_avg_amount_prior"]
    hour = transaction["hour"]
    velocity = transaction["transactions_last_hour"]

    if average > 0 and amount >= average * 3:

        signals.append(
            "Transaction amount is at least 3 times the user's historical average."
        )

    if hour <= 5:

        signals.append(
            "Transaction occurred during late-night or early-morning hours."
        )

    if velocity >= 3:

        signals.append(
            "Multiple transactions occurred within the previous hour."
        )

    if average == 0:

        signals.append(
            "No previous average transaction amount is available."
        )

    if not signals:

        signals.append(
            "No predefined suspicious indicators were triggered."
        )

    return signals


def predict_transaction(
    amount,
    hour,
    user_avg_amount_prior,
    user_transaction_count_prior,
    transactions_last_hour,
    merchant_category,
    device_type,
    location,
    is_weekend=0,
    model_name="random_forest"
):
    """
    Predict a transaction using a supervised fraud classifier.

    Historical features must contain information available
    strictly before the transaction being evaluated.

    The returned score is an uncalibrated model score,
    not a verified fraud probability.
    """

    validate_numeric_inputs(
        amount,
        hour,
        user_avg_amount_prior,
        user_transaction_count_prior,
        transactions_last_hour,
        is_weekend
    )

    categorical_values = {
        "merchant_category": merchant_category,
        "device_type": device_type,
        "location": location
    }

    for name, value in categorical_values.items():

        if pd.isna(value) or not str(value).strip():
            raise ValueError(f"{name} cannot be empty.")

    model, feature_columns = load_model(model_name)

    transaction = {
        "amount": float(amount),
        "hour": int(hour),
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "is_weekend": int(is_weekend),
        "user_avg_amount_prior": float(user_avg_amount_prior),
        "amount_deviation": float(amount - user_avg_amount_prior),
        "user_transaction_count_prior": int(user_transaction_count_prior),
        "transactions_last_hour": int(transactions_last_hour),
        "merchant_category": str(merchant_category),
        "device_type": str(device_type),
        "location": str(location)
    }

    input_df = pd.DataFrame([transaction])

    input_df = input_df[feature_columns]

    prediction = int(model.predict(input_df)[0])

    # Find the probability/score column belonging to fraud class 1.
    classes = list(model.classes_)

    if 1 not in classes:
        raise ValueError("Trained model does not contain fraud class 1.")

    fraud_index = classes.index(1)

    model_score = float(
        model.predict_proba(input_df)[0][fraud_index]
    )

    score_percentage = round(model_score * 100, 2)

    if score_percentage >= 70:
        risk_level = "High"

    elif score_percentage >= 30:
        risk_level = "Medium"

    else:
        risk_level = "Low"

    signals = generate_transaction_signals(transaction)

    return {
        "prediction": prediction,
        "risk_score": score_percentage,
        "model_score": score_percentage,
        "risk_level": risk_level,
        "model": model_name,
        "signals": signals,
        "score_type": "Uncalibrated model score"
    }


if __name__ == "__main__":

    result = predict_transaction(
        amount=15000,
        hour=22,
        user_avg_amount_prior=3000,
        user_transaction_count_prior=10,
        transactions_last_hour=4,
        merchant_category="Shopping",
        device_type="Mobile",
        location="Delhi",
        is_weekend=0
    )

    print("\nTransaction Prediction")
    print("----------------------")
    print("Model:", result["model"])
    print("Prediction:", result["prediction"])
    print("Model Score:", result["model_score"], "%")
    print("Risk Level:", result["risk_level"])
    print("Score Type:", result["score_type"])

    print("\nTransaction Signals:")

    for signal in result["signals"]:
        print("-", signal)
