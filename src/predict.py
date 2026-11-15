
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"


def load_model(model_name="random_forest"):

    model_path = MODEL_DIR / f"{model_name}.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    saved_data = joblib.load(model_path)

    return saved_data["model"], saved_data["feature_columns"]


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
    Predict the fraud risk of a new transaction.

    Historical inputs must come from information available
    before the transaction being evaluated.
    """

    if amount < 0:
        raise ValueError("Amount cannot be negative.")

    if not 0 <= hour <= 23:
        raise ValueError("Hour must be between 0 and 23.")

    if user_avg_amount_prior < 0:
        raise ValueError("Historical average cannot be negative.")

    if user_transaction_count_prior < 0:
        raise ValueError("Transaction count cannot be negative.")

    if transactions_last_hour < 0:
        raise ValueError("Last-hour count cannot be negative.")

    if is_weekend not in [0, 1]:
        raise ValueError("is_weekend must be 0 or 1.")

    model, feature_columns = load_model(model_name)

    transaction = {
        "amount": amount,
        "hour": hour,
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "is_weekend": is_weekend,
        "user_avg_amount_prior": user_avg_amount_prior,
        "amount_deviation": amount - user_avg_amount_prior,
        "user_transaction_count_prior": user_transaction_count_prior,
        "transactions_last_hour": transactions_last_hour,
        "merchant_category": merchant_category,
        "device_type": device_type,
        "location": location
    }

    input_df = pd.DataFrame([transaction])

    input_df = input_df[feature_columns]

    prediction = int(model.predict(input_df)[0])

    probability = float(
        model.predict_proba(input_df)[0][1]
    )

    risk_score = round(probability * 100, 2)

    if risk_score >= 70:
        risk_level = "High"
    elif risk_score >= 30:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    return {
        "prediction": prediction,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "model": model_name
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
    print("Risk Score:", result["risk_score"], "%")
    print("Risk Level:", result["risk_level"])
