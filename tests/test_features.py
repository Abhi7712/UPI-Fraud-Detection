
import pandas as pd

from src.feature_engineering import create_features


def sample_data():

    return pd.DataFrame({
        "transaction_id": ["T1", "T2", "T3", "T4"],
        "user_id": ["U1", "U1", "U1", "U2"],
        "amount": [1000, 2000, 3000, 500],
        "transaction_time": [
            "2026-01-01 10:00:00",
            "2026-01-01 10:30:00",
            "2026-01-01 12:00:00",
            "2026-01-01 11:00:00"
        ],
        "merchant_category": [
            "Shopping", "Food", "Travel", "Food"
        ],
        "device_type": [
            "Mobile", "Mobile", "Laptop", "Mobile"
        ],
        "location": [
            "Delhi", "Delhi", "Mumbai", "Delhi"
        ],
        "is_fraud": [0, 0, 1, 0]
    })


def test_historical_average_uses_only_previous_transactions():

    df = create_features(sample_data())

    user_transactions = df[df["user_id"] == "U1"]

    assert user_transactions.iloc[0]["user_avg_amount_prior"] == 0

    assert user_transactions.iloc[1]["user_avg_amount_prior"] == 1000

    assert user_transactions.iloc[2]["user_avg_amount_prior"] == 1500


def test_transaction_count_uses_previous_transactions():

    df = create_features(sample_data())

    user_transactions = df[df["user_id"] == "U1"]

    assert user_transactions.iloc[0]["user_transaction_count_prior"] == 0

    assert user_transactions.iloc[1]["user_transaction_count_prior"] == 1

    assert user_transactions.iloc[2]["user_transaction_count_prior"] == 2


def test_last_hour_transaction_count():

    df = create_features(sample_data())

    user_transactions = df[df["user_id"] == "U1"]

    assert user_transactions.iloc[0]["transactions_last_hour"] == 0

    assert user_transactions.iloc[1]["transactions_last_hour"] == 1

    assert user_transactions.iloc[2]["transactions_last_hour"] == 0


def test_first_transaction_of_new_user():

    df = create_features(sample_data())

    new_user = df[df["user_id"] == "U2"].iloc[0]

    assert new_user["user_avg_amount_prior"] == 0

    assert new_user["user_transaction_count_prior"] == 0

    assert new_user["transactions_last_hour"] == 0


def test_time_features_are_created():

    df = create_features(sample_data())

    required_columns = [
        "hour",
        "hour_sin",
        "hour_cos",
        "is_weekend"
    ]

    for column in required_columns:
        assert column in df.columns
