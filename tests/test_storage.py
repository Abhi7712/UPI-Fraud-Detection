import pandas as pd

from src.storage import (
    save_prediction,
    get_prediction_history,
    clear_prediction_history
)


def test_save_and_read_prediction_history(tmp_path):
    db_path = tmp_path / "history.db"
    transaction = {
        "amount": 1250,
        "hour": 12,
        "user_avg_amount_prior": 1000,
        "user_transaction_count_prior": 4,
        "transactions_last_hour": 1,
        "merchant_category": "Food",
        "device_type": "Android",
        "location": "Delhi",
        "is_weekend": 0
    }
    result = {
        "prediction": 1,
        "model_score": 82.5,
        "risk_level": "High",
        "model": "random_forest",
        "signals": ["Amount is unusual"]
    }

    record_id = save_prediction(transaction, result, db_path=db_path)
    history = get_prediction_history(db_path=db_path)

    assert record_id == 1
    assert len(history) == 1
    assert history[0]["result"] == "FRAUD"
    assert history[0]["model_score"] == 82.5
    assert history[0]["signals"] == ["Amount is unusual"]


def test_clear_prediction_history(tmp_path):
    db_path = tmp_path / "history.db"
    transaction = {
        "amount": 100,
        "hour": 10,
        "user_avg_amount_prior": 100,
        "user_transaction_count_prior": 1,
        "transactions_last_hour": 0,
        "merchant_category": "Food",
        "device_type": "Android",
        "location": "Delhi",
        "is_weekend": 0
    }
    result = {
        "prediction": 0,
        "model_score": 5.0,
        "risk_level": "Low",
        "model": "random_forest",
        "signals": []
    }

    save_prediction(transaction, result, db_path=db_path)
    clear_prediction_history(db_path=db_path)

    assert get_prediction_history(db_path=db_path) == []
