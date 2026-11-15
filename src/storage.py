from pathlib import Path
import json
import sqlite3
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "processed" / "prediction_history.db"


def _connect(db_path=None):
    path = Path(db_path) if db_path else DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_history(db_path=None):
    with _connect(db_path) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                checked_at TEXT NOT NULL,
                amount REAL NOT NULL,
                hour INTEGER NOT NULL,
                user_avg_amount_prior REAL NOT NULL,
                user_transaction_count_prior INTEGER NOT NULL,
                transactions_last_hour INTEGER NOT NULL,
                merchant_category TEXT NOT NULL,
                device_type TEXT NOT NULL,
                location TEXT NOT NULL,
                is_weekend INTEGER NOT NULL,
                model TEXT NOT NULL,
                prediction INTEGER NOT NULL,
                model_score REAL NOT NULL,
                risk_level TEXT NOT NULL,
                signals TEXT NOT NULL
            )
        """)
        connection.commit()


def save_prediction(transaction, result, db_path=None):
    initialize_history(db_path)
    with _connect(db_path) as connection:
        cursor = connection.execute("""
            INSERT INTO prediction_history (
                checked_at, amount, hour, user_avg_amount_prior,
                user_transaction_count_prior, transactions_last_hour,
                merchant_category, device_type, location, is_weekend,
                model, prediction, model_score, risk_level, signals
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().astimezone().isoformat(timespec="seconds"),
            float(transaction["amount"]),
            int(transaction["hour"]),
            float(transaction["user_avg_amount_prior"]),
            int(transaction["user_transaction_count_prior"]),
            int(transaction["transactions_last_hour"]),
            str(transaction["merchant_category"]),
            str(transaction["device_type"]),
            str(transaction["location"]),
            int(transaction["is_weekend"]),
            str(result["model"]),
            int(result["prediction"]),
            float(result["model_score"]),
            str(result["risk_level"]),
            json.dumps(result.get("signals", []), ensure_ascii=False)
        ))
        connection.commit()
        return cursor.lastrowid


def get_prediction_history(limit=500, db_path=None):
    initialize_history(db_path)
    with _connect(db_path) as connection:
        rows = connection.execute("""
            SELECT * FROM prediction_history
            ORDER BY id DESC
            LIMIT ?
        """, (int(limit),)).fetchall()
    records = [dict(row) for row in rows]
    for record in records:
        record["result"] = "FRAUD" if record["prediction"] == 1 else "NOT FRAUD"
        record["signals"] = json.loads(record["signals"])
    return records


def clear_prediction_history(db_path=None):
    initialize_history(db_path)
    with _connect(db_path) as connection:
        connection.execute("DELETE FROM prediction_history")
        connection.commit()
