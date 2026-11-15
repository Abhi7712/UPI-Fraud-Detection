
import pandas as pd
import numpy as np


TARGET_COLUMN = "is_fraud"

NUMERIC_FEATURES = [
    "amount",
    "hour",
    "hour_sin",
    "hour_cos",
    "is_weekend",
    "user_avg_amount_prior",
    "amount_deviation",
    "user_transaction_count_prior",
    "transactions_last_hour"
]

CATEGORICAL_FEATURES = [
    "merchant_category",
    "device_type",
    "location"
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_transaction_data(file_path):
    """Load and validate the transaction dataset."""

    df = pd.read_csv(file_path)

    required_columns = [
        "transaction_id",
        "user_id",
        "amount",
        "transaction_time",
        "merchant_category",
        "device_type",
        "location",
        TARGET_COLUMN
    ]

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["transaction_time"] = pd.to_datetime(
        df["transaction_time"],
        errors="coerce"
    )

    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df[TARGET_COLUMN] = pd.to_numeric(
        df[TARGET_COLUMN],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "transaction_id", "user_id", "transaction_time",
            "amount", TARGET_COLUMN
        ]
    ).copy()

    if not np.isfinite(df["amount"].to_numpy(dtype=float)).all():
        raise ValueError("Transaction amounts must be finite numbers.")

    if not np.isfinite(df[TARGET_COLUMN].to_numpy(dtype=float)).all():
        raise ValueError("is_fraud must contain finite numeric values.")

    if not df[TARGET_COLUMN].isin([0, 1]).all():
        raise ValueError("is_fraud must contain only 0 and 1.")

    if (df["amount"] < 0).any():
        raise ValueError("Transaction amounts cannot be negative.")

    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    return df


def create_features(df):
    """
    Create leakage-free transaction features.

    Every historical feature uses only transactions
    that occurred before the current transaction.
    """

    df = df.copy()

    df["transaction_time"] = pd.to_datetime(
        df["transaction_time"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["transaction_time", "amount", "user_id"]
    )

    df = df.sort_values(
        ["transaction_time", "transaction_id"]
    ).reset_index(drop=True)

    # Time-based features
    df["hour"] = df["transaction_time"].dt.hour

    df["hour_sin"] = np.sin(
        2 * np.pi * df["hour"] / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * df["hour"] / 24
    )

    df["is_weekend"] = (
        df["transaction_time"].dt.day_of_week >= 5
    ).astype(int)

    # Historical user features
    df["user_avg_amount_prior"] = (
        df.groupby("user_id")["amount"]
        .transform(lambda s: s.shift(1).expanding().mean())
    )

    df["user_transaction_count_prior"] = (
        df.groupby("user_id").cumcount()
    )

    # For a user's first transaction, historical average is unknown.
    # Use the current dataset's training-independent neutral fallback.
    # This value is only a fallback, not a learned user statistic.
    df["user_avg_amount_prior"] = (
        df["user_avg_amount_prior"].fillna(0)
    )

    df["amount_deviation"] = (
        df["amount"] - df["user_avg_amount_prior"]
    )


    # Count same user's transactions in the previous 60 minutes
    
    # Count same user's transactions in the previous 60 minutes
    previous_hour_counts = np.zeros(len(df), dtype=int)

    for _, group in df.groupby("user_id", sort=False):
        # Explicitly convert timestamps to nanosecond precision
        times = group["transaction_time"].to_numpy(
            dtype="datetime64[ns]"
        )

        # Start of the previous 60-minute window
        window_starts = times - np.timedelta64(1, "h")

        # Find earlier transactions strictly newer than the window start
        left_positions = np.searchsorted(
            times,
            window_starts,
            side="right"
        )

        counts = np.arange(len(times)) - left_positions

        previous_hour_counts[group.index.to_numpy()] = counts

    df["transactions_last_hour"] = previous_hour_counts


    # Missing categorical values
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna("Unknown").astype(str)

    return df


def get_model_features(df):
    """Return model input features and target."""

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    return X, y


def prepare_dataset(file_path):
    """Load raw data and create model-ready features."""

    df = load_transaction_data(file_path)
    df = create_features(df)

    return df
