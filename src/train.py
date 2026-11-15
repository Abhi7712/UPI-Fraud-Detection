
from pathlib import Path
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.feature_engineering import (
    prepare_dataset,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN
)


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "upi_transactions.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


def build_preprocessor():

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES)
    ])

    return preprocessor


def build_models():

    preprocessor = build_preprocessor()

    models = {
        "logistic_regression": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42
            ))
        ]),

        "random_forest": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                min_samples_split=5,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            ))
        ])
    }

    return models


def main():

    print("Loading transaction dataset...")

    df = prepare_dataset(DATA_PATH)

    df = df.sort_values("transaction_time").reset_index(drop=True)

    split_index = int(len(df) * 0.80)

    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    print("\nDataset Summary")
    print("-------------------------")
    print("Total rows:", len(df))
    print("Training rows:", len(train_df))
    print("Testing rows:", len(test_df))
    print("Training fraud cases:", int(y_train.sum()))
    print("Testing fraud cases:", int(y_test.sum()))

    if y_train.nunique() < 2:
        raise ValueError("Training data must contain both fraud classes.")

    models = build_models()

    for model_name, model in models.items():

        print(f"\nTraining {model_name}...")

        model.fit(X_train, y_train)

        model_path = MODEL_DIR / f"{model_name}.joblib"

        joblib.dump({
            "model": model,
            "feature_columns": FEATURE_COLUMNS,
            "training_cutoff": str(
                train_df["transaction_time"].max()
            ),
            "model_name": model_name
        }, model_path)

        print("Saved:", model_path)

    # Save the exact chronological test set for evaluation.
    test_path = BASE_DIR / "data" / "processed" / "test_data.csv"

    test_df.to_csv(test_path, index=False)

    print("\nTest data saved:", test_path)
    print("\nModel training completed successfully!")


if __name__ == "__main__":
    main()
