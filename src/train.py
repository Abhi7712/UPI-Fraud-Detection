
from pathlib import Path
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest

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

CONTAMINATION_VALUES = [0.02, 0.03, 0.04, 0.05, 0.08]


def build_preprocessor():

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES)
    ])


def build_supervised_models():

    return {
        "logistic_regression": Pipeline([
            ("preprocessor", build_preprocessor()),
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


def save_model(model, model_name, training_cutoff):

    model_path = MODEL_DIR / f"{model_name}.joblib"

    joblib.dump({
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "training_cutoff": str(training_cutoff),
        "model_name": model_name
    }, model_path)

    print("Saved:", model_path)


def main():

    print("Loading transaction dataset...")

    df = prepare_dataset(DATA_PATH)
    df = df.sort_values("transaction_time").reset_index(drop=True)

    split_index = int(len(df) * 0.80)

    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]

    print("\nDataset Summary")
    print("-------------------------")
    print("Total rows:", len(df))
    print("Training rows:", len(train_df))
    print("Testing rows:", len(test_df))
    print("Training fraud cases:", int(y_train.sum()))
    print("Testing fraud cases:", int(test_df[TARGET_COLUMN].sum()))

    if y_train.nunique() < 2:
        raise ValueError("Training data must contain both fraud classes.")

    # Train supervised models
    supervised_models = build_supervised_models()

    for model_name, model in supervised_models.items():

        print(f"\nTraining {model_name}...")

        model.fit(X_train, y_train)

        save_model(
            model,
            model_name,
            train_df["transaction_time"].max()
        )

    # Train Isolation Forest with multiple contamination values
    print("\nTraining Isolation Forest variants...")

    normal_data = X_train[y_train == 0]

    print("Legitimate training transactions:", len(normal_data))

    for contamination in CONTAMINATION_VALUES:

        model_name = f"isolation_forest_{str(contamination).replace('.', '_')}"

        print(
            f"\nTraining {model_name} "
            f"(contamination={contamination})..."
        )

        model = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", IsolationForest(
                n_estimators=300,
                contamination=contamination,
                random_state=42,
                n_jobs=-1
            ))
        ])

        model.fit(normal_data)

        save_model(
            model,
            model_name,
            train_df["transaction_time"].max()
        )

    # Save test data
    test_path = BASE_DIR / "data" / "processed" / "test_data.csv"

    test_df.to_csv(test_path, index=False)

    print("\nTest data saved:", test_path)
    print("\nAll models trained successfully!")


if __name__ == "__main__":
    main()
