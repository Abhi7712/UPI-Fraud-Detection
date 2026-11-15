
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    average_precision_score
)


# --------------------------------------------------
# PATHS AND CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "raw" / "upi_transactions.csv"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

st.set_page_config(
    page_title="UPI Fraud Intelligence",
    page_icon="💳",
    layout="wide"
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

.block-container {
    padding-top: 1.5rem;
}

.metric-card {
    background-color: white;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# DATA LOADING
# --------------------------------------------------

@st.cache_data
def load_data():

    df = pd.read_csv(DATA_PATH)

    df["transaction_time"] = pd.to_datetime(
        df["transaction_time"],
        errors="coerce"
    )

    df["amount"] = pd.to_numeric(
        df["amount"],
        errors="coerce"
    )

    df["is_fraud"] = pd.to_numeric(
        df["is_fraud"],
        errors="coerce"
    ).fillna(0).astype(int)

    df["is_suspicious"] = (
        df["if_flag_v2"].astype(str).str.lower()
        .isin(["1", "true", "yes"])
    )

    return df


@st.cache_resource
def load_saved_model(model_name):

    model_path = MODEL_DIR / f"{model_name}.joblib"

    if not model_path.exists():
        return None

    saved_data = joblib.load(model_path)

    return saved_data["model"]


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("💳 UPI Fraud Intelligence")

st.sidebar.caption("AI-Powered Transaction Analysis")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Live Prediction",
        "Model Performance",
        "Transactions",
        "Fraud Analysis",
        "About Project"
    ]
)

st.sidebar.divider()

st.sidebar.info(
    "Portfolio project using a synthetic UPI transaction dataset."
)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

try:
    df = load_data()

except Exception as error:

    st.error(f"Dataset loading failed: {error}")
    st.stop()


# --------------------------------------------------
# OVERVIEW
# --------------------------------------------------

if page == "Overview":

    st.title("UPI Fraud Intelligence Dashboard")

    st.caption(
        "Monitor transaction patterns and identify suspicious activity."
    )

    total_transactions = len(df)

    fraud_transactions = int(df["is_fraud"].sum())

    suspicious_transactions = int(
        df["is_suspicious"].sum()
    )

    total_amount = df["amount"].sum()

    fraud_rate = (
        fraud_transactions / total_transactions * 100
        if total_transactions else 0
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Transactions",
        f"{total_transactions:,}"
    )

    col2.metric(
        "Fraud Labels",
        f"{fraud_transactions:,}"
    )

    col3.metric(
        "Anomaly Flags",
        f"{suspicious_transactions:,}"
    )

    col4.metric(
        "Fraud Label Rate",
        f"{fraud_rate:.2f}%"
    )

    st.divider()

    left, right = st.columns(2)

    with left:

        st.subheader("Transaction Amount Distribution")

        fig = px.histogram(
            df,
            x="amount",
            nbins=40,
            title="Transaction Amount"
        )

        st.plotly_chart(fig, width="stretch")

    with right:

        st.subheader("Fraud Label Distribution")

        label_data = pd.DataFrame({
            "Category": ["Legitimate", "Fraud"],
            "Count": [
                int((df["is_fraud"] == 0).sum()),
                fraud_transactions
            ]
        })

        fig = px.pie(
            label_data,
            names="Category",
            values="Count",
            hole=0.5
        )

        st.plotly_chart(fig, width="stretch")

    st.subheader("Transactions by Hour")

    hourly = (
        df.groupby("hour")
        .size()
        .reset_index(name="Transactions")
    )

    fig = px.bar(
        hourly,
        x="hour",
        y="Transactions"
    )

    st.plotly_chart(fig, width="stretch")

    st.subheader("Recent Transactions")

    st.dataframe(
        df.sort_values(
            "transaction_time",
            ascending=False
        ).head(10),
        width="stretch"
    )


# --------------------------------------------------
# LIVE PREDICTION
# --------------------------------------------------

elif page == "Live Prediction":

    st.title("Live Transaction Risk Prediction")

    st.info(
        "Enter historical transaction information and transaction details."
    )

    model_name = st.selectbox(
        "Select Model",
        ["random_forest", "logistic_regression"]
    )

    col1, col2 = st.columns(2)

    with col1:

        amount = st.number_input(
            "Transaction Amount (₹)",
            min_value=0.0,
            value=15000.0
        )

        hour = st.slider(
            "Transaction Hour",
            0, 23, 22
        )

        user_avg = st.number_input(
            "User's Previous Average Amount",
            min_value=0.0,
            value=3000.0
        )

        previous_count = st.number_input(
            "Previous Transaction Count",
            min_value=0,
            value=10
        )

    with col2:

        last_hour = st.number_input(
            "Transactions in Previous Hour",
            min_value=0,
            value=2
        )

        merchant = st.selectbox(
            "Merchant Category",
            sorted(df["merchant_category"].dropna().astype(str).unique())
        )

        device = st.selectbox(
            "Device Type",
            sorted(df["device_type"].dropna().astype(str).unique())
        )

        location = st.selectbox(
            "Location",
            sorted(df["location"].dropna().astype(str).unique())
        )

        weekend = st.selectbox(
            "Is Weekend?",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

    if st.button(
        "Analyze Transaction",
        type="primary",
        width="stretch"
    ):

        model_path = MODEL_DIR / f"{model_name}.joblib"

        if not model_path.exists():

            st.error("Selected model file not found.")

        else:

            from src.predict import predict_transaction

            try:

                result = predict_transaction(
                    amount=amount,
                    hour=hour,
                    user_avg_amount_prior=user_avg,
                    user_transaction_count_prior=previous_count,
                    transactions_last_hour=last_hour,
                    merchant_category=merchant,
                    device_type=device,
                    location=location,
                    is_weekend=weekend,
                    model_name=model_name
                )

                st.divider()

                st.subheader("Prediction Result")

                c1, c2 = st.columns(2)

                c1.metric(
                    "Risk Score",
                    f"{result['risk_score']:.2f}%"
                )

                c2.metric(
                    "Risk Level",
                    result["risk_level"]
                )

                if result["prediction"] == 1:

                    st.error(
                        "⚠️ Model flagged this transaction as potentially fraudulent."
                    )

                else:

                    st.success(
                        "Model did not flag this transaction as fraud."
                    )

                st.caption(
                    "This is a model-generated risk estimate, not a confirmed fraud decision."
                )

            except Exception as error:

                st.error(f"Prediction failed: {error}")


# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------

elif page == "Model Performance":

    st.title("Model Performance & Comparison")

    test_path = BASE_DIR / "data" / "processed" / "test_data.csv"

    if not test_path.exists():

        st.warning(
            "Test dataset not found. Run python -m src.train first."
        )

        st.stop()

    test_df = pd.read_csv(test_path)

    test_df["transaction_time"] = pd.to_datetime(
        test_df["transaction_time"]
    )

    model_names = [
        "logistic_regression",
        "random_forest"
    ]

    results = []

    for model_name in model_names:

        saved_path = MODEL_DIR / f"{model_name}.joblib"

        if not saved_path.exists():
            continue

        saved = joblib.load(saved_path)

        model = saved["model"]

        features = saved["feature_columns"]

        X_test = test_df[features]

        y_test = test_df["is_fraud"].astype(int)

        predictions = model.predict(X_test)

        probabilities = model.predict_proba(X_test)[:, 1]

        results.append({

            "Model": model_name,

            "Accuracy": accuracy_score(
                y_test, predictions
            ),

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

        })

    if not results:

        st.error("No trained models found.")

        st.stop()

    comparison = pd.DataFrame(results)

    st.subheader("Model Comparison")

    st.dataframe(
        comparison.style.format({
            "Accuracy": "{:.2%}",
            "Precision": "{:.2%}",
            "Recall": "{:.2%}",
            "F1 Score": "{:.2%}",
            "PR-AUC": "{:.4f}"
        }),
        width="stretch"
    )

    metric = st.selectbox(
        "Compare Metric",
        [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "PR-AUC"
        ]
    )

    fig = px.bar(
        comparison,
        x="Model",
        y=metric,
        title=f"Model Comparison - {metric}",
        text_auto=".3f"
    )

    st.plotly_chart(fig, width="stretch")

    selected_model = st.selectbox(
        "View Confusion Matrix",
        model_names
    )

    selected_path = MODEL_DIR / f"{selected_model}.joblib"

    saved = joblib.load(selected_path)

    model = saved["model"]

    features = saved["feature_columns"]

    predictions = model.predict(test_df[features])

    matrix = confusion_matrix(
        test_df["is_fraud"],
        predictions,
        labels=[0, 1]
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=["Actual Legitimate", "Actual Fraud"],
        columns=["Predicted Legitimate", "Predicted Fraud"]
    )

    st.subheader("Confusion Matrix")

    st.dataframe(matrix_df, width="stretch")

    st.download_button(
        "Download Model Comparison",
        comparison.to_csv(index=False),
        file_name="model_comparison.csv",
        mime="text/csv"
    )


# --------------------------------------------------
# TRANSACTIONS
# --------------------------------------------------

elif page == "Transactions":

    st.title("Transaction Explorer")

    search = st.text_input(
        "Search Transaction ID"
    )

    locations = sorted(
        df["location"].dropna().astype(str).unique()
    )

    selected_location = st.multiselect(
        "Filter by Location",
        locations
    )

    categories = sorted(
        df["merchant_category"].dropna().astype(str).unique()
    )

    selected_category = st.multiselect(
        "Merchant Category",
        categories
    )

    status = st.selectbox(
        "Fraud Label",
        ["All", "Legitimate", "Fraud"]
    )

    filtered = df.copy()

    if search:

        filtered = filtered[
            filtered["transaction_id"].astype(str)
            .str.contains(search, case=False, na=False)
        ]

    if selected_location:

        filtered = filtered[
            filtered["location"].astype(str)
            .isin(selected_location)
        ]

    if selected_category:

        filtered = filtered[
            filtered["merchant_category"].astype(str)
            .isin(selected_category)
        ]

    if status == "Legitimate":

        filtered = filtered[
            filtered["is_fraud"] == 0
        ]

    elif status == "Fraud":

        filtered = filtered[
            filtered["is_fraud"] == 1
        ]

    st.metric(
        "Filtered Transactions",
        len(filtered)
    )

    st.dataframe(
        filtered,
        width="stretch"
    )

    st.download_button(
        "Download Filtered Transactions",
        filtered.to_csv(index=False),
        file_name="filtered_transactions.csv",
        mime="text/csv"
    )


# --------------------------------------------------
# FRAUD ANALYSIS
# --------------------------------------------------

elif page == "Fraud Analysis":

    st.title("Fraud & Anomaly Analysis")

    fraud_df = df[df["is_fraud"] == 1]

    normal_df = df[df["is_fraud"] == 0]

    c1, c2, c3 = st.columns(3)

    c1.metric("Fraud-labelled Transactions", len(fraud_df))

    c2.metric("Legitimate Transactions", len(normal_df))

    c3.metric(
        "Average Fraud-labelled Amount",
        f"₹{fraud_df['amount'].mean():,.2f}"
        if len(fraud_df) else "₹0"
    )

    st.subheader("Fraud vs Legitimate Amounts")

    amount_data = df.copy()

    amount_data["Label"] = amount_data[
        "is_fraud"
    ].map({
        0: "Legitimate",
        1: "Fraud"
    })

    fig = px.box(
        amount_data,
        x="Label",
        y="amount",
        color="Label"
    )

    st.plotly_chart(fig, width="stretch")

    st.subheader("Anomaly Detection Flags")

    flag_columns = [
        col for col in [
            "iqr_flag",
            "if_flag",
            "if_flag_tuned",
            "if_flag_v2"
        ]
        if col in df.columns
    ]

    if flag_columns:

        flag_summary = pd.DataFrame({

            "Detection Method": flag_columns,

            "Flagged Transactions": [
                pd.to_numeric(
                    df[col], errors="coerce"
                ).fillna(0).astype(int).sum()
                for col in flag_columns
            ]

        })

        fig = px.bar(
            flag_summary,
            x="Detection Method",
            y="Flagged Transactions",
            text="Flagged Transactions"
        )

        st.plotly_chart(fig, width="stretch")

        st.dataframe(
            flag_summary,
            width="stretch"
        )

    st.caption(
        "Anomaly flags are separate from the synthetic is_fraud target label."
    )


# --------------------------------------------------
# ABOUT PROJECT
# --------------------------------------------------

elif page == "About Project":

    st.title("About UPI Fraud Intelligence")

    st.markdown("""
    ### Project Overview

    This project demonstrates a machine learning workflow
    for analyzing synthetic UPI transactions.

    ### Machine Learning Models

    - Logistic Regression
    - Random Forest

    ### Feature Engineering

    - Transaction amount
    - Transaction hour
    - Cyclical time features
    - Historical user average amount
    - Amount deviation
    - Previous transaction count
    - Transactions in the previous hour
    - Merchant category
    - Device type
    - Location

    ### Evaluation Metrics

    - Accuracy
    - Precision
    - Recall
    - F1 Score
    - PR-AUC
    - Confusion Matrix

    ### Limitations

    - Dataset is synthetic.
    - Labels do not represent verified bank fraud.
    - Model predictions are experimental.
    - This project is not intended for production financial decisions.
    """)

    st.info(
        "Developed as a Data Science and Machine Learning portfolio project."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.sidebar.divider()

st.sidebar.caption(
    "UPI Fraud Intelligence | Machine Learning Portfolio Project"
)
