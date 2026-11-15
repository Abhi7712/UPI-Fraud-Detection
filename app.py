
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

    required_columns = [
        "transaction_id", "user_id", "amount", "transaction_time",
        "merchant_category", "device_type", "location", "is_fraud"
    ]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    df = df.dropna(subset=["transaction_time", "amount", "is_fraud"]).copy()
    df["is_fraud"] = df["is_fraud"].astype(int)
    if not df["is_fraud"].isin([0, 1]).all():
        raise ValueError("is_fraud must contain only 0 and 1.")

    # Derive hour if an input CSV does not already contain it.
    if "hour" not in df.columns:
        df["hour"] = df["transaction_time"].dt.hour

    # Keep anomaly flags separate from the fraud target label.
    if "if_flag_v2" in df.columns:
        df["is_suspicious"] = (
            df["if_flag_v2"].astype(str).str.lower()
            .isin(["1", "true", "yes"])
        )
    else:
        df["is_suspicious"] = False

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
        "Check Transaction",
        "Bulk CSV Detection",
        "Prediction History",
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
# CHECK SINGLE TRANSACTION
# --------------------------------------------------

elif page == "Check Transaction":

    st.title("Check a UPI Transaction")
    st.write(
        "Enter a transaction and the user's historical activity. "
        "The model will classify it as FRAUD or NOT FRAUD."
    )

    st.warning(
        "Demonstration only: this project uses synthetic data and is not connected "
        "to a bank, UPI app, or live payment network."
    )

    model_name = st.selectbox(
        "Machine Learning Model",
        ["random_forest", "logistic_regression"]
    )

    with st.form("single_transaction_form"):
        left, right = st.columns(2)

        with left:
            amount = st.number_input(
                "Transaction Amount (₹)", min_value=0.0,
                value=1500.0, step=100.0
            )
            hour = st.slider("Transaction Hour", 0, 23, 14)
            user_avg = st.number_input(
                "User's Previous Average Amount (₹)",
                min_value=0.0, value=1200.0, step=100.0
            )
            previous_count = st.number_input(
                "Previous Transaction Count",
                min_value=0, value=10, step=1
            )

        with right:
            last_hour = st.number_input(
                "Transactions in Previous Hour",
                min_value=0, value=1, step=1
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
                "Is Weekend?", [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

        submitted = st.form_submit_button(
            "Check Transaction", type="primary", use_container_width=True
        )

    if submitted:
        from src.predict import predict_transaction
        from src.storage import save_prediction

        transaction = {
            "amount": amount,
            "hour": hour,
            "user_avg_amount_prior": user_avg,
            "user_transaction_count_prior": previous_count,
            "transactions_last_hour": last_hour,
            "merchant_category": merchant,
            "device_type": device,
            "location": location,
            "is_weekend": weekend
        }

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
            history_id = save_prediction(transaction, result)

            st.session_state["last_prediction"] = result
            st.session_state["last_prediction_id"] = history_id

        except Exception as error:
            st.error(f"Prediction failed: {error}")
            st.stop()

    result = st.session_state.get("last_prediction")
    if result:
        st.divider()
        st.subheader("Transaction Decision")

        if result["prediction"] == 1:
            st.error("🚨 FRAUD — The model flagged this transaction as potentially fraudulent.")
        else:
            st.success("✅ NOT FRAUD — The model did not flag this transaction as fraud.")

        c1, c2, c3 = st.columns(3)
        c1.metric("Model Score (uncalibrated)", f"{result['model_score']:.2f}%")
        c2.metric("Risk Band", result["risk_level"])
        c3.metric("Model", result["model"].replace("_", " ").title())

        st.caption(
            "Model score is not a calibrated fraud probability. "
            "The model's decision and the illustrative risk band are different outputs."
        )

        st.subheader("Why was it flagged?")
        for signal in result["signals"]:
            st.warning(signal)

        st.caption(
            f"Saved to local prediction history as record #{st.session_state.get('last_prediction_id', '—')}."
        )


# --------------------------------------------------
# BULK CSV DETECTION
# --------------------------------------------------

elif page == "Bulk CSV Detection":

    st.title("Bulk UPI Transaction Detection")
    st.write("Upload a CSV containing one transaction per row to classify multiple transactions.")

    required_input_columns = [
        "amount",
        "hour",
        "user_avg_amount_prior",
        "user_transaction_count_prior",
        "transactions_last_hour",
        "merchant_category",
        "device_type",
        "location",
        "is_weekend"
    ]

    template = pd.DataFrame([{
        "amount": 2500,
        "hour": 14,
        "user_avg_amount_prior": 1200,
        "user_transaction_count_prior": 10,
        "transactions_last_hour": 1,
        "merchant_category": "Shopping",
        "device_type": "Android",
        "location": "Delhi",
        "is_weekend": 0
    }])

    st.download_button(
        "Download CSV Template",
        template.to_csv(index=False),
        file_name="upi_transaction_input_template.csv",
        mime="text/csv"
    )

    model_name = st.selectbox(
        "Select Model for Bulk Detection",
        ["random_forest", "logistic_regression"],
        key="bulk_model"
    )

    uploaded_file = st.file_uploader(
        "Upload transaction CSV",
        type=["csv"]
    )

    if uploaded_file is not None:
        try:
            upload_df = pd.read_csv(uploaded_file)
            missing = [c for c in required_input_columns if c not in upload_df.columns]

            if missing:
                st.error(f"Missing required columns: {', '.join(missing)}")
            elif upload_df.empty:
                st.warning("The uploaded CSV has no transaction rows.")
            else:
                st.dataframe(upload_df.head(10), use_container_width=True)
                st.caption(f"{len(upload_df):,} transaction rows found.")

                if st.button("Analyze All Transactions", type="primary"):
                    from src.predict import predict_transaction
                    from src.storage import save_prediction

                    output_rows = []
                    progress = st.progress(0)

                    for index, row in upload_df.iterrows():
                        try:
                            transaction = {
                                "amount": float(row["amount"]),
                                "hour": float(row["hour"]),
                                "user_avg_amount_prior": float(row["user_avg_amount_prior"]),
                                "user_transaction_count_prior": float(row["user_transaction_count_prior"]),
                                "transactions_last_hour": float(row["transactions_last_hour"]),
                                "merchant_category": row["merchant_category"],
                                "device_type": row["device_type"],
                                "location": row["location"],
                                "is_weekend": float(row["is_weekend"])
                            }

                            prediction = predict_transaction(
                                **transaction,
                                model_name=model_name
                            )
                            save_prediction(transaction, prediction)

                            output_rows.append({
                                **transaction,
                                "prediction": "FRAUD" if prediction["prediction"] == 1 else "NOT FRAUD",
                                "model_score_percent": prediction["model_score"],
                                "risk_level": prediction["risk_level"],
                                "signals": " | ".join(prediction["signals"]),
                                "error": ""
                            })
                        except Exception as row_error:
                            output_rows.append({
                                **row.to_dict(),
                                "prediction": "NOT PROCESSED",
                                "model_score_percent": None,
                                "risk_level": "Unavailable",
                                "signals": "",
                                "error": str(row_error)
                            })

                        progress.progress((index + 1) / len(upload_df))

                    result_df = pd.DataFrame(output_rows)
                    st.session_state["bulk_prediction_results"] = result_df

                result_df = st.session_state.get("bulk_prediction_results")
                if result_df is not None:
                    fraud_count = int((result_df["prediction"] == "FRAUD").sum())
                    normal_count = int((result_df["prediction"] == "NOT FRAUD").sum())
                    failed_count = int((result_df["prediction"] == "NOT PROCESSED").sum())

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Flagged as Fraud", fraud_count)
                    c2.metric("Not Flagged", normal_count)
                    c3.metric("Processing Errors", failed_count)

                    st.dataframe(result_df, use_container_width=True, hide_index=True)
                    st.download_button(
                        "Download Prediction Results",
                        result_df.to_csv(index=False),
                        file_name="upi_fraud_predictions.csv",
                        mime="text/csv"
                    )

        except Exception as error:
            st.error(f"Could not read CSV: {error}")


# --------------------------------------------------
# PREDICTION HISTORY
# --------------------------------------------------

elif page == "Prediction History":

    st.title("Prediction History")
    st.caption("Predictions made through the single-transaction and bulk-checker pages are stored locally.")

    from src.storage import get_prediction_history, clear_prediction_history

    history = get_prediction_history(limit=1000)

    if not history:
        st.info("No predictions saved yet. Use Check Transaction or Bulk CSV Detection first.")
    else:
        history_df = pd.DataFrame(history)

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Checked", len(history_df))
        c2.metric("Flagged as Fraud", int((history_df["prediction"] == 1).sum()))
        c3.metric("Not Flagged", int((history_df["prediction"] == 0).sum()))

        display_cols = [
            "id", "checked_at", "amount", "result", "model",
            "model_score", "risk_level", "merchant_category",
            "device_type", "location"
        ]
        st.dataframe(
            history_df[display_cols],
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download Prediction History",
            history_df.drop(columns=["signals"], errors="ignore").to_csv(index=False),
            file_name="upi_prediction_history.csv",
            mime="text/csv"
        )

        if st.button("Clear Local Prediction History", type="secondary"):
            clear_prediction_history()
            st.rerun()


# MODEL PERFORMANCE
# --------------------------------------------------

elif page == "Model Performance":

    st.title("Model Performance & Comparison")

    st.caption(
        "Compare supervised classification and unsupervised anomaly detection models."
    )

    comparison_path = REPORT_DIR / "model_comparison.csv"
    matrix_path = REPORT_DIR / "confusion_matrices.csv"

    if not comparison_path.exists() or not matrix_path.exists():

        st.warning(
            "Evaluation reports not found. Run: python -m src.evaluate"
        )

        st.stop()

    comparison = pd.read_csv(comparison_path)
    matrices_df = pd.read_csv(matrix_path)

    if comparison.empty:

        st.error("Model comparison report is empty.")
        st.stop()

    # -------------------------------
    # MODEL COMPARISON TABLE
    # -------------------------------

    st.subheader("Model Comparison")

    display_columns = [
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "PR-AUC",
        "Fraud Detected",
        "False Positives",
        "Missed Fraud"
    ]

    comparison = comparison[display_columns]

    st.dataframe(
        comparison.style.format({
            "Accuracy": "{:.2%}",
            "Precision": "{:.2%}",
            "Recall": "{:.2%}",
            "F1 Score": "{:.2%}",
            "PR-AUC": "{:.4f}"
        }),
        width="stretch",
        hide_index=True
    )

    # -------------------------------
    # BEST SUPERVISED MODEL
    # -------------------------------

    supervised = comparison[
        comparison["Model"].isin([
            "logistic_regression",
            "random_forest"
        ])
    ]

    if not supervised.empty:

        best_model = supervised.sort_values(
            "F1 Score",
            ascending=False
        ).iloc[0]

        st.subheader("Best Supervised Model")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Model",
            best_model["Model"].replace("_", " ").title()
        )

        c2.metric(
            "F1 Score",
            f"{best_model['F1 Score']:.2%}"
        )

        c3.metric(
            "Precision",
            f"{best_model['Precision']:.2%}"
        )

    st.divider()

    # -------------------------------
    # METRIC COMPARISON CHART
    # -------------------------------

    st.subheader("Performance Metrics")

    metric = st.selectbox(
        "Select Evaluation Metric",
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
        color="Model",
        text_auto=".3f",
        title=f"Model Comparison - {metric}"
    )

    fig.update_layout(
        xaxis_title="Machine Learning Model",
        yaxis_title=metric,
        showlegend=False,
        xaxis_tickangle=-30
    )

    st.plotly_chart(fig, width="stretch")

    # -------------------------------
    # ISOLATION FOREST ANALYSIS
    # -------------------------------

    st.divider()

    st.subheader("Isolation Forest - Contamination Analysis")

    isolation_df = comparison[
        comparison["Model"].str.startswith(
            "isolation_forest",
            na=False
        )
    ].copy()

    if not isolation_df.empty:

        isolation_df["Contamination"] = (
            isolation_df["Model"]
            .str.replace("isolation_forest_", "", regex=False)
            .str.replace("_", ".", regex=False)
            .astype(float)
        )

        isolation_df = isolation_df.sort_values(
            "Contamination"
        )

        selected_if_metric = st.selectbox(
            "Select Isolation Forest Metric",
            [
                "Precision",
                "Recall",
                "F1 Score",
                "Accuracy",
                "False Positives"
            ],
            key="if_metric"
        )

        fig = px.line(
            isolation_df,
            x="Contamination",
            y=selected_if_metric,
            markers=True,
            text=selected_if_metric,
            title=f"Contamination vs {selected_if_metric}"
        )

        fig.update_traces(textposition="top center")

        st.plotly_chart(fig, width="stretch")

        st.dataframe(
            isolation_df[
                [
                    "Contamination",
                    "Precision",
                    "Recall",
                    "F1 Score",
                    "PR-AUC",
                    "Fraud Detected",
                    "False Positives",
                    "Missed Fraud"
                ]
            ].style.format({
                "Contamination": "{:.2f}",
                "Precision": "{:.2%}",
                "Recall": "{:.2%}",
                "F1 Score": "{:.2%}",
                "PR-AUC": "{:.4f}"
            }),
            width="stretch",
            hide_index=True
        )

        st.info(
            "Higher contamination increases the number of transactions "
            "flagged as anomalies. It can improve recall but may also "
            "increase false positives. PR-AUC measures ranking quality "
            "and is not improved simply by changing the prediction threshold."
        )

    # -------------------------------
    # CONFUSION MATRIX
    # -------------------------------

    st.divider()

    st.subheader("Confusion Matrix")

    available_models = matrices_df["Model"].tolist()

    selected_model = st.selectbox(
        "Select Model for Confusion Matrix",
        available_models,
        key="matrix_model"
    )

    selected_matrix = matrices_df[
        matrices_df["Model"] == selected_model
    ].iloc[0]

    matrix = np.array([
        [selected_matrix["TN"], selected_matrix["FP"]],
        [selected_matrix["FN"], selected_matrix["TP"]]
    ])

    matrix_df = pd.DataFrame(
        matrix.astype(int),
        index=["Actual Legitimate", "Actual Fraud"],
        columns=["Predicted Legitimate", "Predicted Fraud"]
    )

    st.dataframe(
        matrix_df,
        width="stretch"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("True Negatives", int(selected_matrix["TN"]))
    col2.metric("False Positives", int(selected_matrix["FP"]))
    col3.metric("False Negatives", int(selected_matrix["FN"]))
    col4.metric("True Positives", int(selected_matrix["TP"]))

    # -------------------------------
    # DOWNLOAD REPORT
    # -------------------------------

    st.divider()

    st.subheader("Download Evaluation Reports")

    col1, col2 = st.columns(2)

    with col1:

        st.download_button(
            "Download Model Comparison",
            comparison.to_csv(index=False),
            file_name="model_comparison.csv",
            mime="text/csv",
            width="stretch"
        )

    with col2:

        st.download_button(
            "Download Confusion Matrices",
            matrices_df.to_csv(index=False),
            file_name="confusion_matrices.csv",
            mime="text/csv",
            width="stretch"
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

    - Logistic Regression (Supervised Learning)
    - Random Forest (Supervised Learning)
    - Isolation Forest (Unsupervised Anomaly Detection)

    ### Model Evaluation

    - Accuracy
    - Precision
    - Recall
    - F1 Score
    - PR-AUC
    - Confusion Matrix
    - Contamination Analysis

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
    - Isolation Forest showed limited ability to rank synthetic fraud labels.
    - Contamination changes the anomaly detection threshold and false-positive trade-off.
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
