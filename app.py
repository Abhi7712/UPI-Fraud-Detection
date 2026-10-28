import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

from pathlib import Path

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="UPI Fraud Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "upi_transactions.csv"
MODEL_PATH = BASE_DIR / "upi_fraud_model.joblib"

# ---------------- PROFESSIONAL UI ----------------

st.markdown("""
<style>
    .stApp {
        background-color: #0b1120;
        color: #f1f5f9;
    }

    [data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #263247;
    }

    .main-title {
        font-size: 34px;
        font-weight: 800;
        color: #e2e8f0;
        margin-bottom: 4px;
    }

    .subtitle {
        color: #94a3b8;
        font-size: 15px;
        margin-bottom: 25px;
    }

    .metric-card {
        background: linear-gradient(135deg, #172033, #111827);
        padding: 20px;
        border-radius: 14px;
        border: 1px solid #263247;
        margin-bottom: 10px;
    }

    .metric-label {
        color: #94a3b8;
        font-size: 14px;
    }

    .metric-value {
        color: #f8fafc;
        font-size: 27px;
        font-weight: 800;
        margin-top: 7px;
    }

    .section-title {
        color: #e2e8f0;
        font-size: 21px;
        font-weight: 700;
        margin-top: 18px;
        margin-bottom: 12px;
    }

    div[data-testid="stMetric"] {
        background-color: #111827;
        border: 1px solid #263247;
        padding: 16px;
        border-radius: 12px;
    }

    .stButton button {
        border-radius: 9px;
        font-weight: 600;
    }

    .notice {
        background-color: #172033;
        border-left: 4px solid #38bdf8;
        padding: 13px;
        border-radius: 6px;
        color: #cbd5e1;
    }
</style>
""", unsafe_allow_html=True)


# ---------------- LOAD DATA AND MODEL ----------------

@st.cache_data
def load_data(path, modified_time):
    data = pd.read_csv(path)

    if "transaction_time" in data.columns:
        data["transaction_time"] = pd.to_datetime(
            data["transaction_time"], errors="coerce"
        )

    if "amount" in data.columns:
        data["amount"] = pd.to_numeric(
            data["amount"], errors="coerce"
        )

    if "is_suspicious" not in data.columns:
        if "if_flag_v2" in data.columns:
            data["is_suspicious"] = data["if_flag_v2"]
        else:
            data["is_suspicious"] = False

    data["is_suspicious"] = (
        data["is_suspicious"]
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    return data


@st.cache_resource
def load_model(path, modified_time):
    return joblib.load(path)


# ---------------- SIDEBAR ----------------

st.sidebar.markdown("# 🛡️ UPI Intelligence")
st.sidebar.caption("Fraud & Anomaly Monitoring")
st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Overview",
        "Live Prediction",
        "Model Performance",
        "Transactions",
        "About Project"
    ]
)

st.sidebar.divider()
st.sidebar.caption("Machine Learning: Isolation Forest")
st.sidebar.caption("Dataset: Synthetic transactions")


# ---------------- CHECK FILES ----------------

if not DATA_PATH.exists():
    st.error("Dataset file nahi mili: upi_transactions.csv")
    st.info("CSV file ko app.py ke same folder mein rakho.")
    st.stop()

try:
    df = load_data(
        str(DATA_PATH),
        DATA_PATH.stat().st_mtime
    )
except Exception as error:
    st.error(f"Dataset load nahi hui: {error}")
    st.stop()

if MODEL_PATH.exists():
    try:
        model = load_model(
            str(MODEL_PATH),
            MODEL_PATH.stat().st_mtime
        )
    except Exception as error:
        model = None
        st.error(f"Model load nahi hua: {error}")
else:
    model = None


# ---------------- HELPER FUNCTIONS ----------------

def money(value):
    return f"₹{value:,.2f}"


def metric_card(label, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def chart_layout(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b1120",
        plot_bgcolor="#111827",
        font_color="#e2e8f0",
        margin=dict(l=20, r=20, t=45, b=20),
        legend_title_text=""
    )
    return fig


# ---------------- PAGE: OVERVIEW ----------------

if page == "Overview":

    st.markdown(
        '<div class="main-title">🛡️ UPI Fraud Intelligence</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="subtitle">'
        'Transaction monitoring, anomaly analytics and risk insights'
        '</div>',
        unsafe_allow_html=True
    )

    total = len(df)

    if "is_suspicious" in df.columns:
        suspicious = int(df["is_suspicious"].sum())
    else:
        suspicious = 0

    normal = total - suspicious
    rate = (suspicious / total * 100) if total else 0

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card("TOTAL TRANSACTIONS", f"{total:,}")

    with c2:
        metric_card("FLAGGED ANOMALIES", f"{suspicious:,}")

    with c3:
        metric_card("OTHER TRANSACTIONS", f"{normal:,}")

    with c4:
        metric_card("FLAGGED RATE", f"{rate:.2f}%")

    st.markdown(
        '<div class="section-title">📊 Transaction Analytics</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        if "amount" in df.columns:
            fig = px.histogram(
                df,
                x="amount",
                color="is_suspicious",
                nbins=35,
                title="Transaction Amount Distribution",
                labels={
                    "amount": "Amount (₹)",
                    "is_suspicious": "Flagged anomaly"
                },
                color_discrete_map={
                    True: "#fb7185",
                    False: "#38bdf8"
                }
            )
            st.plotly_chart(chart_layout(fig), width="stretch")
        else:
            st.info("Dataset mein amount column nahi hai.")

    with col2:
        pie_df = pd.DataFrame({
            "Status": ["Flagged anomaly", "Other transactions"],
            "Count": [suspicious, normal]
        })

        fig = px.pie(
            pie_df,
            names="Status",
            values="Count",
            hole=0.65,
            title="Transaction Status Breakdown",
            color="Status",
            color_discrete_map={
                "Flagged anomaly": "#fb7185",
                "Other transactions": "#38bdf8"
            }
        )
        st.plotly_chart(chart_layout(fig), width="stretch")

    col3, col4 = st.columns(2)

    with col3:
        if "location" in df.columns:
            location_df = (
                df.groupby("location", dropna=False)["is_suspicious"]
                .sum()
                .reset_index(name="Flagged Count")
                .sort_values("Flagged Count", ascending=False)
                .head(10)
            )

            fig = px.bar(
                location_df,
                x="location",
                y="Flagged Count",
                title="Flagged Anomalies by Location",
                color="Flagged Count",
                color_continuous_scale="Reds"
            )
            st.plotly_chart(chart_layout(fig), width="stretch")
        else:
            st.info("Location column available nahi hai.")

    with col4:
        if "merchant_category" in df.columns:
            merchant_df = (
                df.groupby("merchant_category", dropna=False)
                ["is_suspicious"]
                .sum()
                .reset_index(name="Flagged Count")
                .sort_values("Flagged Count", ascending=False)
            )

            fig = px.bar(
                merchant_df,
                x="merchant_category",
                y="Flagged Count",
                title="Flagged Anomalies by Merchant",
                color="Flagged Count",
                color_continuous_scale="Blues"
            )
            st.plotly_chart(chart_layout(fig), width="stretch")
        else:
            st.info("Merchant category column available nahi hai.")

    st.markdown(
        """
        <div class="notice">
        <b>Important:</b> This project uses synthetic transaction data.
        Flagged anomalies are model alerts, not confirmed financial fraud.
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------------- ADVANCED FRAUD INSIGHTS ----------------

    st.divider()

    st.markdown(
        '<div class="section-title">🔍 Advanced Fraud Insights</div>',
        unsafe_allow_html=True
    )

    # 1. Actual labelled fraud vs normal transactions
    if "is_fraud" in df.columns:

        fraud_data = df.copy()
        fraud_data["is_fraud"] = pd.to_numeric(
            fraud_data["is_fraud"], errors="coerce"
        ).fillna(0).astype(int)

        fraud_summary = (
            fraud_data["is_fraud"]
            .map({0: "Normal", 1: "Fraud"})
            .value_counts()
            .rename_axis("Transaction Type")
            .reset_index(name="Count")
        )

        col1, col2 = st.columns(2)

        with col1:
            fig = px.bar(
                fraud_summary,
                x="Transaction Type",
                y="Count",
                color="Transaction Type",
                title="Actual Label Distribution",
                color_discrete_map={
                    "Normal": "#38bdf8",
                    "Fraud": "#fb7185"
                }
            )
            st.plotly_chart(
                chart_layout(fig),
                use_container_width=True
            )

        # 2. Transaction amount comparison
        with col2:
            if "amount" in fraud_data.columns:
                fig = px.box(
                    fraud_data,
                    x="is_fraud",
                    y="amount",
                    color="is_fraud",
                    title="Transaction Amount: Normal vs Fraud",
                    labels={
                        "is_fraud": "Actual Label (0=Normal, 1=Fraud)",
                        "amount": "Transaction Amount (₹)"
                    },
                    color_discrete_map={
                        0: "#38bdf8",
                        1: "#fb7185"
                    }
                )
                st.plotly_chart(
                    chart_layout(fig),
                    use_container_width=True
                )

    else:
        st.info(
            "Actual fraud analysis ke liye CSV mein is_fraud column chahiye."
        )

    # 3. Hourly transaction trend
    if "hour" in df.columns:

        hourly_data = df.copy()
        hourly_data["hour"] = pd.to_numeric(
            hourly_data["hour"], errors="coerce"
        )
        hourly_data = hourly_data.dropna(subset=["hour"])

        if not hourly_data.empty:
            hourly_summary = (
                hourly_data.groupby("hour")
                .size()
                .reset_index(name="Transactions")
                .sort_values("hour")
            )

            fig = px.line(
                hourly_summary,
                x="hour",
                y="Transactions",
                markers=True,
                title="Transaction Activity by Hour",
                labels={
                    "hour": "Hour of Day",
                    "Transactions": "Number of Transactions"
                }
            )

            st.plotly_chart(
                chart_layout(fig),
                use_container_width=True
            )

    # 4. High-value flagged transactions
    if "amount" in df.columns and "is_suspicious" in df.columns:

        st.subheader("🚨 High-Value Flagged Transactions")

        flagged_data = df[df["is_suspicious"]].copy()

        if not flagged_data.empty:

            flagged_data = flagged_data.sort_values(
                "amount", ascending=False
            ).head(10)

            display_columns = [
                col for col in [
                    "transaction_id",
                    "amount",
                    "transaction_time",
                    "merchant_category",
                    "device_type",
                    "location"
                ]
                if col in flagged_data.columns
            ]

            st.dataframe(
                flagged_data[display_columns],
                use_container_width=True,
                hide_index=True
            )

        else:
            st.info("Dataset mein koi flagged transaction nahi mili.")

    st.caption(
        "Note: Dataset synthetic hai. Flagged transactions model alerts hain, "
        "confirmed financial fraud nahi."
    )
    
# ---------------- PAGE: LIVE PREDICTION ----------------

elif page == "Live Prediction":

    st.markdown(
        '<div class="main-title">🤖 Live Anomaly Prediction</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="subtitle">'
        'Enter transaction details to evaluate an anomaly score.'
        '</div>',
        unsafe_allow_html=True
    )

    if model is None:
        st.error(
            "Trained model load nahi hua. "
            "upi_fraud_model.joblib ko app.py ke same folder mein rakho."
        )
        st.stop()

    st.info(
        "Prediction ke liye transaction amount, hour, "
        "user average amount aur last-hour transaction count chahiye."
    )

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        with col1:
            amount = st.number_input(
                "Transaction Amount (₹)",
                min_value=0.01,
                max_value=10000000.0,
                value=2500.0,
                step=100.0
            )

            hour = st.slider(
                "Transaction Hour",
                min_value=0,
                max_value=23,
                value=14
            )

        with col2:
            user_avg_amount = st.number_input(
                "User's Average Transaction Amount (₹)",
                min_value=0.0,
                max_value=10000000.0,
                value=1500.0,
                step=100.0
            )

            transactions_last_hour = st.number_input(
                "Transactions by User in Last Hour",
                min_value=0,
                max_value=1000,
                value=1,
                step=1
            )

        submitted = st.form_submit_button(
            "🔍 Analyze Transaction",
            width="stretch"
        )

    if submitted:

        amount_deviation = amount - user_avg_amount

        # Feature order must match the training notebook.
        input_data = pd.DataFrame(
            [[
                amount,
                hour,
                amount_deviation,
                transactions_last_hour
            ]],
            columns=[
                "amount",
                "hour",
                "amount_deviation",
                "transactions_last_hour"
            ]
        )

        try:
            prediction = model.predict(input_data)[0]
            anomaly_score = float(model.decision_function(input_data)[0])

            st.divider()
            st.subheader("Analysis Result")

            r1, r2 = st.columns(2)

            with r1:
                if prediction == -1:
                    st.error("⚠️ ANOMALY FLAGGED")
                    st.write(
                        "The model considers this transaction unusual "
                        "relative to the patterns it learned."
                    )
                else:
                    st.success("✅ NO ANOMALY FLAGGED")
                    st.write(
                        "The model did not flag this transaction "
                        "as an anomaly."
                    )

            with r2:
                st.metric("Model Decision Score", f"{anomaly_score:.5f}")
                st.caption(
                    "Lower scores indicate more anomalous observations "
                    "for this Isolation Forest model."
                )

            st.warning(
                "This is an experimental model using synthetic training "
                "data. It is not connected to a bank or live UPI network. "
                "An alert is not proof of fraud, and a normal result "
                "does not guarantee safety."
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")

# ---------------- PAGE: TRANSACTIONS ----------------

elif page == "Transactions":

    st.markdown(
        '<div class="main-title">🔎 Transaction Explorer</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="subtitle">'
        'Search, filter and export transaction records'
        '</div>',
        unsafe_allow_html=True
    )

    filtered = df.copy()

    col1, col2, col3 = st.columns(3)

    with col1:
        if "location" in filtered.columns:
            locations = sorted(
                filtered["location"].dropna().astype(str).unique()
            )
            selected_locations = st.multiselect(
                "Location",
                locations,
                default=locations
            )
            filtered = filtered[
                filtered["location"].astype(str).isin(selected_locations)
            ]

    with col2:
        if "merchant_category" in filtered.columns:
            categories = sorted(
                filtered["merchant_category"].dropna().astype(str).unique()
            )
            selected_categories = st.multiselect(
                "Merchant Category",
                categories,
                default=categories
            )
            filtered = filtered[
                filtered["merchant_category"].astype(str)
                .isin(selected_categories)
            ]

    with col3:
        status = st.selectbox(
            "Transaction Status",
            ["All", "Flagged anomalies", "Other transactions"]
        )

        if status == "Flagged anomalies":
            filtered = filtered[filtered["is_suspicious"]]
        elif status == "Other transactions":
            filtered = filtered[~filtered["is_suspicious"]]

    col4, col5 = st.columns(2)

    with col4:
        if "amount" in filtered.columns and not filtered.empty:
            min_amount = float(filtered["amount"].min())
            max_amount = float(filtered["amount"].max())

            amount_range = st.slider(
                "Amount Range (₹)",
                min_value=min_amount,
                max_value=max_amount if max_amount > min_amount
                else min_amount + 1.0,
                value=(min_amount, max_amount if max_amount > min_amount
                       else min_amount + 1.0)
            )

            filtered = filtered[
                filtered["amount"].between(
                    amount_range[0], amount_range[1]
                )
            ]

    with col5:
        transaction_search = st.text_input("Search Transaction ID")

        if transaction_search and "transaction_id" in filtered.columns:
            filtered = filtered[
                filtered["transaction_id"].astype(str)
                .str.contains(transaction_search, case=False, na=False)
            ]

    st.write(f"**Matching transactions:** {len(filtered):,}")

    st.dataframe(
        filtered,
        width="stretch",
        hide_index=True
    )

    csv_data = filtered.to_csv(index=False).encode("utf-8")

    st.download_button(
        "📥 Download Filtered CSV",
        data=csv_data,
        file_name="upi_filtered_transactions.csv",
        mime="text/csv",
        width="stretch"
    )


# ---------------- PAGE: ABOUT ----------------

elif page == "About Project":

    st.markdown(
        '<div class="main-title">ℹ️ About This Project</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    ### UPI Fraud Detection using Machine Learning

    **Objective:** Explore transaction patterns and flag potentially
    unusual transactions for further review.

    **Technology stack**
    - Python
    - Pandas
    - Scikit-learn
    - Isolation Forest
    - Streamlit
    - Plotly

    **Features**
    - Transaction analytics dashboard
    - Location and merchant analysis
    - Interactive transaction explorer
    - CSV report export
    - Live anomaly scoring using the saved model

    **Limitations**

    This project uses synthetic data. Its results do not establish
    real-world fraud detection performance. The live prediction page
    accepts manually entered values and does not connect to a payment
    provider or bank.
    """)

# ---------------- PAGE: MODEL PERFORMANCE ----------------

elif page == "Model Performance":

    st.markdown(
        '<div class="main-title">📊 Model Performance</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Evaluate Isolation Forest on a held-out test set'
        '</div>',
        unsafe_allow_html=True
    )

    if model is None:
        st.error("Trained model load nahi hua.")
        st.stop()

    required_features = [
        "amount",
        "hour",
        "amount_deviation",
        "transactions_last_hour"
    ]

    missing_features = [
        feature for feature in required_features
        if feature not in df.columns
    ]

    if missing_features:
        st.error(
            "Performance calculate karne ke liye CSV mein ye "
            f"features missing hain: {missing_features}"
        )
        st.info(
            "Pehle original Colab notebook se in features ko "
            "reconstruct karke CSV mein save karna hoga."
        )
        st.stop()

    if "is_fraud" not in df.columns:
        st.error(
            "CSV mein is_fraud label missing hai. "
            "Actual test evaluation ke liye labelled data zaroori hai."
        )
        st.stop()

    X = (
        df[required_features]
        .replace([float("inf"), float("-inf")], 0)
        .fillna(0)
    )

    y = df["is_fraud"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Model was originally trained on the training split.
    # Do not fit it on the test data.
    predictions = model.predict(X_test)

    # Isolation Forest: -1 means anomaly, +1 means normal.
    y_pred = (predictions == -1).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Normal", "Fraud"],
        output_dict=True,
        zero_division=0
    )

    precision = report["Fraud"]["precision"]
    recall = report["Fraud"]["recall"]
    f1 = report["Fraud"]["f1-score"]

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Accuracy", f"{accuracy:.1%}")

    with c2:
        st.metric("Fraud Precision", f"{precision:.1%}")

    with c3:
        st.metric("Fraud Recall", f"{recall:.1%}")

    with c4:
        st.metric("Fraud F1-score", f"{f1:.1%}")

    st.caption(
        f"Evaluation uses {len(X_test):,} held-out test transactions. "
        "Results are specific to this synthetic dataset."
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Confusion Matrix")

        cm = confusion_matrix(
            y_test,
            y_pred,
            labels=[0, 1]
        )

        fig, ax = plt.subplots()
        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Normal", "Fraud"]
        )
        display.plot(ax=ax, values_format="d", colorbar=False)
        ax.set_title("Actual vs Predicted")
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.subheader("Classification Report")

        report_df = pd.DataFrame(report).transpose()
        st.dataframe(
            report_df.round(3),
            width="stretch"
        )

    st.subheader("Interpretation")

    st.write(
        f"- **Precision:** {precision:.1%} of transactions flagged "
        "by the model were labelled fraud in the test data."
    )

    st.write(
        f"- **Recall:** The model detected {recall:.1%} of the "
        "labelled fraud transactions in the test data."
    )

    st.write(
        f"- **F1-score:** The combined precision-recall score is "
        f"{f1:.1%}."
    )

    st.warning(
        "This evaluation is valid only if the saved model was trained "
        "on the same training split and feature definitions. If the "
        "model was retrained on a different split, retrain and evaluate "
        "it consistently before reporting these metrics."
    )