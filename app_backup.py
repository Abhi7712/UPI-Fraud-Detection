
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# -------------------------------
# PAGE CONFIGURATION
# -------------------------------
st.set_page_config(
    page_title="UPI Fraud Detection",
    page_icon="🛡️",
    layout="wide"
)

# -------------------------------
# CUSTOM CSS
# -------------------------------
st.markdown("""
<style>
    .main {
        background-color: #f5f7fb;
    }

    .block-container {
        padding-top: 1.5rem;
    }

    div[data-testid="stMetric"] {
        background-color: white;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
    }

    h1, h2, h3 {
        color: #172554;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    file_path = Path(__file__).parent / "upi_transactions.csv"

    data = pd.read_csv(file_path)
    data["transaction_time"] = pd.to_datetime(
        data["transaction_time"],
        errors="coerce"
    )

    data["is_suspicious"] = (
        data["is_suspicious"].astype(bool)
    )

    return data


try:
    df = load_data()
except FileNotFoundError:
    st.error(
        "upi_transactions.csv file app.py ke same folder mein rakho."
    )
    st.stop()

# -------------------------------
# SIDEBAR FILTERS
# -------------------------------
st.sidebar.title("🔎 Transaction Filters")

locations = sorted(df["location"].dropna().unique())
categories = sorted(df["merchant_category"].dropna().unique())
devices = sorted(df["device_type"].dropna().unique())

selected_locations = st.sidebar.multiselect(
    "Select Location",
    locations,
    default=locations
)

selected_categories = st.sidebar.multiselect(
    "Merchant Category",
    categories,
    default=categories
)

selected_devices = st.sidebar.multiselect(
    "Device Type",
    devices,
    default=devices
)

min_amount = int(df["amount"].min())
max_amount = int(df["amount"].max())

amount_range = st.sidebar.slider(
    "Transaction Amount (₹)",
    min_value=min_amount,
    max_value=max_amount,
    value=(min_amount, max_amount)
)

search_id = st.sidebar.text_input(
    "Search Transaction ID"
)

# -------------------------------
# APPLY FILTERS
# -------------------------------
filtered_df = df[
    df["location"].isin(selected_locations)
    & df["merchant_category"].isin(selected_categories)
    & df["device_type"].isin(selected_devices)
    & df["amount"].between(
        amount_range[0],
        amount_range[1]
    )
].copy()

if search_id.strip():
    filtered_df = filtered_df[
        filtered_df["transaction_id"].str.contains(
            search_id.strip(),
            case=False,
            na=False
        )
    ]

# -------------------------------
# DASHBOARD HEADER
# -------------------------------
st.title("🛡️ UPI Fraud Detection Dashboard")

st.markdown(
    "### Transaction Monitoring & Anomaly Analysis"
)

st.caption(
    "Educational project using synthetic transaction data. "
    "Suspicious transactions are model-detected anomalies, "
    "not confirmed fraud."
)

st.divider()

# -------------------------------
# KPI CARDS
# -------------------------------
total = len(filtered_df)

suspicious = int(
    filtered_df["is_suspicious"].sum()
)

normal = total - suspicious

suspicious_rate = (
    suspicious / total * 100
    if total > 0 else 0
)

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Transactions", f"{total:,}")
c2.metric("Suspicious Transactions", f"{suspicious:,}")
c3.metric("Normal Transactions", f"{normal:,}")
c4.metric("Suspicious Rate", f"{suspicious_rate:.2f}%")

st.divider()

if filtered_df.empty:
    st.warning("No transactions match the selected filters.")
    st.stop()

# -------------------------------
# CHARTS
# -------------------------------
left, right = st.columns(2)

with left:
    st.subheader("Transaction Amount Distribution")

    fig_amount = px.histogram(
        filtered_df,
        x="amount",
        color="is_suspicious",
        nbins=40,
        barmode="overlay",
        opacity=0.75,
        color_discrete_map={
            True: "#dc2626",
            False: "#2563eb"
        },
        labels={
            "amount": "Amount (₹)",
            "is_suspicious": "Suspicious"
        }
    )

    fig_amount.update_layout(
        template="plotly_white",
        legend_title_text="Transaction Type"
    )

    st.plotly_chart(
        fig_amount,
        width="stretch"
    )

with right:
    st.subheader("Normal vs Suspicious")

    type_counts = pd.DataFrame({
        "Transaction Type": ["Normal", "Suspicious"],
        "Count": [normal, suspicious]
    })

    fig_type = px.pie(
        type_counts,
        names="Transaction Type",
        values="Count",
        hole=0.55,
        color="Transaction Type",
        color_discrete_map={
            "Normal": "#2563eb",
            "Suspicious": "#dc2626"
        }
    )

    fig_type.update_layout(
        template="plotly_white"
    )

    st.plotly_chart(
        fig_type,
        width="stretch"
    )

# -------------------------------
# LOCATION ANALYSIS
# -------------------------------
st.subheader("Suspicious Transactions by Location")

location_data = (
    filtered_df.groupby("location", as_index=False)
    .agg(
        Total_Transactions=("transaction_id", "count"),
        Suspicious_Transactions=("is_suspicious", "sum")
    )
)

fig_location = px.bar(
    location_data,
    x="location",
    y="Suspicious_Transactions",
    hover_data=["Total_Transactions"],
    color="Suspicious_Transactions",
    color_continuous_scale="Reds",
    labels={
        "location": "Location",
        "Suspicious_Transactions": "Suspicious Transactions"
    }
)

fig_location.update_layout(
    template="plotly_white",
    coloraxis_showscale=False
)

st.plotly_chart(
    fig_location,
    width="stretch"
)

# -------------------------------
# TRANSACTION EXPLORER
# -------------------------------
st.subheader("🔍 Transaction Explorer")

status_filter = st.radio(
    "Choose transaction type",
    ["All", "Suspicious Only", "Normal Only"],
    horizontal=True
)

if status_filter == "Suspicious Only":
    table_df = filtered_df[
        filtered_df["is_suspicious"]
    ].copy()

elif status_filter == "Normal Only":
    table_df = filtered_df[
        ~filtered_df["is_suspicious"]
    ].copy()

else:
    table_df = filtered_df.copy()

table_df = table_df.sort_values(
    "amount",
    ascending=False
)

display_columns = [
    "transaction_id",
    "user_id",
    "amount",
    "merchant_category",
    "device_type",
    "location",
    "transaction_time",
    "is_suspicious"
]

st.dataframe(
    table_df[display_columns],
    width="stretch",
    hide_index=True
)

# -------------------------------
# CSV EXPORT
# -------------------------------
csv_data = table_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📥 Download Filtered Transactions CSV",
    data=csv_data,
    file_name="filtered_upi_transactions.csv",
    mime="text/csv"
)

# -------------------------------
# FOOTER
# -------------------------------
st.divider()

st.caption(
    "UPI Fraud Detection | Python • Pandas • "
    "Scikit-learn • Streamlit • Plotly"
)

st.caption(
    "Note: This dashboard uses synthetic data. "
    "Anomaly detection does not prove that a transaction is fraudulent."
)