
# UPI Fraud Detection & Intelligence Dashboard

An end-to-end Machine Learning project for analyzing synthetic UPI transactions, identifying suspicious patterns, and estimating transaction fraud risk.

## Project Overview

This project demonstrates a machine learning workflow for UPI transaction analysis using feature engineering, supervised classification, model evaluation, and an interactive Streamlit dashboard.

The goal is to explore how transaction characteristics and historical user behavior can be used to identify potentially fraudulent activity.

**Important:** This project uses a synthetic dataset. Its labels and results do not represent verified real-world banking fraud.

## Features

- Transaction analytics dashboard
- Fraud label distribution analysis
- Historical transaction feature engineering
- Leakage-aware chronological train-test split
- Logistic Regression model
- Random Forest model
- Fraud risk score prediction
- Model comparison
- Confusion matrix
- Precision, Recall, F1 Score and PR-AUC
- Transaction filtering and CSV export
- Automated feature tests

## Project Structure

```text
UPI-Fraud-Dashboard/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   ├── raw/
│   │   └── upi_transactions.csv
│   └── processed/
│       └── test_data.csv
│
├── models/
│   ├── logistic_regression.joblib
│   └── random_forest.joblib
│
├── src/
│   ├── __init__.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── notebooks/
│   └── UPI_Fraud_Detection.ipynb
│
├── reports/
│   ├── model_comparison.csv
│   ├── confusion_matrices.csv
│   └── figures/
│
└── tests/
    ├── __init__.py
    └── test_features.py
```

## Dataset

- Total transactions: 10,000
- Fraud-labelled transactions: 420
- Legitimate transactions: 9,580
- Dataset type: Synthetic
- Target variable: `is_fraud`

## Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Plotly
- Matplotlib
- Joblib
- Pytest

## Machine Learning Models

### 1. Logistic Regression
A supervised classification baseline for transaction risk analysis.

### 2. Random Forest
An ensemble classification model used to capture nonlinear relationships between transaction features.

## Feature Engineering

The project creates historical and time-based features:

- Transaction amount
- Transaction hour
- Cyclical hour encoding
- Weekend indicator
- Previous average transaction amount
- Transaction amount deviation
- Previous transaction count
- Transactions during the previous hour
- Merchant category
- Device type
- Location

Historical features are calculated using transactions that occurred before the current transaction.

## Model Evaluation

Chronological split:

- Training records: 8,000
- Testing records: 2,000

Initial evaluation results:

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 89.35% | 97.35% |
| Precision | 21.40% | 100.00% |
| Recall | 51.11% | 41.11% |
| F1 Score | 30.16% | 58.27% |
| PR-AUC | 0.4921 | 0.4794 |

These are experimental results on the supplied synthetic dataset, not evidence of real-world fraud detection performance.

## Installation

Clone the repository:

```bash
git clone https://github.com/Abhi7712/UPI-Fraud-Detection.git
cd UPI-Fraud-Detection
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Project

Train the models:

```bash
python -m src.train
```

Evaluate the models:

```bash
python -m src.evaluate
```

Run automated tests:

```bash
python -m pytest tests/ -v
```

Launch the dashboard:

```bash
python -m streamlit run app.py
```

## Limitations

- The dataset is synthetic and does not contain independently verified banking fraud outcomes.
- Model probabilities are not calibrated fraud probabilities.
- Risk thresholds are experimental.
- Results may not generalize to actual UPI transactions.
- Further validation on representative, privacy-protected real-world data would be required before any operational use.

## Future Improvements

- Add Isolation Forest comparison.
- Improve recall through threshold tuning.
- Add explainable AI using SHAP.
- Add temporal validation and model monitoring.
- Improve fraud-risk visualization.

## Author

**Abhishek Gupta**

Machine Learning & Data Science Portfolio Project
