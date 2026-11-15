# UPI Fraud Detection & Intelligence Dashboard

An end-to-end machine-learning portfolio project that explores transaction analytics, historical feature engineering, supervised fraud classification, unsupervised anomaly detection, model evaluation, and an interactive Streamlit dashboard.

> **Research/portfolio use only.** The included data is synthetic. Its labels and model scores are not verified banking outcomes and must not be used to approve, block, or investigate real payments.

## What is included

- Streamlit dashboard: Overview, Check Transaction, Bulk CSV Detection, Prediction History, Model Performance, Transactions, Fraud Analysis, and About.
- Single-transaction FRAUD / NOT FRAUD classification with suspicious indicators.
- Bulk CSV scoring with downloadable prediction results.
- Local SQLite prediction history stored in `data/processed/prediction_history.db`.
- Leakage-aware historical features based on transactions occurring before the current transaction.
- Chronological 80/20 train-test split.
- Logistic Regression and Random Forest supervised classifiers.
- Isolation Forest anomaly-detection variants trained on legitimate training rows.
- Evaluation reports: Accuracy, Precision, Recall, F1, PR-AUC, and confusion-matrix counts.
- Input validation, heuristic suspicious indicators, and automated tests.

## Dataset

The supplied CSV contains 10,000 synthetic transactions and 420 records labelled as fraud (`is_fraud=1`). These are synthetic labels, not verified fraud cases. The dataset also contains precomputed flags such as `iqr_flag`, `if_flag`, and `if_flag_v2`; the dashboard displays anomaly flags separately from the `is_fraud` target.

## Project structure

```text
UPI-Fraud-Dashboard/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   ├── raw/upi_transactions.csv
│   └── processed/test_data.csv
├── models/
│   ├── logistic_regression.joblib
│   ├── random_forest.joblib
│   └── isolation_forest_*.joblib
├── notebooks/UPI_Fraud_Detection.ipynb
├── reports/
│   ├── model_comparison.csv
│   └── confusion_matrices.csv
├── src/
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── storage.py
└── tests/
    ├── test_features.py
    └── test_storage.py
```

## Features used by the models

**Numeric:** `amount`, `hour`, `hour_sin`, `hour_cos`, `is_weekend`, `user_avg_amount_prior`, `amount_deviation`, `user_transaction_count_prior`, `transactions_last_hour`.

**Categorical:** `merchant_category`, `device_type`, `location`.

Historical average, prior transaction count, and previous-hour velocity are computed using earlier transactions only. The model does not use the raw dataset's precomputed full-history average columns.

## Model evaluation

The saved evaluation report from the supplied project contains the following chronological holdout results:

| Model | Accuracy | Precision | Recall | F1 | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 89.35% | 21.40% | 51.11% | 30.16% | 0.4921 |
| Random Forest | 97.35% | 100.00% | 41.11% | 58.27% | 0.4794 |
| Isolation Forest (0.02) | 92.60% | 10.81% | 8.89% | 9.76% | 0.0884 |
| Isolation Forest (0.03) | 91.65% | 11.11% | 12.22% | 11.64% | 0.0884 |
| Isolation Forest (0.04) | 91.00% | 11.21% | 14.44% | 12.62% | 0.0884 |
| Isolation Forest (0.05) | 89.50% | 12.50% | 22.22% | 16.00% | 0.0884 |
| Isolation Forest (0.08) | 86.40% | 12.40% | 33.33% | 18.07% | 0.0884 |

Random Forest has the highest F1 among the two supervised models in this saved run and produced no false positives on this particular holdout, but it missed 53 of 90 fraud-labelled test records. That trade-off matters: high precision does not mean high fraud coverage.

Isolation Forest contamination changes the anomaly decision threshold. It changes the number of flagged records and the precision/recall trade-off; the same underlying anomaly ranking can retain the same PR-AUC across contamination variants.

These are results on one synthetic dataset and one chronological split. They do not establish real-world performance or generalization.

## Setup (Windows / PowerShell)

Open a terminal in the project folder:

```powershell
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run

Train/rebuild all models and the chronological test set:

```powershell
python -m src.train
```

Evaluate models and regenerate CSV reports:

```powershell
python -m src.evaluate
```

Run automated tests:

```powershell
python -m pytest tests/ -v
```

Launch the dashboard:

```powershell
python -m streamlit run app.py
```

Open the local URL printed by Streamlit (usually `http://localhost:8501`).

## Transaction checking

The **Check Transaction** page takes transaction amount, hour, merchant, device, location, weekend status, and user-history aggregates. It returns a clear **FRAUD** or **NOT FRAUD** model decision, an uncalibrated model score, an illustrative risk band, and rule-based indicators.

The **Bulk CSV Detection** page accepts a CSV with these exact columns:

```text
amount,hour,user_avg_amount_prior,user_transaction_count_prior,
transactions_last_hour,merchant_category,device_type,location,is_weekend
```

Download the CSV template from that page to get a ready-to-fill example. The **Prediction History** page shows single and bulk predictions saved in a local SQLite database. This history is local to the machine running the dashboard.

## Live prediction notes

The Live Prediction page accepts a transaction and historical user aggregates. For a meaningful simulation, those historical values must be computed from transactions strictly before the transaction being scored. The app shows heuristic indicators (e.g. unusually high amount vs. the entered historical average); these are simple rules, not SHAP explanations.

The percentage displayed is an **uncalibrated model score** from `predict_proba`, not a calibrated probability that fraud will occur. The High/Medium/Low bands are illustrative UI categories, not bank policy thresholds.

## Limitations and responsible use

- The dataset is synthetic and does not represent real payment-system traffic.
- Fraud labels are not independently verified.
- One chronological holdout is not sufficient for production validation.
- The class balance and generated patterns may differ from real fraud.
- The model score is not calibrated.
- No live UPI/bank API, account linkage, alert delivery, or case-management system is connected.
- A real deployment would require privacy/security review, representative labelled data, temporal backtesting, threshold governance, human review, monitoring, and regulatory/compliance assessment.

## Author

**Abhishek Gupta**  
Machine Learning / Data Science Portfolio Project


## Model artifact compatibility

The included `.joblib` models were serialized with **scikit-learn 1.6.1**. The requirements file pins that version because scikit-learn does not guarantee that serialized models can be loaded across arbitrary versions. If you intentionally change the scikit-learn version, retrain the models using `python -m src.train` and regenerate reports using `python -m src.evaluate`.

## Prediction history data

Prediction history is stored in `data/processed/prediction_history.db` (created automatically after the first prediction). It is local application data and is ignored by Git. Do not store real payment credentials, PINs, OTPs, or sensitive personal information in this demo.
