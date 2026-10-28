# UPI Fraud Detection Using Machine Learning

## Project Overview
This project uses machine learning to identify suspicious UPI transactions through anomaly detection. It analyzes transaction amounts, transaction timing, and user transaction behavior.

## Technologies Used
- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Plotly
- Scikit-learn
- Isolation Forest

## Dataset
A synthetic dataset containing 10,000 transactions was generated for educational and experimental purposes.

## Project Workflow
1. Generate transaction data.
2. Perform exploratory data analysis (EDA).
3. Engineer transaction behavior features.
4. Apply the Isolation Forest algorithm.
5. Evaluate predictions using precision, recall, F1-score, and a confusion matrix.
6. Compare different contamination settings.
7. Visualize suspicious transactions through charts and a dashboard.

## Model Evaluation
The model was evaluated on a held-out test set of 2,000 transactions.

Results vary with the model configuration. The evaluation uses synthetic labels and does not represent verified performance on real banking data.

## Limitations
- The dataset is synthetic.
- The model detects anomalies, not confirmed fraud.
- Real-world deployment would require representative transaction data, stronger validation, and appropriate security and privacy controls.

## Future Improvements
- Evaluate on realistic, independently labeled data.
- Compare Isolation Forest with supervised classification algorithms.
- Improve feature engineering and threshold selection.
- Deploy a dashboard for transaction monitoring.

## Author
Abhishek
