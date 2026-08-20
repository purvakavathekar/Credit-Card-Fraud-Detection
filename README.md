# 💳 Credit Card Fraud Detection Pipeline & Web App

An end-to-end Machine Learning system designed to identify fraudulent transactions in highly imbalanced financial datasets using XGBoost, SMOTE resampling, and threshold tuning.

## 🚀 Key Features
- **Data Leakage Prevention:** `StandardScaler` fitted strictly on training data split prior to transformation.
- **Class Imbalance Handling:** SMOTE oversampling applied exclusively to training data.
- **Model Benchmark:** Evaluated Random Forest vs. XGBoost using ROC-AUC and Precision-Recall (PR-AUC) metrics.
- **Threshold Optimization:** Decision boundary tuned via Precision-Recall curve to maximize F1-score.
- **Live Web Interface:** Interactive Streamlit dashboard for real-time transaction inference.

## 🛠️ Project Architecture
```text
├── models/                     # Serialized model, scaler, & threshold
│   ├── xgb_fraud_model.pkl
│   ├── scaler.pkl
│   └── best_threshold.pkl
├── notebooks/                  # Analysis notebook
│   └── fraud_detection.ipynb
├── app.py                      # Streamlit application entry point
├── .gitignore                  # Git tracking rules
└── README.md                   # Documentation