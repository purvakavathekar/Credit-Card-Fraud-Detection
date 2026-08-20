import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set Page Config
st.set_page_config(
    page_title="Credit Card Fraud Detector",
    page_icon="💳",
    layout="wide"
)

# Title & Description
st.title("💳 Credit Card Fraud Detection System")
st.markdown("""
This app predicts the likelihood of a fraudulent transaction using a trained **XGBoost** model.
Adjust the transaction parameters below to test the detection pipeline.
""")

# Load Models & Artifacts
@st.cache_resource
def load_artifacts():
    # Adjust paths if app.py is inside notebooks folder or project root
    try:
        model = joblib.load("models/xgb_fraud_model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        threshold = joblib.load("models/best_threshold.pkl")
    except FileNotFoundError:
        model = joblib.load("notebooks/models/xgb_fraud_model.pkl")
        scaler = joblib.load("notebooks/models/scaler.pkl")
        threshold = joblib.load("notebooks/models/best_threshold.pkl")
    return model, scaler, threshold

try:
    model, scaler, threshold = load_artifacts()
    st.sidebar.success("Model artifacts loaded successfully!")
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# Sidebar Information
st.sidebar.header("Model Info")
st.sidebar.write(f"**Algorithm:** XGBoost Classifier")
st.sidebar.write(f"**Decision Threshold:** `{threshold:.4f}`")

# Form Inputs
st.header("Enter Transaction Details")

col1, col2 = st.columns(2)

with col1:
    amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=150.0, step=10.0)

with col2:
    st.write("PCA Features (V1 - V28):")
    st.caption("Default values set to zero (typical baseline for normal transactions).")

# Create input dictionary for features V1 to V28
input_data = {}
for i in range(1, 29):
    input_data[f'V{i}'] = 0.0

# Expandable section to tweak specific V1-V28 values if desired
with st.expander("Modify PCA Feature Values (Optional)"):
    exp_col1, exp_col2, exp_col3, exp_col4 = st.columns(4)
    cols = [exp_col1, exp_col2, exp_col3, exp_col4]
    for i in range(1, 29):
        col_idx = (i - 1) % 4
        input_data[f'V{i}'] = cols[col_idx].number_input(f"V{i}", value=0.0, step=0.1)

# Predict Button
if st.button("Analyze Transaction", type="primary"):
    # 1. Scale Amount
    scaled_amount = scaler.transform([[amount]])[0][0]
    
    # 2. Build Feature DataFrame matching exact training column order
    feature_dict = input_data.copy()
    feature_dict['scaled_amount'] = scaled_amount
    
    input_df = pd.DataFrame([feature_dict])
    
    # 3. Predict Probabilities
    fraud_probability = model.predict_proba(input_df)[0][1]
    is_fraud = fraud_probability >= threshold

    # 4. Display Results
    st.subheader("Analysis Results")
    
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        st.metric(
            label="Fraud Probability", 
            value=f"{fraud_probability * 100:.2f}%"
        )
    
    with res_col2:
        if is_fraud:
            st.error("🚨 ALERT: Fraudulent Transaction Detected!")
        else:
            st.success("✅ SAFE: Transaction appears normal.")

    # Progress bar indicator
    st.progress(float(fraud_probability))