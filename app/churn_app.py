import streamlit as st
import pandas as pd
import joblib
import os

# --- Load model and feature columns ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/rf_churn_model.joblib")
FEATURE_PATH = os.path.join(os.path.dirname(__file__), "../models/feature_columns.joblib")

model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURE_PATH)  # This contains all columns used in training

# --- Streamlit UI ---
st.title("📊 Customer Churn Prediction App")

# Input fields
tenure = st.number_input("Tenure (months)", 0, 100, 12)
monthly_charges = st.number_input("Monthly Charges", 0.0, 500.0, 70.0)
total_charges = st.number_input("Total Charges", 0.0, 20000.0, 1500.0)
num_services = st.number_input("Number of active services", 0, 10, 3)
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

# Example: if you want to add more categorical inputs, handle one-hot same way

# --- Create input dataframe ---
input_dict = {
    'tenure': [tenure],
    'MonthlyCharges': [monthly_charges],
    'TotalCharges': [total_charges],
    'NumServices': [num_services],
    'Contract_Month-to-month': [1 if contract=='Month-to-month' else 0],
    'Contract_One year': [1 if contract=='One year' else 0],
    'Contract_Two year': [1 if contract=='Two year' else 0],
    # Add other columns as 0
}

input_df = pd.DataFrame(input_dict)

# Add missing columns as zeros
for col in feature_columns:
    if col not in input_df.columns:
        input_df[col] = 0

# Ensure correct order
input_df = input_df[feature_columns]

# --- Prediction ---
if st.button("Predict"):
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]
    st.success(f"Predicted Churn: {'Yes' if pred==1 else 'No'}")
    st.info(f"Probability: {proba*100:.2f}%")