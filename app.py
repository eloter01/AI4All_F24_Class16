import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model, scaler, and feature columns
xgb_model = joblib.load('xgboost_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_columns = joblib.load('feature_columns.pkl')

# Define numerical and binary features
numerical_cols = ['funding_total_usd', 'funding_rounds', 'days_to_first_funding',
                  'funding_duration', 'company_age_at_last_funding', 'founded_at_year',
                  'founded_at_month', 'first_funding_at_year', 'first_funding_at_month',
                  'last_funding_at_year', 'last_funding_at_month', 'founded_quarter',
                  'category_count', 'sector_count']

binary_cols = [col for col in feature_columns if col not in numerical_cols]

# User inputs
st.title('Company Success Predictor')
st.write('Predict whether a company will succeed or close based on company data.')

input_data = {}

# Numerical Inputs
for col in numerical_cols:
    input_data[col] = st.number_input(f'Enter {col}', value=0.0)

# Binary Inputs
for col in binary_cols:
    input_data[col] = st.selectbox(f'Does the company have {col}?', ['No', 'Yes'])
    input_data[col] = 1 if input_data[col] == 'Yes' else 0

# Create DataFrame
input_df = pd.DataFrame([input_data])

# Ensure all features are included
missing_cols = set(feature_columns) - set(input_df.columns)
for col in missing_cols:
    input_df[col] = 0

# Reorder columns
input_df = input_df[feature_columns]

# Scale numerical features
input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

# Prediction
if st.button('Predict'):
    prediction = xgb_model.predict(input_df)
    status_mapping = {0: 'closed', 1: 'operating', 2: 'acquired', 3: 'ipo'}  # Adjust as per your labels
    predicted_status = status_mapping.get(prediction[0], 'Unknown')

    st.subheader('Prediction Result:')
    st.write(f'The company is predicted to **{predicted_status.upper()}**.')
