import streamlit as st
import pandas as pd
import numpy as np

#--- Page Configuration ---
st.set_page_config(page_title="Customer churn predictor", layout= "centered")

st.title("Customer churn prediction")
st.write("Enter the customer's profile information below to predict the chances of them churning")
st.markdown("---")

#  --- Input section ---
# 1. Demographics
st.header("Customer Demographics")
col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", options=["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", options= ["Yes", "No"])
with col2:
    partner = st.selectbox("Has partner", ["Yes","No"])
    dependents = st.selectbox("Has dependents", ["Yes","No"])

# 2. Account and billing
st.header("Account and Billing Information")
col3, col4 = st.columns(2)
with col3:
    contract = st.selectbox("Contract Type",["Month-month"," One year", "Two years"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    paperless = st.selectbox("Paperless Billing",["Yes", "No"])
with col4:
    tenure = st.slider("Tenure (Months)", min_value= 0, max_value= 72, value = 6)
    monthly_charges = st.number_input("Monthly Charges ($)", min_value = 18.0, max_value = 120.0, value= 50.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=8700.0, value= 600.0)

# 3. Subscribed servicees
with st.expander("Subscrbed services"):
    col5, col6, col7 = st.columns(3)
    with col5:
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    with col6:
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    with col7:
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

st.markdown("---")