import streamlit as st
import joblib
import requests
import pandas as pd
import numpy as np
from io import BytesIO
from sklearn.preprocessing import StandardScaler

# GitHub repo URL for raw files (change the URL to your actual GitHub repo URL)
GITHUB_RAW_URL = "https://raw.githubusercontent.com/Robby1421/ChurnPrediction/main/"

# Load scaler and model from GitHub
@st.cache_data
def load_scaler_and_model():
    # Raw URL for model and scaler files
    scaler_url = GITHUB_RAW_URL + "random_forest_scaler.pkl"
    model_url = GITHUB_RAW_URL + "model.pkl"
    
    # Fetch and load scaler and model using joblib
    scaler = joblib.load(BytesIO(requests.get(scaler_url).content))
    model = joblib.load(BytesIO(requests.get(model_url).content))
    return scaler, model

# Load model and scaler
scaler, model = load_scaler_and_model()

# Define the feature mappings for categorical variables
categorical_map = {
    "gender": {"Male": 0, "Female": 1},
    "country": {"Country A": 0, "Country B": 1, "Country C": 2},
    "contract_type": {"Month-to-Month": 0, "One Year": 1, "Two Year": 2},
    "payment_method": {"Credit Card": 0, "PayPal": 1, "Bank Transfer": 2, "Electronic Check": 3},
    "has_internet_service": {"Yes": 1, "No": 0},
}

# Function to preprocess and predict user input
def preprocess_and_predict(user_input):
    # Preprocess the categorical features using the mapping (handle unknown categories)
    user_input['gender'] = categorical_map["gender"].get(user_input['gender'], 0)  # Default to 'Male' (0)
    user_input['country'] = categorical_map["country"].get(user_input['country'], 0)  # Default to 'Country A' (0)
    user_input['contract_type'] = categorical_map["contract_type"].get(user_input['contract_type'], 0)  # Default to 'Month-to-Month' (0)
    user_input['payment_method'] = categorical_map["payment_method"].get(user_input['payment_method'], 0)  # Default to 'Credit Card' (0)
    user_input['has_internet_service'] = categorical_map["has_internet_service"].get(user_input['has_internet_service'], 0)  # Default to 'No' (0)

    # Convert input data into a DataFrame
    user_input_df = pd.DataFrame([user_input])

    # Scale the numerical features using the loaded scaler
    user_input_scaled = scaler.transform(user_input_df)

    # Predict with the trained model
    prediction = model.predict(user_input_scaled)
    return prediction

# Streamlit UI to input user data
st.title("Customer Churn Prediction")

# User input fields
age = st.number_input("Age", min_value=18, max_value=100, value=30)
tenure_months = st.number_input("Tenure (in months)", min_value=0, max_value=100, value=10)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=50.0)
total_charges = st.number_input("Total Charges", min_value=0.0, value=500.0)
number_of_logins = st.number_input("Number of Logins", min_value=0, max_value=100, value=5)
watch_hours = st.number_input("Watch Hours", min_value=0, max_value=100, value=20)

# Categorical input fields
gender = st.selectbox("Gender", options=["Male", "Female"], index=0)
country = st.selectbox("Country", options=["Country A", "Country B", "Country C"], index=0)
contract_type = st.selectbox("Contract Type", options=["Month-to-Month", "One Year", "Two Year"], index=0)
payment_method = st.selectbox("Payment Method", options=["Credit Card", "PayPal", "Bank Transfer", "Electronic Check"], index=0)
has_internet_service = st.selectbox("Has Internet Service", options=["Yes", "No"], index=0)

# Prepare the input data as a dictionary
user_input = {
    "age": age,
    "tenure_months": tenure_months,
    "monthly_charges": monthly_charges,
    "total_charges": total_charges,
    "number_of_logins": number_of_logins,
    "watch_hours": watch_hours,
    "gender": gender,
    "country": country,
    "contract_type": contract_type,
    "payment_method": payment_method,
    "has_internet_service": has_internet_service
}

# When user clicks on the predict button
if st.button("Predict Churn"):
    # Preprocess the input and make a prediction
    prediction = preprocess_and_predict(user_input)
    
    # Show the result
    if prediction[0] == 0:
        st.write("The customer is likely to **not churn**.")
    else:
        st.write("The customer is likely to **churn**.")

