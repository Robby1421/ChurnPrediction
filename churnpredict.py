import streamlit as st
import pandas as pd
import joblib
import requests
from io import BytesIO

# GitHub repo URL (replace with your actual GitHub repo URL)
GITHUB_RAW_URL = "https://github.com/Robby1421/ChurnPrediction/main/"

# Load scaler and model from GitHub
@st.cache_data
def load_scaler_and_model():
    scaler_url = GITHUB_RAW_URL + "random_forest_scaler.pkl"
    model_url = GITHUB_RAW_URL + "model.pkl"
    
    scaler = joblib.load(BytesIO(requests.get(scaler_url).content))
    model = joblib.load(BytesIO(requests.get(model_url).content))
    return scaler, model

# Load the scaler and model
scaler, model = load_scaler_and_model()

# App Title
st.title("Customer Churn Prediction App")

# Sidebar for input features
st.sidebar.header("Customer Input Features")

# Function to get user inputs
def get_user_input():
    age = st.sidebar.slider("Age", 18, 100, 35)
    tenure_months = st.sidebar.slider("Tenure (months)", 0, 100, 12)
    monthly_charges = st.sidebar.number_input("Monthly Charges", min_value=0.0, max_value=500.0, value=80.0)
    total_charges = st.sidebar.number_input("Total Charges", min_value=0.0, max_value=20000.0, value=960.0)
    number_of_logins = st.sidebar.slider("Number of Logins", 0, 100, 5)
    watch_hours = st.sidebar.slider("Watch Hours", 0, 500, 50)
    
    gender = st.sidebar.selectbox("Gender", options=["Male", "Female"])
    country = st.sidebar.selectbox("Country", options=["Country A", "Country B", "Country C"])
    contract_type = st.sidebar.selectbox("Contract Type", options=["Month-to-Month", "One Year", "Two Year"])
    payment_method = st.sidebar.selectbox("Payment Method", options=["Credit Card", "PayPal", "Bank Transfer", "Electronic Check"])
    has_internet_service = st.sidebar.radio("Has Internet Service?", options=["Yes", "No"])
    
    # Encoding the categorical inputs to match model requirements
    gender_encoded = 0 if gender == "Male" else 1
    country_encoded = {"Country A": 0, "Country B": 1, "Country C": 2}[country]
    contract_type_encoded = {"Month-to-Month": 0, "One Year": 1, "Two Year": 2}[contract_type]
    payment_method_encoded = {"Credit Card": 0, "PayPal": 1, "Bank Transfer": 2, "Electronic Check": 3}[payment_method]
    has_internet_service_encoded = 1 if has_internet_service == "Yes" else 0
    
    # Return as a dictionary
    data = {
        "age": age,
        "tenure_months": tenure_months,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "number_of_logins": number_of_logins,
        "watch_hours": watch_hours,
        "gender": gender_encoded,
        "country": country_encoded,
        "contract_type": contract_type_encoded,
        "payment_method": payment_method_encoded,
        "has_internet_service": has_internet_service_encoded
    }
    return pd.DataFrame(data, index=[0])

# Get user inputs
user_input = get_user_input()

# Display user inputs
st.subheader("Customer Input Features")
st.write(user_input)

# Scale numerical features
numerical_cols = ['age', 'tenure_months', 'monthly_charges', 'total_charges', 'number_of_logins', 'watch_hours']
user_input[numerical_cols] = scaler.transform(user_input[numerical_cols])

# Predict churn
prediction = model.predict(user_input)
prediction_proba = model.predict_proba(user_input)[:, 1]

# Display predictions
st.subheader("Prediction")
st.write("Churn" if prediction[0] == 1 else "No Churn")

# Display prediction probability
st.subheader("Churn Probability")
st.write(f"{prediction_proba[0]:.2%}")
