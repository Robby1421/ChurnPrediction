import pandas as pd
import numpy as np
import streamlit as st
import joblib
import requests
from io import BytesIO
from sklearn.preprocessing import StandardScaler
from streamlit_option_menu import option_menu

# Set up the page configuration
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

# Sidebar options
with st.sidebar:
    st.header("Settings")
    options = option_menu(
        "Churn Prediction App",
        ["Home", "Customer Churn Prediction"],
        default_index=0
    )

# Load the pre-trained model from GitHub
@st.cache_resource
def load_pkl_from_github(url):
    response = requests.get(url)
    if response.status_code == 200:
        return joblib.load(BytesIO(response.content))
    else:
        st.error(f"Error loading file from GitHub: {response.status_code}")
        st.stop()

# Replace with your actual GitHub raw URL
model_url = "https://github.com/Robby1421/ChurnPrediction/blob/main/random_forest_model.pkl"
model = load_pkl_from_github(model_url)  # Load model

# Load the dataset to fit the scaler
@st.cache_resource
def load_training_data():
    data_url = "https://github.com/Robby1421/ChurnPrediction/blob/main/customer_churn.csv"  # Replace with your CSV file URL
    df = pd.read_csv(data_url)
    
    # Drop unnecessary columns and handle missing values
    df = df.drop(columns=['customer_id', 'gender', 'country'], errors='ignore')  # Drop unnecessary categorical columns
    df = df.dropna(subset=['total_charges'])  # Handle missing data
    
    # Ensure numerical columns are present
    numerical_cols = ['age', 'tenure_months', 'monthly_charges', 'total_charges', 'number_of_logins', 'watch_hours']
    df[numerical_cols] = df[numerical_cols].apply(pd.to_numeric, errors='coerce')
    
    return df

# Fit the scaler on the numerical columns
@st.cache_resource
def fit_scaler(df):
    numerical_cols = ['age', 'tenure_months', 'monthly_charges', 'total_charges', 'number_of_logins', 'watch_hours']
    scaler = StandardScaler()
    scaler.fit(df[numerical_cols])
    return scaler

# Load training data and fit the scaler
training_data = load_training_data()
scaler = fit_scaler(training_data)

# Function to preprocess the input data
def preprocess_input(input_data):
    # Convert input data into a DataFrame
    data = pd.DataFrame([input_data])

    # List of numerical columns to scale
    numerical_cols = ['age', 'tenure_months', 'monthly_charges', 'total_charges', 'number_of_logins', 'watch_hours']

    # Scale numerical features using the fitted scaler
    data[numerical_cols] = scaler.transform(data[numerical_cols])

    return data

# Page: Home
if options == "Home":
    st.title("Welcome to Customer Churn Prediction!")
    st.write("""
    This app predicts whether a customer is likely to churn based on their demographic and service usage information. 
    You can upload customer data, train a model, and make predictions.
    """)

# Page: Customer Churn Prediction
elif options == "Customer Churn Prediction":
    st.title("Customer Churn Prediction")
    st.write("""
    Enter customer details to predict the likelihood of them churning. The model uses customer data such as age, tenure, 
    monthly charges, total charges, number of logins, and watch hours to make predictions.
    """)

    # Customer input fields
    age = st.number_input('Age', min_value=18, max_value=100, value=30)
    tenure_months = st.number_input('Tenure (months)', min_value=1, max_value=72, value=12)
    monthly_charges = st.number_input('Monthly Charges', min_value=20.0, max_value=200.0, value=50.0)
    total_charges = st.number_input('Total Charges', min_value=50.0, max_value=10000.0, value=500.0)
    number_of_logins = st.number_input('Number of Logins', min_value=0, max_value=1000, value=50)
    watch_hours = st.number_input('Watch Hours', min_value=0, max_value=100, value=10)

    # Create a dictionary with the input values
    input_data = {
        'age': age,
        'tenure_months': tenure_months,
        'monthly_charges': monthly_charges,
        'total_charges': total_charges,
        'number_of_logins': number_of_logins,
        'watch_hours': watch_hours,
    }

    # Preprocess the input data
    processed_input = preprocess_input(input_data)

    # Make predictions on the input data
    prediction = model.predict(processed_input)
    prediction_proba = model.predict_proba(processed_input)[:, 1]  # Get probability for class 1 (churn)

    if prediction == 1:
        st.write(f"The customer is predicted to churn with a probability of {prediction_proba[0]:.2f}.")
    else:
        st.write(f"The customer is predicted to stay with a probability of {1 - prediction_proba[0]:.2f}.")
