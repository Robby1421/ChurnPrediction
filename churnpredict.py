import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load pre-trained model and scaler
@st.cache_resource
def load_model():
    with open('random_forest_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    return model, scaler

# Function for predicting churn
def predict_churn(input_data, model, scaler):
    if hasattr(scaler, 'transform'):
        scaled_data = scaler.transform(input_data)
    else:
        st.warning("Scaler not recognized; using raw input data.")
        scaled_data = input_data.values
    prediction = model.predict(scaled_data)
    probability = model.predict_proba(scaled_data)[:, 1]
    return prediction, probability

# Load the model and scaler
model, scaler = load_model()

# App Title
st.title("Customer Churn Prediction App")
st.write("Upload customer data to predict churn or use the manual input below.")

# File Upload Section
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:")
    st.write(data.head())
    
    if st.button("Predict Churn for Uploaded Data"):
        predictions, probabilities = predict_churn(data, model, scaler)
        data['Churn Prediction'] = predictions
        data['Churn Probability'] = probabilities
        st.write("Predictions:")
        st.write(data)
        st.download_button(
            label="Download Predictions as CSV",
            data=data.to_csv(index=False),
            file_name="churn_predictions.csv",
            mime="text/csv",
        )

# Manual Input Section
st.header("Manual Input")
st.write("Fill in the fields to predict churn for a single customer.")

# Example input form with dataset features
st.write("Please enter the customer details below:")
manual_input = {
    'Gender': st.selectbox("Gender", ['Male', 'Female']),
    'SeniorCitizen': st.selectbox("Senior Citizen", [0, 1]),
    'Partner': st.selectbox("Partner", ['Yes', 'No']),
    'Dependents': st.selectbox("Dependents", ['Yes', 'No']),
    'tenure': st.number_input("Tenure (months)", min_value=0, value=12),
    'PhoneService': st.selectbox("Phone Service", ['Yes', 'No']),
    'MultipleLines': st.selectbox("Multiple Lines", ['Yes', 'No', 'No phone service']),
    'InternetService': st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No']),
    'OnlineSecurity': st.selectbox("Online Security", ['Yes', 'No', 'No internet service']),
    'OnlineBackup': st.selectbox("Online Backup", ['Yes', 'No', 'No internet service']),
    'DeviceProtection': st.selectbox("Device Protection", ['Yes', 'No', 'No internet service']),
    'TechSupport': st.selectbox("Tech Support", ['Yes', 'No', 'No internet service']),
    'StreamingTV': st.selectbox("Streaming TV", ['Yes', 'No', 'No internet service']),
    'StreamingMovies': st.selectbox("Streaming Movies", ['Yes', 'No', 'No internet service']),
    'Contract': st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year']),
    'PaperlessBilling': st.selectbox("Paperless Billing", ['Yes', 'No']),
    'PaymentMethod': st.selectbox("Payment Method", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']),
    'MonthlyCharges': st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0),
    'TotalCharges': st.number_input("Total Charges ($)", min_value=0.0, value=500.0),
}

# Convert manual input to dataframe
if st.button("Predict Churn for Manual Input"):
    input_data = pd.DataFrame([manual_input])
    # Map categorical variables to numeric if required
    prediction, probability = predict_churn(input_data, model, scaler)
    st.write("Churn Prediction:", "Yes" if prediction[0] == 1 else "No")
    st.write("Churn Probability:", f"{probability[0]:.2f}")
