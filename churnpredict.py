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
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)
    probability = model.predict_proba(scaled_data)[:, 1]
    return prediction, probability

# Load the model and scaler
model, scaler = load_model()

# App Title
st.title("Churn Prediction App")
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

# Example input form (adjust according to dataset features)
features = {
    'Feature1': st.number_input("Feature1 Value", value=0.0),
    'Feature2': st.number_input("Feature2 Value", value=0.0),
    'Feature3': st.number_input("Feature3 Value", value=0.0),
    # Add more features as per your dataset
}

if st.button("Predict Churn for Manual Input"):
    input_data = pd.DataFrame([features.values()], columns=features.keys())
    prediction, probability = predict_churn(input_data, model, scaler)
    st.write("Churn Prediction:", "Yes" if prediction[0] == 1 else "No")
    st.write("Churn Probability:", f"{probability[0]:.2f}")
