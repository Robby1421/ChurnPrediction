import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# Load the data from GitHub
url = 'https://raw.githubusercontent.com/Robby1421/ChurnPrediction/main/customer_churn.csv'
data = pd.read_csv(url)

# Step 1: Data Cleaning and Exploration
st.title("Customer Churn Prediction App")
st.write("Data from GitHub: customer_churn.csv")

# Show dataset preview
st.write("Dataset Preview:")
st.write(data.head())

# Check for missing values
st.write("Missing Values Check:")
st.write(data.isnull().sum())

# Handle missing values in `total_charges` by dropping rows with missing values
data = data.dropna(subset=['total_charges'])

# Step 2: Feature Engineering and Preprocessing
# Label encode categorical features
categorical_cols = ['gender', 'country', 'contract_type', 'payment_method', 'has_internet_service', 'churn']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Standardize numerical features
numerical_cols = ['age', 'tenure_months', 'monthly_charges', 'total_charges', 'number_of_logins', 'watch_hours']
scaler = StandardScaler()
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

# Split data into features (X) and target (y)
X = data.drop(columns=['customer_id', 'churn'])
y = data['churn']

# Step 3: Model Training
# Train a Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X, y)

# Step 4: Predictions and Evaluation
# Predicting churn for all data
y_pred = rf_model.predict(X)
y_prob = rf_model.predict_proba(X)[:, 1]

# Evaluate the model
st.write("Model Evaluation:")
roc_auc = roc_auc_score(y, y_prob)
st.write(f"ROC-AUC Score: {roc_auc:.2f}")

# Display classification report and confusion matrix
st.write("Classification Report:")
st.text(classification_report(y, y_pred))

st.write("Confusion Matrix:")
cm = confusion_matrix(y, y_pred)
st.write(cm)

# Page 2: Prediction Input (Single User Input)
st.title("Manual Churn Prediction")

# User input for prediction (only use relevant features)
age = st.number_input("Age", min_value=18, max_value=100, value=30)
tenure_months = st.number_input("Tenure in months", min_value=1, max_value=72, value=12)
monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0)
total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=500.0)
number_of_logins = st.number_input("Number of Logins", min_value=0, value=10)
watch_hours = st.number_input("Watch Hours", min_value=0, value=20)

gender = st.selectbox("Gender", ['Male', 'Female'])
country = st.selectbox("Country", ['USA', 'Canada', 'UK'])
contract_type = st.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year'])
payment_method = st.selectbox("Payment Method", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
has_internet_service = st.selectbox("Has Internet Service", ['Yes', 'No'])

# Encode user input and scale
user_input = {
    'age': age,
    'tenure_months': tenure_months,
    'monthly_charges': monthly_charges,
    'total_charges': total_charges,
    'number_of_logins': number_of_logins,
    'watch_hours': watch_hours,
    'gender': gender,
    'country': country,
    'contract_type': contract_type,
    'payment_method': payment_method,
    'has_internet_service': has_internet_service
}

# Convert to DataFrame for prediction
input_data = pd.DataFrame([user_input])

# Label encoding and scaling for the input data
for col in ['gender', 'country', 'contract_type', 'payment_method', 'has_internet_service']:
    input_data[col] = label_encoders[col].transform(input_data[col])

input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])

# Predict using the trained model
if st.button("Predict Churn"):
    prediction = rf_model.predict(input_data)
    prediction_prob = rf_model.predict_proba(input_data)[:, 1]
    
    churn = "Yes" if prediction[0] == 1 else "No"
    st.write(f"Churn Prediction: {churn}")
    st.write(f"Churn Probability: {prediction_prob[0]:.2f}")
