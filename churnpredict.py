import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Streamlit UI components
st.title("Customer Churn Prediction")

st.header("Enter customer details to predict churn:")

# Create inputs for user details
age = st.number_input("Age", min_value=18, max_value=100, value=30)
tenure_months = st.number_input("Tenure (in months)", min_value=1, max_value=100, value=12)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=50.0)
total_charges = st.number_input("Total Charges", min_value=0.0, value=500.0)
number_of_logins = st.number_input("Number of Logins", min_value=0, value=10)
watch_hours = st.number_input("Watch Hours", min_value=0.0, value=5.0)

# Categorical inputs
gender = st.selectbox("Gender", options=["Male", "Female"])
country = st.selectbox("Country", options=["US", "Canada", "UK", "Other"])
contract_type = st.selectbox("Contract Type", options=["Month-to-Month", "One Year", "Two Year"])
payment_method = st.selectbox("Payment Method", options=["Credit Card", "PayPal", "Bank Transfer", "Check"])
has_internet_service = st.selectbox("Has Internet Service", options=["Yes", "No"])

# Prepare the input data for prediction
input_data = pd.DataFrame({
    'age': [age],
    'tenure_months': [tenure_months],
    'monthly_charges': [monthly_charges],
    'total_charges': [total_charges],
    'number_of_logins': [number_of_logins],
    'watch_hours': [watch_hours],
    'gender': [gender],
    'country': [country],
    'contract_type': [contract_type],
    'payment_method': [payment_method],
    'has_internet_service': [has_internet_service],
})

# Load and preprocess the dataset for training the model
@st.cache
def load_data():
    # Replace with your actual data loading
    # For demonstration, let's assume you have a similar dataset
    data = pd.read_csv("customer_churn.csv")  # Replace with your actual dataset path
    return data

data = load_data()

# Handle missing values and preprocess the data
data = data.dropna(subset=['total_charges'])
label_encoders = {}
categorical_cols = ['gender', 'country', 'contract_type', 'payment_method', 'has_internet_service']
numerical_cols = ['age', 'tenure_months', 'monthly_charges', 'total_charges', 'number_of_logins', 'watch_hours']

# Encode categorical variables
for col in categorical_cols + ['churn']:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Scale numerical features
scaler = StandardScaler()
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

# Feature-target split
X = data.drop(columns=['customer_id', 'churn'])
y = data['churn']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate the model
st.subheader("Model Performance")
y_pred = rf_model.predict(X_test)
st.write(classification_report(y_test, y_pred))

# Preprocess the input data
for col in categorical_cols:
    input_data[col] = label_encoders[col].transform(input_data[col])

input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])

# Make prediction
prediction = rf_model.predict(input_data)

# Display the prediction
if prediction[0] == 1:
    st.warning("The customer is likely to churn!")
else:
    st.success("The customer is likely to stay!")

# Optionally, show the feature importance of the Random Forest model
st.subheader("Feature Importance")
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Plot feature importance
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=feature_importances, x='Importance', y='Feature', ax=ax)
st.pyplot(fig)
