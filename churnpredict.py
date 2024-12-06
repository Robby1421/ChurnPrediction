import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report

# Set up the app title
st.title("Customer Churn Prediction")

# Section 1: Upload Data
st.header("Upload Customer Data")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.dataframe(data.head())
else:
    st.warning("Please upload a CSV file to proceed.")
    st.stop()

# Section 2: Data Cleaning and Preprocessing
st.header("Data Preprocessing")
if "total_charges" in data.columns:
    # Handle missing values
    missing_before = data["total_charges"].isna().sum()
    data = data.dropna(subset=["total_charges"])
    missing_after = data["total_charges"].isna().sum()
    st.write(f"Missing values in `total_charges`: {missing_before} before, {missing_after} after.")
else:
    st.warning("`total_charges` column is missing. Ensure the dataset matches the use case.")
    st.stop()

# Encoding categorical variables
categorical_cols = data.select_dtypes(include=["object"]).columns
for col in categorical_cols:
    data[col] = LabelEncoder().fit_transform(data[col])

# Scale numerical features
scaler = StandardScaler()
numerical_cols = data.select_dtypes(include=["int64", "float64"]).columns
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
st.write("Data after encoding and scaling:")
st.dataframe(data.head())

# Section 3: Train-Test Split
st.header("Train-Test Split")
target = st.selectbox("Select Target Column", data.columns)
features = data.drop(columns=[target])
X_train, X_test, y_train, y_test = train_test_split(features, data[target], test_size=0.3, random_state=42)
st.write(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

# Section 4: Model Training
st.header("Model Training")
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Section 5: Evaluation
st.header("Model Evaluation")
y_pred = model.predict(X_test)
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# Section 6: Make Predictions
st.header("Make Predictions")
input_data = {}
for col in features.columns:
    input_data[col] = st.number_input(f"Enter {col}", value=0.0)
input_df = pd.DataFrame([input_data])
st.write("Input Preview:")
st.dataframe(input_df)

# Predict churn
prediction = model.predict(input_df)
st.write(f"Prediction: {'Churn' if prediction[0] else 'No Churn'}")
