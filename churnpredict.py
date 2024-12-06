import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu

# Set up the page configuration
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

# Sidebar options
with st.sidebar:
    st.header("Settings")
    options = option_menu(
        "Churn Prediction App",
        ["Home", "Customer Churn Prediction", "Model Training", "Model Evaluation"],
        default_index=0
    )

# Function to preprocess the input data
def preprocess_input(input_data, le, scaler):
    # Convert input data into a DataFrame
    data = pd.DataFrame([input_data])

    # List of categorical columns (excludes 'gender')
    categorical_cols = ['country', 'contract_type', 'payment_method', 'has_internet_service']

    # Encode categorical variables using LabelEncoder
    for col in categorical_cols:
        # Handle unseen labels by using `le.transform` and `fit_transform` on fit during training
        data[col] = le.transform(data[col])

    # Scale numerical features using StandardScaler
    numerical_cols = ['age', 'tenure_months', 'monthly_charges', 'total_charges', 'number_of_logins', 'watch_hours']
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
    contract type, payment method, and usage to make predictions.
    """)

    # Customer input fields (removed 'gender')
    age = st.number_input('Age', min_value=18, max_value=100, value=30)
    tenure_months = st.number_input('Tenure (months)', min_value=1, max_value=72, value=12)
    monthly_charges = st.number_input('Monthly Charges', min_value=20.0, max_value=200.0, value=50.0)
    total_charges = st.number_input('Total Charges', min_value=50.0, max_value=10000.0, value=500.0)
    number_of_logins = st.number_input('Number of Logins', min_value=0, max_value=1000, value=50)
    watch_hours = st.number_input('Watch Hours', min_value=0, max_value=100, value=10)

    country = st.selectbox('Country', ['USA', 'Canada', 'Germany', 'UK'])
    contract_type = st.selectbox('Contract Type', ['Month-to-Month', 'One-Year', 'Two-Year'])
    payment_method = st.selectbox('Payment Method', ['Electronic Check', 'Mailed Check', 'Bank Transfer', 'Credit Card'])
    has_internet_service = st.selectbox('Has Internet Service', ['Yes', 'No'])

    # Create a dictionary with the input values (gender removed)
    input_data = {
        'age': age,
        'tenure_months': tenure_months,
        'monthly_charges': monthly_charges,
        'total_charges': total_charges,
        'number_of_logins': number_of_logins,
        'watch_hours': watch_hours,
        'country': country,
        'contract_type': contract_type,
        'payment_method': payment_method,
        'has_internet_service': has_internet_service
    }

    # URL to the raw CSV file on GitHub (replace with your file's raw URL)
    github_url = "https://raw.githubusercontent.com/yourusername/yourrepo/main/yourfile.csv"

    # Read the CSV file from GitHub
    df = pd.read_csv(github_url)

    # Drop 'customer_id' column if it exists
    if 'customer_id' in df.columns:
        df = df.drop('customer_id', axis=1)

    # Drop 'gender' column
    if 'gender' in df.columns:
        df = df.drop('gender', axis=1)

    st.write("Data Preview:", df.head())

    # Preprocess the data
    categorical_cols = ['country', 'contract_type', 'payment_method', 'has_internet_service']
    le = LabelEncoder()

    # Fit the LabelEncoder on categorical columns
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    # Scale numerical features
    numerical_cols = ['age', 'tenure_months', 'monthly_charges', 'total_charges', 'number_of_logins', 'watch_hours']
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # Train the model
    X = df.drop('churn', axis=1)
    y = df['churn']

    # Split the data for training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Make predictions on the input data
    processed_input = preprocess_input(input_data, le, scaler)
    prediction = model.predict(processed_input)
    prediction_proba = model.predict_proba(processed_input)[:, 1]  # Get probability for class 1 (churn)

    if prediction == 1:
        st.write(f"The customer is predicted to churn with a probability of {prediction_proba[0]:.2f}.")
    else:
        st.write(f"The customer is predicted to stay with a probability of {1 - prediction_proba[0]:.2f}.")

# Page: Model Training
elif options == "Model Training":
    st.title("Model Training")
    st.write("""
    Here you can upload your customer data and train the churn prediction model. This page allows you to train the model
    dynamically using a dataset you upload.
    """)

    # URL to the raw CSV file on GitHub (replace with your file's raw URL)
    github_url = "https://raw.githubusercontent.com/yourusername/yourrepo/main/yourfile.csv"

    # Read the CSV file from GitHub
    df = pd.read_csv(github_url)

    # Drop 'customer_id' column if it exists
    if 'customer_id' in df.columns:
        df = df.drop('customer_id', axis=1)

    # Drop 'gender' column
    if 'gender' in df.columns:
        df = df.drop('gender', axis=1)

    st.write("Data Preview:", df.head())

    # Preprocess the data and train the model
    categorical_cols = ['country', 'contract_type', 'payment_method', 'has_internet_service']
    le = LabelEncoder()

    # Fit the LabelEncoder on categorical columns
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    # Scale numerical features
    numerical_cols = ['age', 'tenure_months', 'monthly_charges', 'total_charges', 'number_of_logins', 'watch_hours']
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # Train the Random Forest model
    X = df.drop('churn', axis=1)  # Assuming 'churn' is the target column
    y = df['churn']

    # Split the data for training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    st.write("Model trained successfully!")

    # Evaluate the model on the test set
    y_pred = model.predict(X_test)

    # Display classification report
    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # Display confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    st.pyplot()
