import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# App Title
st.title("Customer Churn Prediction App")

# Sidebar: File Upload
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file for training", type=["csv"])

if uploaded_file is not None:
    # Load the uploaded data
    data = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.dataframe(data.head())

    # Drop irrelevant columns
    if 'customer_id' in data.columns:
        data = data.drop(['customer_id'], axis=1)
        st.write("Dropped 'customer_id' column from the dataset.")

    # Identify categorical columns and encode them
    categorical_columns = data.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        label_encoder = LabelEncoder()
        data[col] = label_encoder.fit_transform(data[col])

    # Split data into features and target
    X = data.drop(columns=['churn'], axis=1)  # Replace 'churn' with the actual target column name
    y = data['churn']

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest model
    st.write("### Training Random Forest Model...")
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Model Accuracy: **{accuracy:.2f}**")
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # Save the model for future use
    st.write("Model trained successfully. Ready for predictions.")

    # Prediction Section
    st.sidebar.header("Prediction Input")
    if st.sidebar.checkbox("Enable Manual Prediction"):
        # Manual Input
        st.sidebar.subheader("Enter Customer Features:")
        input_data = {}
        for col in X.columns:
            dtype = X[col].dtype
            if dtype == 'float64' or dtype == 'int64':
                input_data[col] = st.sidebar.number_input(f"Enter {col}", value=float(X[col].mean()))
            else:
                input_data[col] = st.sidebar.text_input(f"Enter {col}", "")

        # Predict
        if st.sidebar.button("Predict"):
            input_df = pd.DataFrame([input_data])
            prediction = model.predict(input_df)[0]
            prediction_text = "Churn" if prediction == 1 else "No Churn"
            st.write(f"Prediction for the customer: **{prediction_text}**")
else:
    st.write("Please upload a dataset to get started.")

st.write("---")
st.write("Developed with ❤️ using Streamlit.")
