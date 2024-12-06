import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the trained model
@st.cache(allow_output_mutation=True)
def load_model():
    return joblib.load('random_forest_model.pkl')

model = load_model()

# App Title
st.title("Customer Churn Prediction App")
st.write("""
This application predicts customer churn based on input features. 
Upload a dataset or input individual customer details for real-time predictions.
""")

# Sidebar: File Upload or Manual Input
st.sidebar.header("Input Options")
input_option = st.sidebar.selectbox("Choose Input Method:", ("Upload CSV File", "Manual Input"))

# Upload CSV Option
if input_option == "Upload CSV File":
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    
    if uploaded_file is not None:
        # Read uploaded file
        data = pd.read_csv(uploaded_file)

        # Check and drop customer_id column if present
        if 'customer_id' in data.columns:
            data = data.drop(['customer_id'], axis=1)
            st.write("Dropped 'customer_id' column from the input data.")
        
        # Ensure input matches expected model features
        expected_columns = ['age', 'tenure_months', 'monthly_charges', 'total_charges', 'number_of_logins', 'watch_hours']
        missing_columns = [col for col in expected_columns if col not in data.columns]
        
        if missing_columns:
            st.error(f"Missing required columns: {missing_columns}")
        else:
            # Make predictions
            predictions = model.predict(data)
            data['churn_prediction'] = predictions
            st.write("Predictions completed:")
            st.dataframe(data)
            
            # Option to download predictions
            csv_output = data.to_csv(index=False)
            st.download_button(
                label="Download Predictions as CSV",
                data=csv_output,
                file_name="churn_predictions.csv",
                mime="text/csv",
            )

# Manual Input Option
elif input_option == "Manual Input":
    st.sidebar.header("Manual Input Features")
    # Input fields for required features
    age = st.sidebar.number_input("Age", min_value=0, max_value=100, step=1)
    tenure_months = st.sidebar.number_input("Tenure (Months)", min_value=0, max_value=120, step=1)
    monthly_charges = st.sidebar.number_input("Monthly Charges", min_value=0.0, step=0.1)
    total_charges = st.sidebar.number_input("Total Charges", min_value=0.0, step=0.1)
    number_of_logins = st.sidebar.number_input("Number of Logins", min_value=0, step=1)
    watch_hours = st.sidebar.number_input("Watch Hours", min_value=0.0, step=0.1)
    
    # Predict button
    if st.sidebar.button("Predict"):
        # Create a DataFrame for the input
        input_data = pd.DataFrame({
            'age': [age],
            'tenure_months': [tenure_months],
            'monthly_charges': [monthly_charges],
            'total_charges': [total_charges],
            'number_of_logins': [number_of_logins],
            'watch_hours': [watch_hours],
        })
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        prediction_text = "Churn" if prediction == 1 else "No Churn"
        st.write(f"Prediction for the customer: **{prediction_text}**")

# Footer
st.write("---")
st.write("Developed with ❤️ using Streamlit.")
