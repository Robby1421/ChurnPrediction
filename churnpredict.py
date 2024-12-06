import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import io
from google.colab import files  # Only for Google Colab users

# Function to load data from an uploaded file
def load_data_from_uploaded_file():
    uploaded = files.upload()  # This will prompt you to upload a file
    file_name = next(iter(uploaded))  # Get the file name from the uploaded dictionary
    data = pd.read_csv(io.BytesIO(uploaded[file_name]))  # Read the CSV into a DataFrame
    print("Data loaded successfully!")
    return data

# Function for Data Exploration
def explore_data(data):
    print("Data Info:")
    print(data.info())

    # Check for missing values
    print("\nMissing values count:")
    print(data.isnull().sum())

    # Display summary statistics
    print("\nSummary Statistics:")
    print(data.describe(include='all'))

    # Visualize numerical columns
    numerical_cols = ['age', 'tenure_months', 'monthly_charges', 'total_charges', 'number_of_logins', 'watch_hours']
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(numerical_cols, 1):
        plt.subplot(2, 3, i)
        sns.histplot(data[col], kde=True, color="blue", bins=30)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    # Visualize categorical features
    categorical_cols = ['gender', 'country', 'contract_type', 'payment_method', 'has_internet_service']
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(categorical_cols, 1):
        plt.subplot(2, 3, i)
        sns.countplot(data=data, x=col, hue="churn", palette="viridis")
        plt.title(f"{col} vs Churn")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.legend(title="Churn", loc="upper right")
    plt.tight_layout()
    plt.show()

# Function for Feature Engineering
def feature_engineering(data):
    categorical_cols = ['gender', 'country', 'contract_type', 'payment_method', 'has_internet_service']

    # Encode categorical variables
    label_encoders = {}
    for col in categorical_cols + ['churn']:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    # Scale numerical features
    numerical_cols = ['age', 'tenure_months', 'monthly_charges', 'total_charges', 'number_of_logins', 'watch_hours']
    scaler = StandardScaler()
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

    # Split data into features and target
    X = data.drop(columns=['customer_id', 'churn'])
    y = data['churn']

    return X, y

# Function to train and evaluate models
def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42)
    }

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        # Classification report
        print(f"\n{model_name} - Classification Report:")
        print(classification_report(y_test, predictions))

        # ROC-AUC Score
        roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        print(f"{model_name} - ROC-AUC Score: {roc_auc}")

        # Confusion Matrix
        cm = confusion_matrix(y_test, predictions)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
        plt.title(f"{model_name} - Confusion Matrix")
        plt.show()

# Function to visualize feature importance for Random Forest
def plot_feature_importance(X, rf_model):
    feature_importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    sns.barplot(data=feature_importances, x='Importance', y='Feature')
    plt.title("Feature Importance - Random Forest")
    plt.show()

# Main function
def main():
    # Step 1: Load Data
    data = load_data_from_uploaded_file()

    # Step 2: Explore Data
    explore_data(data)

    # Step 3: Feature Engineering
    X, y = feature_engineering(data)

    # Step 4: Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Step 5: Train and Evaluate Models
    train_and_evaluate_models(X_train, X_test, y_train, y_test)

    # Step 6: Feature Importance (Random Forest)
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    plot_feature_importance(X, rf_model)

    # Step 7: Recommendations
    print("\nExample Recommendations:")
    print("1. Offer incentives to customers on month-to-month contracts with low tenure.")
    print("2. Improve content engagement for customers with low watch hours.")
    print("3. Provide better retention strategies for customers paying via PayPal.")

if __name__ == '__main__':
    main()
