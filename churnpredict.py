# -*- coding: utf-8 -*-
"""
Customer Churn Prediction - Complete Workflow
This script includes data loading, cleaning, exploratory analysis, feature engineering, model building,
evaluation, and visualization for a customer churn prediction problem.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
# Step 1: Data Loading and Setup
# from google.colab import drive
# import pandas as pd
# import numpy as np

# # Mount Google Drive
# drive.mount('/content/drive')

# Load the data from the CSV file
file_path = 'https://github.com/Robby1421/ChurnPrediction/blob/main/customer_churn.csv'
data = pd.read_csv(file_path)

# Show the first few rows of the dataset
print(data.head())

# Step 2: Data Cleaning
# Check for missing values in the dataset
print("\nMissing values in each column:")
print(data.isnull().sum())

# Handle missing values by dropping rows with missing 'total_charges' values
data = data.dropna(subset=['total_charges'])

# Summary statistics and data types
print("\nSummary statistics:")
print(data.describe(include='all'))

# Check data info
print("\nData Info:")
print(data.info())

# Step 3: Visualizing Numerical Features
import matplotlib.pyplot as plt
import seaborn as sns

# Define numerical columns to explore
numerical_cols = ['age', 'tenure_months', 'monthly_charges', 'total_charges', 'number_of_logins', 'watch_hours']

# Create histograms for each numerical feature
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(2, 3, i)  # Create a 2x3 grid of subplots
    sns.histplot(data[col], kde=True, color="blue", bins=30)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Step 4: Visualizing Categorical Features
# Define categorical columns to explore
categorical_cols = ['gender', 'country', 'contract_type', 'payment_method', 'has_internet_service']

# Visualize categorical features with respect to churn
plt.figure(figsize=(15, 10))
for i, col in enumerate(categorical_cols, 1):
    plt.subplot(2, 3, i)  # Create a 2x3 grid of subplots
    sns.countplot(data=data, x=col, hue="churn", palette="viridis")
    plt.title(f"{col} vs Churn")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.legend(title="Churn", loc="upper right")
plt.tight_layout()
plt.show()

# Step 5: Correlation Heatmap
# Plot correlation heatmap for numerical features
sns.heatmap(data[numerical_cols].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Step 6: Feature Engineering
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Encode categorical variables
label_encoders = {}
for col in categorical_cols + ['churn']:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Scale numerical features
scaler = StandardScaler()
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

# Split features and target variable
X = data.drop(columns=['customer_id', 'churn'])
y = data['churn']

# Step 7: Train-Test Split
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Display the shape of the training and testing sets
print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# Step 8: Model Building
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Logistic Regression
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)

# Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Models are now trained
print("Models trained successfully.")

# Step 9: Model Evaluation
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

def evaluate_model(model, predictions, X_test, y_test):
    print(f"Model: {model.__class__.__name__}")

    # Classification Report
    print("Classification Report:")
    print(classification_report(y_test, predictions))

    # ROC-AUC Score
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print("ROC-AUC Score:", roc_auc)

    # Confusion Matrix
    cm = confusion_matrix(y_test, predictions)
    print("Confusion Matrix:")
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
    plt.title(f"{model.__class__.__name__} - Confusion Matrix")
    plt.show()

# Evaluate each model
evaluate_model(lr_model, lr_model.predict(X_test), X_test, y_test)
evaluate_model(dt_model, dt_model.predict(X_test), X_test, y_test)
evaluate_model(rf_model, rf_model.predict(X_test), X_test, y_test)

# Step 10: Model Comparison
import numpy as np

# Calculate accuracy and ROC-AUC for each model
models = ['Logistic Regression', 'Decision Tree', 'Random Forest']
accuracy = [
    lr_model.score(X_test, y_test),
    dt_model.score(X_test, y_test),
    rf_model.score(X_test, y_test)
]
roc_auc = [
    roc_auc_score(y_test, lr_model.predict_proba(X_test)[:, 1]),
    roc_auc_score(y_test, dt_model.predict_proba(X_test)[:, 1]),
    roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])
]

# Create a comparison bar chart
x = np.arange(len(models))  # positions of bars
fig, ax = plt.subplots(figsize=(10, 6))

# Plot accuracy and ROC-AUC
bar_width = 0.35
ax.bar(x - bar_width/2, accuracy, bar_width, label='Accuracy', color='skyblue')
ax.bar(x + bar_width/2, roc_auc, bar_width, label='ROC-AUC', color='orange')

# Adding labels and title
ax.set_xlabel('Models')
ax.set_ylabel('Scores')
ax.set_title('Model Comparison for Customer Churn Prediction')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

# Highlight the winning model (Random Forest)
winning_model_idx = np.argmax(roc_auc)
ax.bar(x[winning_model_idx] + bar_width/2, roc_auc[winning_model_idx], bar_width, color='green')  # Change color for the winner

plt.tight_layout()
plt.show()

# Step 11: Feature Importance (Random Forest)
# Extract feature importance for the Random Forest model
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Plot feature importance
sns.barplot(data=feature_importances, x='Importance', y='Feature')
plt.title("Feature Importance - Random Forest")
plt.show()
