# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

# Load the Cleveland Heart Disease Dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
columns = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang',
    'oldpeak', 'slope', 'ca', 'thal', 'target'
]
heart_df = pd.read_csv(url, names=columns)

# Check for missing values
print(heart_df.isnull().sum())

print("------------------------------------------------------------")
# Check dataset information
print(heart_df.info())

heart_df.head(10)

# Replace '?' with NaN and convert to numeric
heart_df.replace('?', np.nan, inplace=True)

# Convert necessary columns to numeric
for col in ['ca', 'thal']:
    heart_df[col] = pd.to_numeric(heart_df[col], errors='coerce')

# Fill missing values with the median
heart_df.fillna(heart_df.median(), inplace=True)

# Verify that no missing values remain
print(heart_df.isnull().sum())

# Convert categorical features to numerical (One-Hot Encoding)
heart_df = pd.get_dummies(heart_df, columns=['cp', 'restecg', 'slope', 'thal'])

# Convert 'sex' and 'fbs' to numerical
heart_df['sex'] = heart_df['sex'].map({0: 'female', 1: 'male'})
heart_df = pd.get_dummies(heart_df, columns=['sex'])

# Check transformed dataset
heart_df.head(10)

# Define Features (X) and Target (y)
X = heart_df.drop(columns=['target'])  # Features
y = heart_df['target'].apply(lambda x: 1 if x > 0 else 0)  # Convert to Binary Classification

# Split Data into Training & Test Sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training Data: {X_train.shape}")
print(f"Test Data: {X_test.shape}")

# Initialize Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train Model
rf_model.fit(X_train, y_train)

# Predict on Test Data
y_pred_rf = rf_model.predict(X_test)

# Evaluate Model
print("Random Forest Model Performance:")
print(classification_report(y_test, y_pred_rf))
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")

# Initialize XGBoost Model
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")

# Train Model
xgb_model.fit(X_train, y_train)

# Predict on Test Data
y_pred_xgb = xgb_model.predict(X_test)

# Evaluate Model
print("XGBoost Model Performance:")
print(classification_report(y_test, y_pred_xgb))
print(f"Accuracy: {accuracy_score(y_test, y_pred_xgb):.4f}")

# Feature Importance for Random Forest
importances = rf_model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df.sort_values(by='Importance', ascending=False, inplace=True)

# Plot Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title("Feature Importance in Heart Disease Prediction")
plt.show()

# Save the trained model
with open("heart_disease_model.pkl", "wb") as model_file:
    pickle.dump(xgb_model, model_file)

print("Model saved successfully as 'heart_disease_model.pkl'")
