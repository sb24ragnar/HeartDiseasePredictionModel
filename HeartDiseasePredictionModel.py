# Import required libraries
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

# Load the Cleveland Heart Disease Dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
columns = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang',
    'oldpeak', 'slope', 'ca', 'thal', 'target'
]

try:
    heart_df = pd.read_csv(url, names=columns)
    print("✅ Dataset loaded successfully!")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    exit()

# Replace '?' with NaN and convert necessary columns to numeric
heart_df.replace('?', np.nan, inplace=True)
for col in ['ca', 'thal']:
    heart_df[col] = pd.to_numeric(heart_df[col], errors='coerce')

# Fill missing values with the median
heart_df.fillna(heart_df.median(), inplace=True)

# One-Hot Encoding for categorical variables
heart_df = pd.get_dummies(heart_df, columns=['cp', 'restecg', 'slope', 'thal', 'sex'])

# Ensure column order is consistent across multiple runs
feature_columns = [col for col in heart_df.columns if col != 'target']
X = heart_df[feature_columns]  # Features
y = heart_df['target'].apply(lambda x: 1 if x > 0 else 0)  # Convert to binary classification

# Print class distribution before balancing
print("\nBefore SMOTE - Class Distribution:")
print(y.value_counts())

# Balance the dataset using SMOTE
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)

# Print class distribution after SMOTE
print("\nAfter SMOTE - Class Distribution:")
print(y_balanced.value_counts())

# Split Data into Training & Test Sets
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced)

# Scale Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler for Streamlit App
with open("scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

# Train XGBoost Model with Better Hyperparameters
xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="logloss"
)
xgb_model.fit(X_train_scaled, y_train)

# Predict and Evaluate Model
y_pred_xgb = xgb_model.predict(X_test_scaled)
print("\n🔍 XGBoost Model Performance:")
print(classification_report(y_test, y_pred_xgb))
print(f"📊 Accuracy: {accuracy_score(y_test, y_pred_xgb):.4f}")

# Save the trained model
with open("heart_disease_model.pkl", "wb") as model_file:
    pickle.dump(xgb_model, model_file)

# Feature Importance Analysis
feature_importances = pd.DataFrame({"Feature": X.columns, "Importance": xgb_model.feature_importances_})
feature_importances = feature_importances.sort_values(by="Importance", ascending=False)

print("\n🔹 Top 10 Important Features:")
print(feature_importances.head(10))

print("\n✅ Model and scaler saved successfully!")
