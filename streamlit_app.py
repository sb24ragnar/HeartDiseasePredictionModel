import streamlit as st
import pickle
import numpy as np
import xgboost as xgb

# Load trained model
model = pickle.load(open("heart_disease_model.pkl", "rb"))

# Streamlit Web App
st.title("💖 Heart Disease Prediction App")

# Sidebar for input fields
st.sidebar.header("Patient Information")
age = st.sidebar.number_input("Age", 20, 100)
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
cp = st.sidebar.selectbox("Chest Pain Type", [0, 1, 2, 3])
trestbps = st.sidebar.number_input("Resting Blood Pressure (mm Hg)", 80, 200)
chol = st.sidebar.number_input("Cholesterol Level (mg/dL)", 100, 600)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
restecg = st.sidebar.selectbox("Resting ECG Results", [0, 1, 2])
thalach = st.sidebar.number_input("Max Heart Rate Achieved", 60, 220)
exang = st.sidebar.selectbox("Exercise-Induced Angina", [0, 1])
oldpeak = st.sidebar.number_input("ST Depression Induced by Exercise", 0.0, 5.0)
slope = st.sidebar.selectbox("Slope of the Peak Exercise ST Segment", [0, 1, 2])
ca = st.sidebar.number_input("Number of Major Vessels (0-4)", 0, 4)
thal = st.sidebar.selectbox("Thalassemia Type", [0, 1, 2, 3])

# Convert categorical variables
sex = 1 if sex == "Male" else 0

# Prediction
if st.sidebar.button("Predict"):
    # Convert input data to NumPy array (Ensure float type)

    input_data = np.array([[float(age), float(sex), float(cp), float(trestbps), float(chol),
                            float(fbs), float(restecg), float(thalach), float(exang),
                            float(oldpeak), float(slope), float(ca), float(thal),
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32)  # Placeholder values

    # Make sure shape matches model expectations
    print("Input Data Shape:", input_data.shape)  # Should be (1, 23)
    # Make prediction without using DMatrix (for XGBClassifier)
    prediction = model.predict(input_data)

    # Convert output to binary classification
    predicted_class = int(round(prediction))

    # Display result
    st.write(f"### 🔍 Prediction: {'⚠️ High Risk of Heart Disease' if predicted_class == 1 else '✅ Low Risk'}")
