import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load trained model and scaler
with open("heart_disease_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Define expected feature columns (Must match training)
expected_features = [
    'age', 'trestbps', 'chol', 'fbs', 'thalach', 'exang', 'oldpeak', 'ca',
    'cp_0', 'cp_1', 'cp_2', 'restecg_0', 'restecg_1', 'restecg_2',
    'slope_0', 'slope_1', 'slope_2', 'thal_0', 'thal_1', 'thal_2', 'thal_3',
    'sex_0', 'sex_1'
]

# Streamlit Web App
st.title("💖 Heart Disease Prediction App")

# Sidebar Inputs
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

# Create DataFrame to match training feature format
input_data = pd.DataFrame(columns=expected_features)
input_data.loc[0, 'age'] = age
input_data.loc[0, 'trestbps'] = trestbps
input_data.loc[0, 'chol'] = chol
input_data.loc[0, 'fbs'] = fbs
input_data.loc[0, 'thalach'] = thalach
input_data.loc[0, 'exang'] = exang
input_data.loc[0, 'oldpeak'] = oldpeak
input_data.loc[0, 'ca'] = ca
input_data.loc[0, f'cp_{cp}'] = 1
input_data.loc[0, f'restecg_{restecg}'] = 1
input_data.loc[0, f'slope_{slope}'] = 1
input_data.loc[0, f'thal_{thal}'] = 1
input_data.loc[0, f'sex_{0 if sex == "Female" else 1}'] = 1

# Fill missing columns with 0
input_data.fillna(0, inplace=True)

# Ensure input_data matches training features
input_data = input_data.reindex(columns=expected_features, fill_value=0)

# Convert to NumPy and apply scaling
input_data_scaled = scaler.transform(input_data.to_numpy(dtype=np.float32))

# Prediction
if st.sidebar.button("Predict"):
    prediction = model.predict(input_data_scaled)
    predicted_class = int(round(float(prediction[0])))

    # Display result
    st.write("### 🔍 Prediction Result:")
    if predicted_class == 1:
        st.error("⚠️ **High Risk of Heart Disease! Consult a Doctor.**")
    else:
        st.success("✅ **No Signs of Heart Disease Detected!**")
