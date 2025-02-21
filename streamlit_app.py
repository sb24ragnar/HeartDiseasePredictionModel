import streamlit as st
import pickle
import numpy as np
import pandas as pd

#trained model and scaler file loading
with open("heart_disease_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Defining expected feature columns (Must match training)
expected_features = [
    'age', 'trestbps', 'chol', 'fbs', 'thalach', 'exang', 'oldpeak', 'ca',
    'cp_0', 'cp_1', 'cp_2', 'restecg_0', 'restecg_1', 'restecg_2',
    'slope_0', 'slope_1', 'slope_2', 'thal_0', 'thal_1', 'thal_2', 'thal_3',
    'sex_0', 'sex_1'
]

# Streamlit Web App
st.title("💖 Heart Disease Prediction App")
st.markdown(
    """
    **Welcome to the Heart Disease Prediction App!**  
    Enter patient details in the sidebar, and click **Predict** to check heart disease risk.  
    **Disclaimer:** This model provides **AI-based predictions**, not a medical diagnosis.  
    """
)

# Sidebar Inputs
st.sidebar.header("🩺 Patient Information")
age = st.sidebar.number_input("Age", 20, 100, value=45)
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
cp = st.sidebar.selectbox("Chest Pain Type", [0, 1, 2, 3], index=0)
trestbps = st.sidebar.number_input("Resting Blood Pressure (mm Hg)", 80, 200, value=120)
chol = st.sidebar.number_input("Cholesterol Level (mg/dL)", 100, 600, value=200)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1], index=0)
restecg = st.sidebar.selectbox("Resting ECG Results", [0, 1, 2], index=1)
thalach = st.sidebar.number_input("Max Heart Rate Achieved", 60, 220, value=150)
exang = st.sidebar.selectbox("Exercise-Induced Angina", [0, 1], index=0)
oldpeak = st.sidebar.number_input("ST Depression Induced by Exercise", 0.0, 5.0, value=1.0)
slope = st.sidebar.selectbox("Slope of the Peak Exercise ST Segment", [0, 1, 2], index=1)
ca = st.sidebar.number_input("Number of Major Vessels (0-4)", 0, 4, value=0)
thal = st.sidebar.selectbox("Thalassemia Type", [0, 1, 2, 3], index=2)

#DataFrame to match traing feature format
input_data = pd.DataFrame(columns=expected_features)

#user inputs to DataFrame
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

# Filling missing columns with 0 (Ensures correct feature count that's why)
input_data.fillna(0, inplace=True)

# Ensures input_data matches training features
input_data = input_data.reindex(columns=expected_features, fill_value=0)

# Converting to NumPy and apply scaling
input_data_scaled = scaler.transform(input_data.to_numpy(dtype=np.float32))

# Prediction Button
if st.sidebar.button("🩺 Predict"):
    prediction_prob = model.predict_proba(input_data_scaled)[0][1]  # Get probability of disease risk (class 1)

    # Categorisation risk based on probability
    if prediction_prob < 0.01:
        risk_level = "🟢 **No Risk** (0%)"
        st.success("✅ **No signs of heart disease detected!**")
    elif prediction_prob < 0.34:
        risk_level = "🟡 **Low Risk** (1% - 33%)"
        st.warning("⚠️ **Mild risk detected. Regular health checkups recommended.**")
    elif prediction_prob < 0.67:
        risk_level = "🟠 **Moderate Risk** (34% - 66%)"
        st.warning("⚠️ **Moderate risk. Consider lifestyle changes and medical consultation.**")
    elif prediction_prob < 1.0:
        risk_level = "🔴 **High Risk** (67% - 99%)"
        st.error("⚠️ **High risk! Medical consultation is strongly advised.**")
    else:
        risk_level = "🚨 **Critical Risk! Immediate action needed!**"
        st.error("🚨 **Immediate medical attention required!**")

    # Prediction Results(O/P)
    st.subheader("🔍 Prediction Result:")
    st.write(f"### {risk_level}")
    st.write(f"📊 **Predicted Risk Probability:** {prediction_prob:.2%}")

# Footer
st.markdown("---")
st.markdown(
    """
    **📌 Note:** This model is built using **Machine Learning** and may not be 100% accurate.  
    For medical concerns, please consult a ***qualified healthcare professional***.  
    """
)
st.markdown(
    """
    <style>
        .footer {
            position: fixed;
            bottom: 10px;
            right: 15px;
            font-size: 14px;
            color: grey;
        }
    </style>
    <div class="footer">
        Developed by <b>@ragnarsk</b>
    </div>
    """,
    unsafe_allow_html=True
)
