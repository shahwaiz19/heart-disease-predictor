import streamlit as st
import pandas as pd
import joblib

# Load model and features
model = joblib.load("heart_model.pkl")
features = joblib.load("model_features.pkl")

st.title("❤️ Heart Disease Risk Predictor")
st.write("Enter patient details to estimate heart disease risk.")

# Input fields for patient data
user_input = {}

for col in features:
    if "Age" in col:
        user_input[col] = st.slider("Age", 20, 100, 50)
    elif "RestingBP" in col:
        user_input[col] = st.slider("Resting BP", 80, 200, 120)
    elif "Cholesterol" in col:
        user_input[col] = st.slider("Cholesterol", 100, 400, 200)
    elif "MaxHR" in col:
        user_input[col] = st.slider("Max Heart Rate", 60, 220, 150)
    elif "Oldpeak" in col:
        user_input[col] = st.slider("ST depression (oldpeak)", 0.0, 6.0, 1.0, step=0.1)
    else:
        user_input[col] = st.selectbox(col, [0, 1])

# Convert to DataFrame
input_df = pd.DataFrame([user_input])
input_df = input_df[features]  # Reorder columns

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.error("🚨 High Risk: Patient may have heart disease!")
    else:
        st.success("✅ Low Risk: Patient is unlikely to have heart disease.")
