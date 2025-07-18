import joblib
import pandas as pd

# Load model and features
model = joblib.load("heart_model.pkl")
features = joblib.load("model_features.pkl")

# Example: Predict on new data
new_patient = pd.DataFrame([{
    'Age': 70,
    'Sex': 1,
    'ChestPainType_ASY': 1,
    'ChestPainType_ATA': 0,
    'ChestPainType_NAP': 0,
    'ChestPainType_TA': 1,
    'RestingBP': 140,
    'Cholesterol': 233,
    'FastingBS': 1,
    'RestingECG_Normal': 1,
    'RestingECG_LVH': 0,
    'RestingECG_ST': 0,
    'MaxHR': 150,
    'ExerciseAngina': 0,
    'Oldpeak': 1.0,
    'ST_Slope_Flat': 1,
    'ST_Slope_Up': 0,
    'ST_Slope_Down': 0
}])

# Ensure all features exist
for col in features:
    if col not in new_patient.columns:
        new_patient[col] = 0
new_patient = new_patient[features]  # same order

# Predict
prediction = model.predict(new_patient)
print("Heart Disease Risk (1 = Yes, 0 = No):", prediction[0])
