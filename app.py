import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Disease Predictor", layout="centered")
st.title("🩺 Disease Prediction System")
st.write("Predict **Heart Disease** based on patient information")

# Load models
model = joblib.load('model.pkl')
label_encoder = joblib.load('label_encoder.pkl')
scaler = joblib.load('scaler.pkl')
feature_encoders = joblib.load('feature_encoders.pkl')

st.header("Patient Information")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", 18, 100, 45)
    gender = st.selectbox("Gender", ["Male", "Female"])
    bp = st.selectbox("Blood Pressure", ["Low", "Normal", "High"])
    cholesterol = st.selectbox("Cholesterol", ["Normal", "High"])
    glucose = st.selectbox("Glucose Level", ["Normal", "High"])

with col2:
    smoking = st.selectbox("Smoking", ["No", "Yes"])
    alcohol = st.selectbox("Alcohol Consumption", ["No", "Yes"])
    exercise = st.selectbox("Physical Exercise", ["No", "Yes"])
    bmi = st.number_input("BMI", 15.0, 45.0, 25.0)
    family_history = st.selectbox("Family History of Disease", ["No", "Yes"])

if st.button("🔍 Predict Heart Disease", type="primary"):
    data = {
        'Age': [age],
        'Gender': [gender],
        'Blood Pressure': [bp],
        'Cholesterol': [cholesterol],
        'Glucose': [glucose],
        'Smoking': [smoking],
        'Alcohol Consumption': [alcohol],
        'Exercise': [exercise],
        'BMI': [bmi],
        'Family History': [family_history]
    }

    input_df = pd.DataFrame(data)

    # Apply the same encoding used in training
    for col, encoder in feature_encoders.items():
        if col in input_df.columns:
            input_df[col] = encoder.transform(input_df[col])

    # Scale
    input_scaled = scaler.transform(input_df)

    # Predict
    pred = model.predict(input_scaled)[0]
    disease_name = label_encoder.inverse_transform([pred])[0]

    if str(disease_name).lower() in ['yes', '1', 'positive']:
        st.error(f"⚠️ **Prediction: High Risk of Heart Disease**")
    else:
        st.success(f"✅ **Prediction: Low Risk / No Heart Disease**")

    st.info("This is a machine learning prediction. Always consult a doctor.")