import streamlit as st
import requests

# FastAPI backend URL
API_URL = "http://127.0.0.1:8000/predict"

# Streamlit app title
st.title("Heart Disease Prediction App")

# Input fields for user data
st.header("Enter Patient Data")
age = st.number_input("Age", min_value=20, max_value=80, step=1)
sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
chest_pain_type = st.selectbox("Chest Pain Type (0-3)", options=[0, 1, 2, 3])
resting_blood_pressure = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, step=1)
cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, step=1)
fasting_blood_sugar = st.selectbox("Fasting Blood Sugar > 120 mg/dl (0/1)", options=[0, 1])
resting_ecg = st.selectbox("Resting ECG Results (0-2)", options=[0, 1, 2])
max_heart_rate = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, step=1)
exercise_induced_angina = st.selectbox("Exercise-Induced Angina (0/1)", options=[0, 1])
st_depression = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=6.0, step=0.1)
st_slope = st.selectbox("Slope of Peak Exercise ST Segment (0-2)", options=[0, 1, 2])
num_major_vessels = st.selectbox("Number of Major Vessels (0-3)", options=[0, 1, 2, 3])
thalassemia = st.selectbox("Thalassemia Test Result (0-3)", options=[0, 1, 2, 3])

# Submit button
if st.button("Predict"):
    # Prepare input data for the FastAPI backend
    input_data = {
        "age": age,
        "sex": sex,
        "chest_pain_type": chest_pain_type,
        "resting_blood_pressure": resting_blood_pressure,
        "cholesterol": cholesterol,
        "fasting_blood_sugar": fasting_blood_sugar,
        "resting_ecg": resting_ecg,
        "max_heart_rate": max_heart_rate,
        "exercise_induced_angina": exercise_induced_angina,
        "st_depression": st_depression,
        "st_slope": st_slope,
        "num_major_vessels": num_major_vessels,
        "thalassemia": thalassemia,
    }

    # Send a POST request to the FastAPI backend
    try:
        response = requests.post(API_URL, json=input_data)
        if response.status_code == 200:
            result = response.json()
            st.success(f"Prediction: {'Disease' if result['prediction'] == 1 else 'No Disease'}")
            st.info(f"Risk Score: {result['risk_score']:.2f}")
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"Failed to connect to the API: {e}")