import pathlib
import pandas as pd
import streamlit as st
import joblib

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="centered")

MODEL_PATH = pathlib.Path(__file__).parent / "models" / "best_model.pkl"

@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Failed to load model at {MODEL_PATH}: {e}")
        return None

model = load_model()

st.title("Heart Disease Prediction")
st.caption("Local inference (no FastAPI call)")

with st.form("input_form"):
    col1, col2 = st.columns(2)
    age = col1.number_input("Age (20-80)", 20, 80, 50)
    sex = col2.selectbox("Sex", [0, 1], format_func=lambda v: "Female" if v == 0 else "Male")
    chest_pain_type = col1.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
    resting_blood_pressure = col2.number_input("Resting BP (80–200)", 80, 200, 120)
    cholesterol = col1.number_input("Cholesterol (100–600)", 100, 600, 200)
    fasting_blood_sugar = col2.selectbox("Fasting Blood Sugar >120 (0/1)", [0, 1])
    resting_ecg = col1.selectbox("Resting ECG (0–2)", [0, 1, 2])
    max_heart_rate = col2.number_input("Max Heart Rate (60–220)", 60, 220, 150)
    exercise_induced_angina = col1.selectbox("Exercise Induced Angina (0/1)", [0, 1])
    st_depression = col2.number_input("ST Depression (0.0–6.0)", 0.0, 6.0, 1.0, step=0.1)
    st_slope = col1.selectbox("ST Slope (0–2)", [0, 1, 2])
    num_major_vessels = col2.selectbox("Num Major Vessels (0–3)", [0, 1, 2, 3])
    thalassemia = col1.selectbox("Thalassemia (0–3)", [0, 1, 2, 3])
    submitted = st.form_submit_button("Predict")

if submitted:
    if model is None:
        st.error("Model not available.")
    else:
        row = {
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
            "thalassemia": thalassemia
        }
        df = pd.DataFrame([row])
        try:
            proba = model.predict_proba(df)[:, 1][0]
            pred = 1 if proba >= 0.5 else 0
            st.success(f"Prediction: {'Disease' if pred == 1 else 'No Disease'}")
            st.metric("Risk Score", f"{proba:.3f}")
        except Exception as e:
            st.error(f"Inference error: {e}")

st.markdown("----")

# Add Model Evaluation & Performance Analysis table (as per your request)
st.markdown("""
### Model Evaluation & Performance Analysis

| Model               | Accuracy | Precision | Recall | F1 Score | ROC-AUC | True Healthy | False Sick | False Healthy | True Sick |
|---------------------|----------|-----------|--------|----------|---------|--------------|------------|---------------|-----------|
| Logistic Regression | 68.8%    | 76.9%     | 65.2%  | 70.6%    | 71.1%   | 25           | 9          | 16            | 30        |
| Random Forest       | 62.5%    | 66.7%     | 69.6%  | 68.1%    | 61.9%   | 18           | 16         | 14            | 32        |
| SVM                 | 57.5%    | 63.0%     | 63.0%  | 63.0%    | 63.7%   | 17           | 17         | 17            | 29        |
| Decision Tree       | 55.0%    | 60.4%     | 63.0%  | 61.7%    | 53.6%   | 15           | 19         | 17            | 29        |
""")

st.caption("If you prefer using the FastAPI backend, deploy it separately and change this app to call its public URL.")

# Example (if you deploy FastAPI):
# API_URL = "https://your-fastapi-host/predict"
# Use requests.post(API_URL, json=row)
