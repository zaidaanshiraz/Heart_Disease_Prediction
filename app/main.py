# Step 11: FastAPI Backend Development
import os
import sys
from pathlib import Path
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd

# Add project root to Python path and set correct model path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
MODEL_PATH = project_root / 'models' / 'best_model.pkl'

# Load the trained model (best_model.pkl contains the full Pipeline)
try:
    MODEL = joblib.load(MODEL_PATH)
except FileNotFoundError:
    raise RuntimeError(f"Model file not found at {MODEL_PATH}. Run 01_train_model.py first.")

app = FastAPI(
    title="Heart Disease Prediction API",
    version="1.0.0",
    description="Predicts heart disease risk based on clinical parameters.",
    docs_url="/docs",  # URL for Swagger UI (default is /docs)
    redoc_url="/redoc",  # URL for ReDoc UI (default is /redoc)
    openapi_url="/openapi.json",  # URL for OpenAPI schema
    swagger_ui_parameters={"defaultModelsExpandDepth": -1}  # Hide schemas by default
)

# 4.1 Pydantic Model for Input Validation (Model is defined here, or in model_schema.py)
class HeartInput(BaseModel):
    age: int = Field(..., ge=20, le=80, description="Patient age (years)")
    sex: int = Field(..., ge=0, le=1, description="Gender (0=Female, 1=Male)")
    chest_pain_type: int = Field(..., ge=0, le=3, description="Chest pain type (0-3)")
    resting_blood_pressure: int = Field(..., ge=80, le=200, description="Resting blood pressure (mm Hg)")
    cholesterol: int = Field(..., ge=100, le=600, description="Serum cholesterol (mg/dl)")
    fasting_blood_sugar: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl (0/1)")
    resting_ecg: int = Field(..., ge=0, le=2, description="Resting electrocardiographic results (0-2)")
    max_heart_rate: int = Field(..., ge=60, le=220, description="Maximum heart rate achieved")
    exercise_induced_angina: int = Field(..., ge=0, le=1, description="Exercise-induced angina (0/1)")
    st_depression: float = Field(..., ge=0.0, le=6.0, description="ST depression induced by exercise")
    st_slope: int = Field(..., ge=0, le=2, description="Slope of peak exercise ST segment (0-2)")
    num_major_vessels: int = Field(..., ge=0, le=3, description="Number of major vessels (0-3)")
    thalassemia: int = Field(..., ge=0, le=3, description="Thalassemia test result (0-3)")

class PredictionOutput(BaseModel):
    prediction: int = Field(..., description="Predicted class (0: No disease, 1: Disease)")
    risk_score: float = Field(..., description="Probability of having heart disease")


@app.post(
    "/predict", 
    response_model=PredictionOutput,
    summary="Predict Heart Disease Risk",
    description="""
    Predicts the risk of heart disease based on patient clinical data.
    
    The model returns:
    - A binary prediction (0: No disease, 1: Disease present)
    - A risk score (probability between 0-1)
    
    A risk score above 0.5 indicates a high likelihood of heart disease.
    """,
    responses={
        200: {"description": "Successful prediction"},
        422: {"description": "Validation error in input data"},
        500: {"description": "Server error during prediction"}
    },
    tags=["Prediction"]
)
def predict_heart_disease(data: HeartInput):
    """Predicts the presence of heart disease for a given patient profile."""
    
    # Convert input Pydantic model to a DataFrame row
    input_df = pd.DataFrame([data.model_dump()])
    
    # Predict probability (risk score)
    try:
        # Predict_proba uses the full pipeline (preprocessor + model)
        risk_score = MODEL.predict_proba(input_df)[:, 1][0]
        prediction = 1 if risk_score >= 0.5 else 0
    except Exception as e:
        # Handle errors during prediction (e.g., unexpected data)
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    return PredictionOutput(prediction=prediction, risk_score=risk_score)

@app.get(
    "/health",
    summary="API Health Check",
    description="Returns the health status of the API and whether the model is loaded.",
    responses={
        200: {"description": "API is healthy and model is loaded"},
        500: {"description": "API is experiencing issues"}
    },
    tags=["System"]
)
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "model_loaded": True}

@app.get(
    "/",
    summary="Root Endpoint",
    description="Welcome message for the Heart Disease Prediction API.",
    tags=["System"]
)
def root():
    """Root endpoint to provide a welcome message."""
    return {"message": "Welcome to the Heart Disease Prediction API!"}

# To run the app: uvicorn app.main:app --reload