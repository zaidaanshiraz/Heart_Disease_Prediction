# app/model_schema.py
from pydantic import BaseModel, Field

# Define the acceptable ranges for the fields based on your data and domain knowledge
class HeartInput(BaseModel):
    """Schema for incoming patient data used for prediction."""
    age: int = Field(..., ge=20, le=80)
    sex: int = Field(..., ge=0, le=1)
    chest_pain_type: int = Field(..., ge=0, le=3)
    resting_blood_pressure: int = Field(..., ge=90, le=180)
    cholesterol: int = Field(..., ge=120, le=350)
    fasting_blood_sugar: int = Field(..., ge=0, le=1)
    resting_ecg: int = Field(..., ge=0, le=2)
    max_heart_rate: int = Field(..., ge=80, le=200)
    exercise_induced_angina: int = Field(..., ge=0, le=1)
    st_depression: float = Field(..., ge=0.0, le=5.0)
    st_slope: int = Field(..., ge=0, le=2)
    num_major_vessels: int = Field(..., ge=0, le=3)
    thalassemia: int = Field(..., ge=0, le=3)

class PredictionOutput(BaseModel):
    """Schema for the prediction response."""
    prediction: int = Field(..., description="Predicted class (0: No disease, 1: Disease)")
    risk_score: float = Field(..., description="Probability of having heart disease")