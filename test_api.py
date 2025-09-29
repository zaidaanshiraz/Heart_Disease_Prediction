#!/usr/bin/env python3
"""
Simple script to test the Heart Disease Prediction API
"""
import requests
import json

# API endpoint
API_URL = "http://127.0.0.1:8000"

def test_health_check():
    """Test the health check endpoint"""
    response = requests.get(f"{API_URL}/health")
    print("Health Check:")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_prediction():
    """Test the prediction endpoint with sample data"""
    # Sample patient data - high risk
    high_risk_patient = {
        "age": 65,
        "sex": 1,  # Male
        "chest_pain_type": 2,
        "resting_blood_pressure": 160,
        "cholesterol": 280,
        "fasting_blood_sugar": 1,
        "resting_ecg": 2,
        "max_heart_rate": 120,
        "exercise_induced_angina": 1,
        "st_depression": 2.5,
        "st_slope": 2,
        "num_major_vessels": 3,
        "thalassemia": 3
    }
    
    # Sample patient data - low risk
    low_risk_patient = {
        "age": 40,
        "sex": 0,  # Female
        "chest_pain_type": 0,
        "resting_blood_pressure": 120,
        "cholesterol": 180,
        "fasting_blood_sugar": 0,
        "resting_ecg": 0,
        "max_heart_rate": 170,
        "exercise_induced_angina": 0,
        "st_depression": 0.2,
        "st_slope": 0,
        "num_major_vessels": 0,
        "thalassemia": 1
    }
    
    print("Testing High Risk Patient:")
    response = requests.post(f"{API_URL}/predict", json=high_risk_patient)
    print(f"Status Code: {response.status_code}")
    print(f"Input: {json.dumps(high_risk_patient, indent=2)}")
    if response.status_code == 200:
        result = response.json()
        print(f"Prediction: {result['prediction']}")
        print(f"Risk Score: {result['risk_score']:.4f}")
        print(f"Risk Level: {'High Risk' if result['risk_score'] > 0.5 else 'Low Risk'}")
    else:
        print(f"Error: {response.text}")
        
    print("\nTesting Low Risk Patient:")
    response = requests.post(f"{API_URL}/predict", json=low_risk_patient)
    print(f"Status Code: {response.status_code}")
    print(f"Input: {json.dumps(low_risk_patient, indent=2)}")
    if response.status_code == 200:
        result = response.json()
        print(f"Prediction: {result['prediction']}")
        print(f"Risk Score: {result['risk_score']:.4f}")
        print(f"Risk Level: {'High Risk' if result['risk_score'] > 0.5 else 'Low Risk'}")
    else:
        print(f"Error: {response.text}")

if __name__ == "__main__":
    try:
        test_health_check()
        test_prediction()
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to the API server.")
        print("Make sure the FastAPI server is running on http://127.0.0.1:8000")
        print("Run: uvicorn app.main:app --reload --port 8000")