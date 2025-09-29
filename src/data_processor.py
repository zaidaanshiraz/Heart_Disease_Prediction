import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Define feature types based on project description and data inspection
NUMERICAL_FEATURES = ['age', 'resting_blood_pressure', 'cholesterol', 'max_heart_rate', 'st_depression']
CATEGORICAL_FEATURES = ['sex', 'chest_pain_type', 'fasting_blood_sugar', 'resting_ecg', 
                        'exercise_induced_angina', 'st_slope', 'num_major_vessels', 'thalassemia']
TARGET = 'heart_disease'

def get_preprocessor():
    """
    Creates and returns the ColumnTransformer pipeline for data preprocessing.
    """
    # 1. Pipeline for Numerical Features (Impute then Scale)
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')), 
        ('scaler', StandardScaler()) 
    ])

    # 2. Pipeline for Categorical Features (One-Hot Encode)
    categorical_pipeline = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # 3. Combine pipelines using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, NUMERICAL_FEATURES),
            ('cat', categorical_pipeline, CATEGORICAL_FEATURES)
        ],
        remainder='passthrough'
    )
    return preprocessor

if __name__ == '__main__':
    # Simple test to verify the processor
    print("Data processor created successfully.")
    get_preprocessor()