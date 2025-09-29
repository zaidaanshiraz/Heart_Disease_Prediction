# scripts/02_evaluate_test.py
import sys
import os
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from src.data_processor import get_preprocessor, TARGET
from src.utils import get_full_metrics, plot_roc_curve, get_feature_names
from sklearn.inspection import permutation_importance
import numpy as np

# --- Configuration ---
DATA_PATH = 'data/heart_disease_dataset.csv'
MODEL_PATH = 'models/best_model.pkl'
SPLIT_SIZE = 0.2
RANDOM_STATE = 42

def load_data(path):
    """Loads data, splits features/target."""
    df = pd.read_csv(path)
    X = df.drop(TARGET, axis=1)
    y = df[TARGET]
    return X, y

def final_evaluation():
    """Loads the best model and performs final evaluation on the test set."""
    X, y = load_data(DATA_PATH)
    
    # Re-split data to get the exact same Test Set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=SPLIT_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    try:
        # Load the entire best pipeline (preprocessor + model)
        best_pipeline = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        print(f"Error: Best model not found at {MODEL_PATH}. Run 01_train_model.py first.")
        return

    # --- 1. Model Evaluation (Performance Metrics) ---
    print("\n--- Final Model Performance on Test Set ---")
    metrics, class_report, conf_matrix = get_full_metrics(best_pipeline, X_test, y_test)
    
    metrics_df = pd.Series(metrics).to_frame(name='Score')
    
    # Try to use markdown format, fallback to regular print if tabulate is not available
    try:
        print(metrics_df.to_markdown(numalign="left", stralign="left"))
    except ImportError:
        print("Tabulate not available, using simple format:")
        print(metrics_df.to_string())
    
    print("\nClassification Report:\n", class_report)
    print("\nConfusion Matrix:\n", conf_matrix)
    
    # --- 2. ROC Curve Plot ---
    # plot_roc_curve(best_pipeline, X_test, y_test, title='Test Set ROC Curve')

    # --- 3. Feature Analysis (Permutation Importance) --- [cite: 17]
    print("\n--- Feature Importance Analysis (Permutation) ---")
    
    # Permutation importance works directly on the full pipeline
    r = permutation_importance(best_pipeline, X_test, y_test, 
                               n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
    
    # Get final feature names from the preprocessor within the pipeline
    final_feature_names = get_feature_names(best_pipeline.named_steps['preprocessor'])
    
    if final_feature_names is None:
        # Fallback: create generic feature names
        final_feature_names = [f'feature_{i}' for i in range(len(r.importances_mean))]
        print("Using generic feature names due to extraction failure.")
    
    print(f"Number of features: {len(final_feature_names)}")
    print(f"Number of importance values: {len(r.importances_mean)}")
    
    # Ensure arrays have the same length
    if len(final_feature_names) != len(r.importances_mean):
        print(f"Mismatch in lengths! Using generic names.")
        final_feature_names = [f'feature_{i}' for i in range(len(r.importances_mean))]
    
    importance_df = pd.DataFrame({
        'Feature': final_feature_names,
        'Importance_Mean': r.importances_mean,
        'Importance_Std': r.importances_std
    }).sort_values(by='Importance_Mean', ascending=False)
    
    # Try to use markdown format, fallback to regular print if tabulate is not available
    try:
        print(importance_df.head(10).to_markdown(index=False, numalign="left", stralign="left"))
    except ImportError:
        print("Tabulate not available, using simple format:")
        print(importance_df.head(10).to_string(index=False))


if __name__ == '__main__':
    final_evaluation()