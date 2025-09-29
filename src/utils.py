# src/utils.py
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, 
    confusion_matrix, classification_report, roc_curve
)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def get_full_metrics(model, X, y, threshold=0.5):
    """
    Calculates and returns all required evaluation metrics for a binary classifier.
    """
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    # Apply threshold for hard predictions
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Calculate Core Metrics
    metrics = {
        'Accuracy': accuracy_score(y, y_pred),
        'Precision': precision_score(y, y_pred, zero_division=0),
        'Recall (Sensitivity)': recall_score(y, y_pred, zero_division=0),
        'F1-Score': f1_score(y, y_pred, zero_division=0),
        'ROC-AUC': roc_auc_score(y, y_pred_proba)
    }
    
    # Calculate Specificity (True Negative Rate)
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()
    metrics['Specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return metrics, classification_report(y, y_pred), cm

def plot_roc_curve(model, X, y, title='ROC Curve'):
    """
    Generates and displays the ROC curve plot.
    """
    y_pred_proba = model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, y_pred_proba)
    roc_auc = roc_auc_score(y, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Recall/Sensitivity)')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()

def get_feature_names(preprocessor):
    """
    Extracts the final feature names after preprocessing/One-Hot Encoding.
    This is essential for interpreting tree-based feature importance.
    """
    try:
        # Get all feature names from the fitted preprocessor
        feature_names = preprocessor.get_feature_names_out()
        return list(feature_names)
    except Exception as e:
        print(f"Warning: Could not extract feature names from preprocessor: {e}")
        # If we can't get the names, we'll return None and let the caller handle it
        return None