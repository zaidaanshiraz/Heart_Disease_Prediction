import sys
import os
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import joblib # Used for model serialization
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, make_scorer, classification_report

# Import models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Import the preprocessor
from src.data_processor import get_preprocessor, TARGET

# --- Configuration ---
DATA_PATH = 'data/heart_disease_dataset.csv'
MODEL_SAVE_PATH = 'models/best_model.pkl'
SPLIT_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

def load_data(path):
    """Loads data, splits features/target."""
    df = pd.read_csv(path)
    X = df.drop(TARGET, axis=1)
    y = df[TARGET]
    return X, y

def define_model_grids():
    """Defines models, pipelines, and hyperparameter grids for GridSearch."""
    preprocessor = get_preprocessor()
    
    # NOTE: The 'classifier__' prefix is crucial for targeting the model in the pipeline
    grids = {
        'Logistic Regression': (
            LogisticRegression(random_state=RANDOM_STATE, max_iter=2000), 
            {
                'classifier__C': [0.1, 1, 10],
                'classifier__penalty': ['l2'],  # Remove l1 to avoid solver conflicts
                'classifier__solver': ['lbfgs']  # Use lbfgs for better convergence
            }
        ),
        'Random Forest': (
            RandomForestClassifier(random_state=RANDOM_STATE),
            {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [5, 10, None],
                'classifier__max_features': ['sqrt', 'log2']
            }
        ),
        'Decision Tree': (
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            {
                'classifier__max_depth': [5, 10, 15, None],
                'classifier__min_samples_split': [2, 5, 10],
                'classifier__min_samples_leaf': [1, 2, 4]
            }
        ),
        'SVC': (
            SVC(random_state=RANDOM_STATE, probability=True),  # Enable probability for ROC-AUC
            {
                'classifier__C': [0.1, 1, 10],
                'classifier__kernel': ['rbf', 'linear'],
                'classifier__gamma': ['scale', 'auto']
            }
        )
    }
    
    # Wrap models in pipelines with the preprocessor
    pipelines = {name: Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)]) for name, (model, _) in grids.items()}
    param_grids = {name: grid for name, (_, grid) in grids.items()}
    
    return pipelines, param_grids

def train_and_optimize():
    """Performs full training, optimization, and model saving."""
    X, y = load_data(DATA_PATH)
    
    # Phase 1: Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=SPLIT_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Data split: Train={X_train.shape[0]}, Test={X_test.shape[0]}")
    
    pipelines, param_grids = define_model_grids()
    best_roc_auc = 0.0
    best_model = None
    best_model_name = ""
    
    scoring = 'roc_auc'  # Use built-in roc_auc scorer instead of make_scorer
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    print("\n--- Phase 3: Hyperparameter Optimization ---")
    
    for name, pipeline in pipelines.items():
        print(f"Optimizing {name}...")
        
        # Grid Search with Cross-Validation
        grid_search = GridSearchCV(
            estimator=pipeline, 
            param_grid=param_grids[name], 
            scoring=scoring, 
            cv=cv, 
            n_jobs=-1, 
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"  Best ROC-AUC: {grid_search.best_score_:.4f}")
        print(f"  Best Params: {grid_search.best_params_}")
        
        # Only consider valid scores (not NaN)
        if not pd.isna(grid_search.best_score_) and grid_search.best_score_ > best_roc_auc:
            best_roc_auc = grid_search.best_score_
            best_model = grid_search.best_estimator_
            best_model_name = name

    # Phase 4: Final Evaluation and Saving
    print(f"\n--- Final Model Selection and Evaluation ---")
    
    if best_model is None:
        print("❌ Error: No valid model was found! All models produced NaN scores.")
        print("This might be due to data issues or incompatible hyperparameters.")
        return
    
    print(f"Selected Best Model: {best_model_name} (ROC-AUC: {best_roc_auc:.4f})")
    
    # Create models directory if it doesn't exist
    import os
    os.makedirs('models', exist_ok=True)
    
    # Save the entire best pipeline (preprocessor + model)
    joblib.dump(best_model, MODEL_SAVE_PATH)
    print(f"Best model saved to {MODEL_SAVE_PATH}")
    
    # Final evaluation on the test set
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    test_roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"Final Test Set ROC-AUC: {test_roc_auc:.4f}")
    
    # Check for Excellence Indicator
    if test_roc_auc > 0.85:
        print("🏆 Excellence Indicator Achieved: Test ROC-AUC > 0.85!")
    else:
        print(f"⚠️ Test ROC-AUC of {test_roc_auc:.4f} is below the 0.85 target.")


if __name__ == '__main__':
    train_and_optimize()