# Heart Disease Prediction Project

This project implements a machine learning model to predict heart disease risk based on clinical parameters. It includes a FastAPI-based web service that exposes the prediction model via REST API.

## Project Structure

```
├── app/                    # FastAPI application
│   ├── main.py             # Main FastAPI app
│   └── model_schema.py     # Data models and schemas
├── data/                   # Data directory
│   └── heart_disease_dataset.csv  # Heart disease dataset
├── models/                 # Directory for saved models
│   └── best_model.pkl      # Trained ML model pipeline
├── scripts/                # Training and evaluation scripts
│   ├── 01_train_model.py   # Model training script
│   └── 02_evaluate_test.py # Model evaluation script
├── src/                    # Source code modules
│   ├── data_processor.py   # Data preprocessing utilities
│   └── utils.py            # Helper functions
├── test_api.py             # API testing script
├── requirements.txt        # Project dependencies
├── Dockerfile              # Docker configuration
└── README.md               # This file
```

## Prerequisites

- Python 3.8+
- Virtual environment (recommended)

## Setup Instructions

### 1. Clone the repository

```bash
git clone <repository-url>
cd CapstoneProject2
```

### 2. Create and activate a virtual environment

```bash
# Windows
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux/Mac
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Running the Project

### 1. Train the Machine Learning Model

```bash
python scripts/01_train_model.py
```

This script:

- Loads and preprocesses the heart disease dataset
- Splits data into training and testing sets
- Trains multiple models with hyperparameter optimization
- Selects the best model based on ROC-AUC score
- Saves the model pipeline to `models/best_model.pkl`

### 2. Evaluate the Model (Optional)

```bash
python scripts/02_evaluate_test.py
```

This script:

- Evaluates the trained model on the test dataset
- Provides comprehensive performance metrics
- Generates feature importance analysis

### 3. Start the API Service

```bash
uvicorn app.main:app --reload
```

The FastAPI service will start on http://127.0.0.1:8000

### 4. Test the API

You can test the API using the provided test script:

```bash
python test_api.py
```

## Using the API

### Interactive API Documentation

The API includes interactive documentation:

1. Swagger UI: http://127.0.0.1:8000/docs
2. ReDoc: http://127.0.0.1:8000/redoc

### API Endpoints

#### 1. Prediction Endpoint

```
POST /predict
```

Request Body:

```json
{
  "age": 65,
  "sex": 1,
  "chest_pain_type": 2,
  "resting_blood_pressure": 160,
  "cholesterol": 286,
  "fasting_blood_sugar": 1,
  "resting_ecg": 2,
  "max_heart_rate": 108,
  "exercise_induced_angina": 1,
  "st_depression": 2.5,
  "st_slope": 2,
  "num_major_vessels": 3,
  "thalassemia": 3
}
```

Response:

```json
{
  "prediction": 1,
  "risk_score": 0.89
}
```

Where:

- `prediction`: Binary classification (0: No disease, 1: Disease)
- `risk_score`: Probability of having heart disease (0-1)

#### 2. Health Check Endpoint

```
GET /health
```

Response:

```json
{
  "status": "ok",
  "model_loaded": true
}
```

## Feature Descriptions

The model uses the following clinical features to predict heart disease:

| Feature                 | Description                                    | Range/Values                  |
| ----------------------- | ---------------------------------------------- | ----------------------------- |
| age                     | Patient age in years                           | 20-80                         |
| sex                     | Patient sex                                    | 0: Female, 1: Male            |
| chest_pain_type         | Type of chest pain                             | 0-3 (0: Typical angina, etc.) |
| resting_blood_pressure  | Resting blood pressure in mm Hg                | 80-200                        |
| cholesterol             | Serum cholesterol in mg/dl                     | 100-600                       |
| fasting_blood_sugar     | Fasting blood sugar > 120 mg/dl                | 0: False, 1: True             |
| resting_ecg             | Resting electrocardiographic results           | 0-2                           |
| max_heart_rate          | Maximum heart rate achieved                    | 60-220                        |
| exercise_induced_angina | Exercise-induced angina                        | 0: No, 1: Yes                 |
| st_depression           | ST depression induced by exercise              | 0.0-6.0                       |
| st_slope                | Slope of peak exercise ST segment              | 0-2                           |
| num_major_vessels       | Number of major vessels colored by fluoroscopy | 0-3                           |
| thalassemia             | Thalassemia test result                        | 0-3                           |

## Docker Support

To build and run using Docker:

```bash
# Build the Docker image
docker build -t heart-disease-predictor .

# Run the Docker container
docker run -p 8000:8000 heart-disease-predictor
```

## Model Performance

The current model achieves:

- ROC-AUC: ~0.76 on the test set
- Accuracy: ~69%
- Precision: ~69%
- Recall: ~77%


