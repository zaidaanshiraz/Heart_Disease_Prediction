# Step 12: Docker Configuration
# Use a lightweight official Python image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code, including the model and application files
# We copy all necessary folders: src, scripts (though not run in Docker), app, and data
COPY . .

# Ensure the models directory exists and the model file is copied
# NOTE: The model file (best_model.pkl) must be created by running 01_train_model.py
# BEFORE building the Docker image.
# We copy the model explicitly to the container's working directory
COPY models/best_model.pkl /app/models/best_model.pkl

# Expose the port the API will run on
EXPOSE 8000

# Command to run the application using Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]