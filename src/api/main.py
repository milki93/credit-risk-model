import os
import logging
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import mlflow
import pandas as pd
from typing import List, Dict, Any
import joblib

from .pydantic_models import PredictionInput, PredictionOutput, HealthCheck

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Credit Risk Prediction API",
    description="API for predicting credit risk based on transaction history",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to store the loaded model
model = None
model_version = "1.0.0"

# Load model at startup
@app.on_event("startup")
async def load_model():
    """Load the trained model from MLflow or local storage"""
    global model
    try:
        # Try to load from MLflow first
        try:
            model_uri = "models:/credit_risk_model/Production"
            model = mlflow.pyfunc.load_model(model_uri)
            logger.info("Model loaded from MLflow Model Registry")
        except Exception as e:
            # Fallback to local model
            model_path = os.getenv("MODEL_PATH", "models/credit_risk_model.joblib")
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                logger.info(f"Model loaded from local path: {model_path}")
            else:
                logger.error(f"Model not found at {model_path}")
                raise FileNotFoundError(f"Model not found at {model_path}")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

@app.get("/health", response_model=HealthCheck, status_code=status.HTTP_200_OK)
async def health_check():
    """Health check endpoint"""
    status = "OK" if model is not None else "Model not loaded"
    return {
        "status": status,
        "model_version": model_version,
        "timestamp": pd.Timestamp.now().isoformat()
    }

@app.post("/predict", response_model=PredictionOutput, status_code=status.HTTP_200_OK)
async def predict(input_data: PredictionInput):
    """
    Make a prediction using the trained credit risk model
    
    - **input_data**: Input features for prediction
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        # Convert input to DataFrame
        input_dict = input_data.dict()
        customer_id = input_dict.pop('customer_id')
        
        # Create DataFrame with the same structure as training data
        input_df = pd.DataFrame([input_dict])
        
        # Make prediction
        risk_score = float(model.predict_proba(input_df)[0][1])  # Probability of class 1 (high risk)
        
        # Determine risk category
        if risk_score < 0.3:
            risk_category = "LOW"
        elif risk_score < 0.7:
            risk_category = "MEDIUM"
        else:
            risk_category = "HIGH"
        
        return {
            "customer_id": customer_id,
            "risk_score": risk_score,
            "risk_category": risk_category,
            "model_version": model_version
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error making prediction: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
