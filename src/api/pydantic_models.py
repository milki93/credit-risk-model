from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class PredictionInput(BaseModel):
    """Input data model for prediction endpoint"""
    customer_id: str = Field(..., description="Unique customer identifier")
    transaction_amount: float = Field(..., description="Amount of the transaction")
    transaction_count: int = Field(..., description="Number of transactions")
    avg_transaction_amount: float = Field(..., description="Average transaction amount")
    days_since_last_transaction: int = Field(..., description="Days since last transaction")
    # Add other features as needed
    
    class Config:
        schema_extra = {
            "example": {
                "customer_id": "C12345",
                "transaction_amount": 150.75,
                "transaction_count": 5,
                "avg_transaction_amount": 120.50,
                "days_since_last_transaction": 3
            }
        }

class PredictionOutput(BaseModel):
    """Output data model for prediction endpoint"""
    customer_id: str
    risk_score: float = Field(..., ge=0, le=1, description="Predicted risk score (0-1)")
    risk_category: str = Field(..., description="Risk category based on score")
    model_version: str = Field(..., description="Version of the model used for prediction")
    
    class Config:
        schema_extra = {
            "example": {
                "customer_id": "C12345",
                "risk_score": 0.25,
                "risk_category": "LOW",
                "model_version": "1.0.0"
            }
        }

class HealthCheck(BaseModel):
    """Health check response model"""
    status: str
    model_version: str
    timestamp: str
