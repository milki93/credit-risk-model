import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from src.api.pydantic_models import PredictionInput

client = TestClient(app)

def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_version" in data
    assert "timestamp" in data

def test_predict():
    """Test prediction endpoint with sample data"""
    test_data = {
        "customer_id": "test_customer_123",
        "transaction_amount": 150.75,
        "transaction_count": 5,
        "avg_transaction_amount": 120.50,
        "days_since_last_transaction": 3
    }
    
    # Mock the model's predict_proba method
    from unittest.mock import patch, MagicMock
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = [[0.8, 0.2]]  # 20% risk score
    
    with patch('src.api.main.model', mock_model):
        response = client.post("/predict", json=test_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "customer_id" in data
        assert "risk_score" in data
        assert "risk_category" in data
        assert "model_version" in data
        assert 0 <= data["risk_score"] <= 1
        assert data["risk_category"] in ["LOW", "MEDIUM", "HIGH"]

def test_predict_invalid_input():
    """Test prediction with invalid input data"""
    invalid_data = {"invalid": "data"}
    response = client.post("/predict", json=invalid_data)
    assert response.status_code == 422  # Validation error

def test_model_not_loaded(monkeypatch):
    """Test behavior when model fails to load"""
    # Import the module to access the actual model object
    from src.api import main
    
    # Save the original model
    original_model = main.model
    
    # Set model to None to simulate loading failure
    monkeypatch.setattr(main, 'model', None)
    
    test_data = {
        "customer_id": "test_customer_123",
        "transaction_amount": 150.75,
        "transaction_count": 5,
        "avg_transaction_amount": 120.50,
        "days_since_last_transaction": 3
    }
    
    response = client.post("/predict", json=test_data)
    assert response.status_code == 503  # Service Unavailable
    assert "Model not loaded" in response.json()["detail"]
    
    # Restore the original model
    monkeypatch.setattr(main, 'model', original_model)
