import pytest
from datetime import datetime
from src.api.pydantic_models import PredictionInput, PredictionOutput, HealthCheck

def test_prediction_input_validation():
    """Test validation of PredictionInput model"""
    # Valid input
    valid_input = {
        "customer_id": "C12345",
        "transaction_amount": 150.75,
        "transaction_count": 5,
        "avg_transaction_amount": 120.50,
        "days_since_last_transaction": 3
    }
    prediction_input = PredictionInput(**valid_input)
    assert prediction_input.customer_id == "C12345"
    assert prediction_input.transaction_amount == 150.75

    # Test with missing required field
    invalid_input = valid_input.copy()
    del invalid_input["customer_id"]
    with pytest.raises(ValueError):
        PredictionInput(**invalid_input)

def test_prediction_output_validation():
    """Test validation of PredictionOutput model"""
    # Valid output
    valid_output = {
        "customer_id": "C12345",
        "risk_score": 0.25,
        "risk_category": "LOW",
        "model_version": "1.0.0"
    }
    prediction_output = PredictionOutput(**valid_output)
    assert prediction_output.risk_score == 0.25
    assert prediction_output.risk_category == "LOW"

    # Test with invalid risk score
    invalid_output = valid_output.copy()
    invalid_output["risk_score"] = 1.5  # Invalid, should be between 0 and 1
    with pytest.raises(ValueError):
        PredictionOutput(**invalid_output)

def test_health_check_model():
    """Test HealthCheck model"""
    health_check = HealthCheck(
        status="OK",
        model_version="1.0.0",
        timestamp="2023-07-22T21:34:12.123456"
    )
    assert health_check.status == "OK"
    assert health_check.model_version == "1.0.0"
    assert isinstance(health_check.timestamp, str)
