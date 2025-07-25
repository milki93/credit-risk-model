"""
Shared test fixtures and utilities.
"""
import os
import sys
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add src directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Sample test data
@pytest.fixture
def sample_transaction_data():
    """Create sample transaction data for testing."""
    data = {
        'TransactionId': [f'TXN{i:04d}' for i in range(1, 11)],
        'CustomerId': ['CUST001', 'CUST001', 'CUST002', 'CUST002', 'CUST003', 
                      'CUST003', 'CUST004', 'CUST004', 'CUST005', 'CUST005'],
        'Amount': [100.0, -50.0, 200.0, 150.0, 75.0, 125.0, 300.0, -100.0, 50.0, 25.0],
        'TransactionStartTime': [
            '2023-01-01 10:00:00', '2023-01-02 11:30:00', '2023-01-01 09:15:00',
            '2023-01-03 14:45:00', '2023-01-02 16:20:00', '2023-01-04 10:30:00',
            '2023-01-03 11:45:00', '2023-01-05 09:30:00', '2023-01-04 14:15:00',
            '2023-01-06 16:45:00'
        ],
        'ProductCategory': ['Electronics', 'Electronics', 'Books', 'Books', 'Clothing',
                          'Electronics', 'Books', 'Clothing', 'Electronics', 'Books'],
        'ChannelId': ['Web', 'Mobile', 'Web', 'Mobile', 'Web', 'Web', 'Mobile', 'Web', 'Mobile', 'Web'],
        'ProviderId': ['P001', 'P001', 'P002', 'P002', 'P003', 'P001', 'P002', 'P003', 'P001', 'P002'],
        'CurrencyCode': ['USD', 'USD', 'EUR', 'EUR', 'USD', 'USD', 'EUR', 'USD', 'USD', 'EUR'],
        'CountryCode': ['US', 'US', 'UK', 'UK', 'US', 'US', 'UK', 'US', 'US', 'UK']
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_processed_data(sample_transaction_data):
    """Create sample processed data with features and target."""
    # Add some basic features
    df = sample_transaction_data.copy()
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    
    # Add RFM features
    snapshot_date = df['TransactionStartTime'].max() + timedelta(days=1)
    
    # Recency
    recency = df.groupby('CustomerId')['TransactionStartTime'].max().reset_index()
    recency['recency'] = (snapshot_date - recency['TransactionStartTime']).dt.days
    recency = recency[['CustomerId', 'recency']]
    
    # Frequency
    frequency = df['CustomerId'].value_counts().reset_index()
    frequency.columns = ['CustomerId', 'frequency']
    
    # Monetary
    monetary = df.groupby('CustomerId')['Amount'].sum().abs().reset_index()
    monetary.columns = ['CustomerId', 'monetary']
    
    # Merge features
    features = pd.merge(recency, frequency, on='CustomerId')
    features = pd.merge(features, monetary, on='CustomerId')
    
    # Add target (20% high risk)
    features['is_high_risk'] = np.random.choice([0, 1], size=len(features), p=[0.8, 0.2])
    
    return features

@pytest.fixture
def sample_training_data(sample_processed_data):
    """Create sample training data with features and target."""
    df = sample_processed_data.copy()
    X = df[['recency', 'frequency', 'monetary']]
    y = df['is_high_risk']
    return X, y

@pytest.fixture
def sample_model():
    """Create a sample trained model for testing."""
    from sklearn.ensemble import RandomForestClassifier
    
    # Create and train a simple model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    X = np.random.rand(100, 3)
    y = np.random.randint(0, 2, 100)
    model.fit(X, y)
    
    return model

@pytest.fixture
def sample_api_client():
    """Create a test client for the API."""
    from fastapi.testclient import TestClient
    from src.api.main import app
    
    # Mock the model loading
    with patch('joblib.load') as mock_load:
        # Create a mock model
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0])  # Low risk
        mock_model.predict_proba.return_value = np.array([[0.8, 0.2]])  # 20% probability of default
        mock_load.return_value = mock_model
        
        # Create test client
        client = TestClient(app)
        
        yield client

# Add any additional fixtures that should be available across test modules
