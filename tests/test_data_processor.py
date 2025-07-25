"""
Tests for the DataProcessor class.
"""
import os
import tempfile
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

from src.features.data_processor import DataProcessor, create_processed_dataset

# Sample transaction data for testing
SAMPLE_DATA = {
    'TransactionId': [f'T{i}' for i in range(1, 101)],
    'BatchId': [f'B{i//10}' for i in range(1, 101)],
    'AccountId': [f'A{i//5}' for i in range(1, 101)],
    'SubscriptionId': [f'S{i//10}' for i in range(1, 101)],
    'CustomerId': [f'C{i%10}' for i in range(100)],  # 10 unique customers (C0-C9)
    'CurrencyCode': ['UGX'] * 100,
    'CountryCode': ['256'] * 100,
    'ProviderId': [f'P{i%5+1}' for i in range(100)],
    'ProductId': [f'PRD{i%10+1}' for i in range(100)],
    'ProductCategory': np.random.choice(['airtime', 'data_bundles', 'utility_bill', 'financial_services'], 100),
    'ChannelId': np.random.choice(['CH1', 'CH2', 'CH3'], 100),
    'Amount': np.random.uniform(100, 10000, 100).round(2),
    'Value': np.random.uniform(100, 10000, 100).round(2),
    'TransactionStartTime': [datetime(2023, 1, 1) + timedelta(days=i) for i in range(100)],
    'PricingStrategy': np.random.choice([1, 2, 3, 4], 100),
    'FraudResult': np.random.choice([0, 1], 100, p=[0.95, 0.05])
}

@pytest.fixture
def sample_data_path():
    """Create a temporary CSV file with sample data for testing."""
    df = pd.DataFrame(SAMPLE_DATA)
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w') as f:
        df.to_csv(f.name, index=False)
        yield f.name
    # Cleanup after test
    if os.path.exists(f.name):
        os.remove(f.name)

def test_data_processor_init():
    """Test DataProcessor initialization."""
    processor = DataProcessor(snapshot_date='2023-12-31')
    assert processor.snapshot_date == pd.Timestamp('2023-12-31')
    assert processor.data is None
    assert processor.rfm_features is None
    assert processor.customer_features is None
    assert processor.processed_data is None

def test_load_data(sample_data_path):
    """Test loading data from a CSV file."""
    processor = DataProcessor(data_path=sample_data_path)
    data = processor.load_data()
    
    assert data is not None
    assert len(data) == 100
    assert 'CustomerId' in data.columns
    assert 'TransactionStartTime' in data.columns
    assert data['TransactionStartTime'].dtype == 'datetime64[ns]'

def test_clean_data(sample_data_path):
    """Test data cleaning functionality."""
    processor = DataProcessor(data_path=sample_data_path)
    processor.load_data()
    cleaned_data = processor.clean_data()
    
    assert 'DayOfWeek' in cleaned_data.columns
    assert 'HourOfDay' in cleaned_data.columns
    assert 'AmountCategory' in cleaned_data.columns
    assert cleaned_data['Value'].min() >= 0  # Should be absolute values

def test_calculate_rfm(sample_data_path):
    """Test RFM feature calculation."""
    processor = DataProcessor(data_path=sample_data_path)
    processor.load_data()
    rfm_features = processor.calculate_rfm()
    
    assert 'Recency' in rfm_features.columns
    assert 'Frequency' in rfm_features.columns
    assert 'Monetary' in rfm_features.columns
    assert 'RFM_Score' in rfm_features.columns
    assert 'RFM_Segment' in rfm_features.columns
    assert len(rfm_features) == 10  # 10 unique customers in sample data

def test_calculate_behavioral_features(sample_data_path):
    """Test behavioral feature calculation."""
    processor = DataProcessor(data_path=sample_data_path)
    # First clean the data to ensure required columns are created
    processor.load_data()
    cleaned_data = processor.clean_data()
    
    # Verify required columns are created in clean_data
    required_columns = ['AmountCategory', 'DayOfWeek', 'HourOfDay']
    for col in required_columns:
        assert col in cleaned_data.columns, f"Required column {col} not found in cleaned data"
    
    # Now calculate behavioral features
    features = processor.calculate_behavioral_features()
    
    # Verify the output features
    expected_columns = [
        'CustomerId', 'TotalTransactions', 'TotalSpend', 'AvgTransactionValue',
        'StdTransactionValue', 'MinTransactionValue', 'MaxTransactionValue',
        'UniqueTransactionDays', 'UniqueProductCategories', 'UniqueProviders',
        'UniqueChannels', 'MostActiveDay', 'MostActiveHour',
        'MostCommonAmountCategory', 'TransactionFrequency', 'CustomerLifetime',
        'AvgDaysBetweenTransactions', 'DaysSinceFirstTransactionDate',
        'DaysSinceLastTransactionDate'
    ]
    
    for col in expected_columns:
        assert col in features.columns, f"Expected column {col} not found in behavioral features"

def test_create_target_variable(sample_data_path):
    """Test target variable creation."""
    # Test RFM clustering method
    processor = DataProcessor(data_path=sample_data_path)
    processor.load_data()
    target_rfm = processor.create_target_variable(method='rfm_cluster', n_clusters=3)
    
    assert 'is_high_risk' in target_rfm.columns
    assert set(target_rfm['is_high_risk'].unique()).issubset({0, 1})
    
    # Test fraud history method
    target_fraud = processor.create_target_variable(method='fraud_history')
    assert 'is_high_risk' in target_fraud.columns
    assert set(target_fraud['is_high_risk'].unique()).issubset({0, 1})

def test_process_features(sample_data_path):
    """Test the complete feature processing pipeline."""
    processor = DataProcessor(data_path=sample_data_path)
    processed_data = processor.process_features(target_method='rfm_cluster')
    
    # Check that we have the expected columns
    assert 'is_high_risk' in processed_data.columns
    assert 'Recency' in processed_data.columns
    assert 'Frequency' in processed_data.columns
    assert 'Monetary' in processed_data.columns
    assert 'TotalTransactions' in processed_data.columns
    assert 'TotalSpend' in processed_data.columns
    
    # Check that we have the expected number of rows (one per customer)
    assert len(processed_data) == 10  # 10 unique customers in sample data

def test_save_processed_data(sample_data_path, tmp_path):
    """Test saving processed data to disk."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    processor = DataProcessor(data_path=sample_data_path)
    processor.process_features()
    output_path = processor.save_processed_data(output_dir=str(output_dir))
    
    assert os.path.exists(output_path)
    assert output_path.endswith('.parquet')
    
    # Verify the saved data can be loaded
    loaded_data = pd.read_parquet(output_path)
    assert len(loaded_data) > 0
    assert 'is_high_risk' in loaded_data.columns

def test_create_processed_dataset(sample_data_path, tmp_path):
    """Test the create_processed_dataset convenience function."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    output_path = create_processed_dataset(
        data_path=sample_data_path,
        output_dir=str(output_dir),
        snapshot_date='2023-12-31',
        target_method='rfm_cluster',
        n_clusters=3
    )
    
    assert os.path.exists(output_path)
    assert output_path.endswith('.parquet')
    
    # Verify the saved data can be loaded
    loaded_data = pd.read_parquet(output_path)
    assert len(loaded_data) > 0
    assert 'is_high_risk' in loaded_data.columns

def test_get_feature_methods(sample_data_path):
    """Test the get_feature_names, get_numeric_features, and get_categorical_features methods."""
    processor = DataProcessor(data_path=sample_data_path)
    processor.process_features()
    
    # Test get_feature_names
    feature_names = processor.get_feature_names()
    assert isinstance(feature_names, list)
    assert len(feature_names) > 0
    assert 'CustomerId' not in feature_names
    assert 'is_high_risk' not in feature_names
    
    # Test get_numeric_features
    numeric_features = processor.get_numeric_features()
    assert isinstance(numeric_features, list)
    assert all(f in feature_names for f in numeric_features)
    
    # Test get_categorical_features
    categorical_features = processor.get_categorical_features()
    assert isinstance(categorical_features, list)
    assert all(f in feature_names for f in categorical_features)
    
    # Check that all features are either numeric or categorical
    all_features = set(numeric_features) | set(categorical_features)
    assert set(feature_names) == all_features
