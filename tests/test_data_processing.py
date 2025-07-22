"""
Tests for the data processing module.
"""
import os
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Add the src directory to the Python path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_processing import DataProcessor, run_data_processing_pipeline

# Sample test data
@pytest.fixture
def sample_transaction_data():
    """Create sample transaction data for testing."""
    data = {
        'TransactionId': [f'TXN{i:04d}' for i in range(1, 6)],
        'CustomerId': ['CUST001', 'CUST001', 'CUST002', 'CUST002', 'CUST003'],
        'Amount': [100.0, -50.0, 200.0, 150.0, 75.0],
        'TransactionStartTime': [
            '2023-01-01 10:00:00',
            '2023-01-02 11:30:00',
            '2023-01-01 09:15:00',
            '2023-01-03 14:45:00',
            '2023-01-02 16:20:00'
        ],
        'ProductCategory': ['Electronics', 'Electronics', 'Books', 'Books', 'Clothing'],
        'ChannelId': ['Web', 'Mobile', 'Web', 'Mobile', 'Web'],
        'ProviderId': ['P001', 'P001', 'P002', 'P002', 'P003'],
        'CurrencyCode': ['USD', 'USD', 'EUR', 'EUR', 'USD'],
        'CountryCode': ['US', 'US', 'UK', 'UK', 'US']
    }
    return pd.DataFrame(data)

# Test DataProcessor class
def test_data_processor_initialization(tmp_path):
    """Test DataProcessor initialization."""
    # Create a temporary directory for testing
    data_dir = tmp_path / "test_data"
    
    # Initialize DataProcessor
    processor = DataProcessor(data_dir=str(data_dir))
    
    # Check if directories are created
    assert os.path.exists(processor.raw_dir)
    assert os.path.exists(processor.processed_dir)
    assert os.path.exists(processor.external_dir)

def test_clean_data(sample_transaction_data):
    """Test data cleaning functionality."""
    # Add some missing values to test
    test_data = sample_transaction_data.copy()
    test_data.loc[0, 'ProductCategory'] = None
    test_data.loc[1, 'Amount'] = None
    
    # Initialize DataProcessor
    processor = DataProcessor()
    
    # Clean the data
    cleaned_data = processor.clean_data(test_data)
    
    # Check if missing values are handled
    assert cleaned_data['ProductCategory'].isna().sum() == 0
    assert cleaned_data['Amount'].isna().sum() == 0
    
    # Check if date column is converted to datetime
    assert pd.api.types.is_datetime64_any_dtype(cleaned_data['TransactionStartTime'])

def test_create_features(sample_transaction_data):
    """Test feature engineering functionality."""
    # Initialize DataProcessor
    processor = DataProcessor()
    
    # Clean the data first
    cleaned_data = processor.clean_data(sample_transaction_data)
    
    # Create features
    features_data = processor.create_features(cleaned_data)
    
    # Check if new features are created
    expected_columns = [
        'amount_abs', 'amount_log', 'transaction_hour', 'transaction_day',
        'transaction_month', 'transaction_year', 'transaction_dayofweek',
        'transaction_is_weekend', 'recency', 'frequency', 'monetary', 'rfm_score'
    ]
    
    for col in expected_columns:
        assert col in features_data.columns, f"Missing expected column: {col}"
    
    # Check if RFM features are calculated correctly
    customer_data = features_data[features_data['CustomerId'] == 'CUST001']
    assert 'frequency' in customer_data.columns
    assert customer_data['frequency'].iloc[0] == 2  # CUST001 has 2 transactions

def test_create_target_variable(sample_transaction_data):
    """Test target variable creation."""
    # Initialize DataProcessor
    processor = DataProcessor()
    
    # Clean and create features
    cleaned_data = processor.clean_data(sample_transaction_data)
    features_data = processor.create_features(cleaned_data)
    
    # Create target variable
    target_data = processor.create_target_variable(features_data)
    
    # Check if target variable is created
    assert 'is_high_risk' in target_data.columns
    assert target_data['is_high_risk'].isin([0, 1]).all()
    
    # Check if the target is created based on RFM score
    # Customer with highest RFM score should not be high risk
    max_rfm_customer = target_data.loc[target_data['rfm_score'].idxmax()]
    assert max_rfm_customer['is_high_risk'] == 0

def test_save_processed_data(sample_transaction_data, tmp_path):
    """Test saving processed data."""
    # Initialize DataProcessor with temporary directory
    processor = DataProcessor(data_dir=str(tmp_path))
    
    # Process the data
    cleaned_data = processor.clean_data(sample_transaction_data)
    features_data = processor.create_features(cleaned_data)
    target_data = processor.create_target_variable(features_data)
    
    # Save the data
    output_file = tmp_path / "processed" / "test_output.parquet"
    saved_path = processor.save_processed_data(target_data, output_file=str(output_file))
    
    # Check if file is saved
    assert os.path.exists(saved_path)
    
    # Check if the saved data can be loaded back
    loaded_data = pd.read_parquet(saved_path)
    assert not loaded_data.empty
    assert len(loaded_data) == len(target_data)

# Test the complete pipeline
@patch('src.data_processing.DataProcessor')
def test_run_data_processing_pipeline(mock_processor_class, tmp_path):
    """Test the complete data processing pipeline."""
    # Setup mock
    mock_processor = MagicMock()
    mock_processor_class.return_value = mock_processor
    
    # Mock the return values
    mock_df = pd.DataFrame({'test': [1, 2, 3]})
    mock_processor.process_data.return_value = (mock_df, {'test_metadata': 'value'})
    
    # Define test file path
    test_input = tmp_path / "input.csv"
    test_output = tmp_path / "output.parquet"
    
    # Run the pipeline
    output_path, metadata = run_data_processing_pipeline(
        input_file=str(test_input),
        output_file=str(test_output),
        data_dir=str(tmp_path)
    )
    
    # Check if the processor was called correctly
    mock_processor_class.assert_called_once_with(data_dir=str(tmp_path))
    mock_processor.process_data.assert_called_once_with(input_file=str(test_input))
    mock_processor.save_processed_data.assert_called_once()
    
    # Check the output
    assert output_path == str(test_output)
    assert 'test_metadata' in metadata

def test_handle_missing_values():
    """Test handling of missing values."""
    # Create test data with missing values
    data = {
        'numeric': [1, 2, None, 4, 5],
        'categorical': ['A', None, 'B', 'B', 'A'],
        'date': ['2023-01-01', '2023-01-02', None, '2023-01-04', '2023-01-05']
    }
    df = pd.DataFrame(data)
    
    # Initialize DataProcessor
    processor = DataProcessor()
    
    # Clean the data
    cleaned_data = processor.clean_data(df)
    
    # Check if missing values are handled
    assert cleaned_data['numeric'].isna().sum() == 0
    assert cleaned_data['categorical'].isna().sum() == 0
    assert cleaned_data['date'].isna().sum() == 0
    
    # Check if numeric missing values are filled with median
    assert cleaned_data['numeric'].iloc[2] == df['numeric'].median()
    
    # Check if categorical missing values are filled with 'missing'
    assert (cleaned_data['categorical'] == 'missing').sum() == 1

if __name__ == "__main__":
    pytest.main(["-v", "--cov=src", "--cov-report=term-missing"])