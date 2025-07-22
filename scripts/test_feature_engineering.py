"""
Test script for the enhanced feature engineering pipeline.
"""
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.feature_engineering import XenteFeatures, TimeFeatures
from src.data_processing import DataProcessor
from src.utils.data_utils import split_data

# Set up paths
DATA_DIR = Path(__file__).parent.parent / 'data'
RAW_DATA_PATH = DATA_DIR / 'raw' / 'transactions.csv'
OUTPUT_DIR = DATA_DIR / 'processed'
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

def test_feature_engineering():
    """Test the feature engineering pipeline with sample data."""
    print("Testing feature engineering pipeline...")
    
    # Load a sample of the data
    print(f"Loading data from {RAW_DATA_PATH}...")
    df = pd.read_csv(RAW_DATA_PATH, nrows=10000)  # Test with first 10k rows
    
    # Convert date column to datetime
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    
    # Initialize feature extractors
    print("Initializing feature extractors...")
    xente_features = XenteFeatures(
        customer_id='CustomerId',
        date_col='TransactionStartTime',
        amount_col='Amount',
        fraud_col='FraudResult'
    )
    
    time_features = TimeFeatures(date_col='TransactionStartTime')
    
    # Fit the feature extractors
    print("Fitting feature extractors...")
    xente_features.fit(df)
    time_features.fit(df)
    
    # Transform the data
    print("Transforming data...")
    xente_transformed = xente_features.transform(df)
    time_transformed = time_features.transform(df)
    
    # Combine the features
    print("Combining features...")
    all_features = pd.concat([
        df[['CustomerId', 'TransactionStartTime', 'Amount', 'FraudResult']],
        xente_transformed,
        time_transformed
    ], axis=1)
    
    # Display feature information
    print("\n=== Feature Engineering Results ===")
    print(f"Original columns: {df.shape[1]}")
    print(f"Engineered features: {xente_transformed.shape[1] + time_transformed.shape[1]}")
    print(f"Total features after engineering: {all_features.shape[1]}")
    
    # Display sample of engineered features
    print("\n=== Sample of Engineered Features ===")
    print(all_features.head())
    
    # Save the processed data
    output_path = OUTPUT_DIR / 'test_engineered_features.parquet'
    all_features.to_parquet(output_path)
    print(f"\nSaved test features to {output_path}")
    
    return all_features

if __name__ == "__main__":
    test_feature_engineering()
