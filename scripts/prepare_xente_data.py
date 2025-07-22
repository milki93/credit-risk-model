#!/usr/bin/env python3
"""
Script to prepare the Xente dataset for model training.
"""

import os
import pandas as pd
from pathlib import Path
from datetime import datetime
import shutil

def prepare_xente_data(input_dir: str = 'sampledata', 
                      output_dir: str = 'data/processed/xente'):
    """
    Prepare the Xente dataset for model training.
    
    Args:
        input_dir: Directory containing the raw Xente data files
        output_dir: Directory to save processed data
    """
    print("Preparing Xente dataset...")
    
    # Create output directories
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load the data
    data_file = Path(input_dir) / 'data.csv'
    print(f"Loading data from {data_file}...")
    df = pd.read_csv(data_file)
    
    # Display basic info
    print("\nDataset Info:")
    print(f"Number of transactions: {len(df):,}")
    print(f"Number of customers: {df['CustomerId'].nunique():,}")
    print(f"Number of providers: {df['ProviderId'].nunique():,}")
    print(f"Number of product categories: {df['ProductCategory'].nunique():,}")
    
    # Check for missing values
    print("\nMissing values per column:")
    print(df.isnull().sum())
    
    # Convert TransactionStartTime to datetime
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    
    # Sort by transaction time
    df = df.sort_values('TransactionStartTime')
    
    # Basic feature engineering
    df['TransactionHour'] = df['TransactionStartTime'].dt.hour
    df['TransactionDayOfWeek'] = df['TransactionStartTime'].dt.dayofweek
    df['TransactionDayOfMonth'] = df['TransactionStartTime'].dt.day
    df['TransactionMonth'] = df['TransactionStartTime'].dt.month
    
    # Convert categorical columns to category type
    cat_cols = ['ProviderId', 'ProductId', 'ProductCategory', 'ChannelId', 
                'PricingStrategy', 'CountryCode', 'CurrencyCode']
    for col in cat_cols:
        df[col] = df[col].astype('category')
    
    # Create train/test split (temporal split - last 20% as test)
    split_idx = int(0.8 * len(df))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    # Save processed data
    print("\nSaving processed data...")
    train_df.to_parquet(output_path / 'train.parquet', index=False)
    test_df.to_parquet(output_path / 'test.parquet', index=False)
    
    # Save sample submission format
    sample_sub = test_df[['TransactionId', 'FraudResult']].rename(
        columns={'FraudResult': 'FraudResult'}
    )
    sample_sub.to_csv(output_path / 'sample_submission.csv', index=False)
    
    # Print dataset statistics
    print("\nDataset split:")
    print(f"Training set: {len(train_df):,} transactions")
    print(f"Test set: {len(test_df):,} transactions")
    print(f"\nFraud rate - Train: {train_df['FraudResult'].mean():.4f}, "
          f"Test: {test_df['FraudResult'].mean():.4f}")
    
    # Save data dictionary
    data_dict = {
        'TransactionId': 'Unique transaction identifier',
        'BatchId': 'Batch identifier for processing',
        'AccountId': 'Customer account ID',
        'SubscriptionId': 'Subscription ID',
        'CustomerId': 'Customer identifier',
        'CurrencyCode': 'Transaction currency',
        'CountryCode': 'Country code',
        'ProviderId': 'Service provider',
        'ProductId': 'Product identifier',
        'ProductCategory': 'Product category',
        'ChannelId': 'Transaction channel (web, mobile, etc.)',
        'Amount': 'Transaction amount (positive for debits, negative for credits)',
        'Value': 'Absolute transaction value',
        'TransactionStartTime': 'Transaction timestamp',
        'PricingStrategy': 'Pricing category',
        'FraudResult': 'Target: 1 for fraud, 0 for legitimate'
    }
    
    with open(output_path / 'data_dictionary.txt', 'w') as f:
        for col, desc in data_dict.items():
            f.write(f"{col}: {desc}\n")
    
    print(f"\nData preparation complete. Files saved to {output_path}")
    return output_path

if __name__ == "__main__":
    # Prepare the data
    output_path = prepare_xente_data()
    
    # Display first few rows of the processed data
    train_df = pd.read_parquet(output_path / 'train.parquet')
    print("\nSample of processed training data:")
    print(train_df.head())
