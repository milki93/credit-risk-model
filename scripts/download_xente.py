import os
import pandas as pd
import numpy as np
from pathlib import Path

def download_xente_dataset(data_dir: str = 'data/raw') -> dict:
    """
    Download the Xente dataset from Kaggle or use a local copy if available.
    
    Args:
        data_dir: Directory to save the dataset
        
    Returns:
        Dictionary with file paths for each dataset file
    """
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Define file paths
    file_paths = {
        'transactions': os.path.join(data_dir, 'transactions.csv'),
        'train': os.path.join(data_dir, 'train.csv'),
        'test': os.path.join(data_dir, 'test.csv'),
        'sample_submission': os.path.join(data_dir, 'sample_submission.csv')
    }
    
    # Check if files already exist
    all_files_exist = all(os.path.exists(path) for path in file_paths.values())
    
    if all_files_exist:
        print("Using existing Xente dataset files")
        return file_paths
    
    print("Xente dataset not found. Please download it from:")
    print("https://www.kaggle.com/competitions/xente-fraud-detection/data")
    print("\nAfter downloading, place the following files in the 'data/raw' directory:")
    print("- transactions.csv")
    print("- train.csv")
    print("- test.csv")
    print("- sample_submission.csv")
    
    # Create empty DataFrames with expected structure
    if not os.path.exists(file_paths['transactions']):
        pd.DataFrame({
            'TransactionId': [],
            'TransactionStartTime': [],
            'AccountId': [],
            'BatchId': [],
            'SubscriptionId': [],
            'CustomerId': [],
            'ProviderId': [],
            'ProductId': [],
            'ProductCategory': [],
            'ChannelId': [],
            'Amount': [],
            'Value': [],
            'TransactionAmount': [],
            'TransactionAmountUSD': [],
            'TransactionAmountLocal': [],
            'CurrencyCode': [],
            'CountryCode': [],
            'MerchantId': [],
            'PricingStrategy': []
        }).to_csv(file_paths['transactions'], index=False)
    
    if not os.path.exists(file_paths['train']):
        pd.DataFrame({
            'TransactionId': [],
            'FraudResult': []
        }).to_csv(file_paths['train'], index=False)
    
    if not os.path.exists(file_paths['test']):
        pd.DataFrame({
            'TransactionId': []
        }).to_csv(file_paths['test'], index=False)
    
    if not os.path.exists(file_paths['sample_submission']):
        pd.DataFrame({
            'TransactionId': [],
            'FraudResult': []
        }).to_csv(file_paths['sample_submission'], index=False)
    
    print("\nCreated empty template files. Please replace them with the actual dataset files.")
    return file_paths

if __name__ == "__main__":
    # Download or verify the dataset
    file_paths = download_xente_dataset()
    
    # Print dataset information
    print("\nDataset files:")
    for name, path in file_paths.items():
        print(f"- {name}: {path} (Exists: {os.path.exists(path)})")
