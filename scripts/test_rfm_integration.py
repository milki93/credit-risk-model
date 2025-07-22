"""
Test script to verify RFM target engineering integration with the feature engineering pipeline.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.feature_engineering import preprocess_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_sample_data(n_samples=10000):
    """Generate sample transaction data for testing."""
    np.random.seed(42)
    
    # Generate customer IDs (fewer unique customers to ensure some have multiple transactions)
    n_customers = max(100, n_samples // 100)  # At least 100 customers
    customer_ids = [f'CUST_{i:05d}' for i in range(1, n_customers + 1)]
    
    # Generate transaction dates (last 180 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    
    # Generate transactions
    data = []
    for i in range(n_samples):
        customer_id = np.random.choice(customer_ids)
        days_ago = np.random.uniform(0, 180)
        transaction_date = (end_date - timedelta(days=days_ago)).strftime('%Y-%m-%d %H:%M:%S')
        amount = np.random.lognormal(mean=3, sigma=1.5)  # Right-skewed distribution
        
        data.append({
            'TransactionId': f'TXN_{i:08d}',
            'CustomerId': customer_id,
            'TransactionStartTime': transaction_date,
            'Amount': amount,
            'ProductCategory': np.random.choice(['Electronics', 'Clothing', 'Groceries', 'Entertainment', 'Other']),
            'ProviderId': np.random.choice(['VISA', 'MASTERCARD', 'AMEX', 'DISCOVER']),
            'CountryCode': np.random.choice(['US', 'UK', 'CA', 'AU', 'DE', 'FR', 'JP', 'CN'])
        })
    
    return pd.DataFrame(data)

def main():
    # Generate sample data
    logger.info("Generating sample transaction data...")
    df = generate_sample_data(10000)
    
    # Define column types
    numerical_cols = ['Amount']
    categorical_cols = ['ProductCategory', 'ProviderId', 'CountryCode']
    
    # Process data with RFM target
    logger.info("Running feature engineering pipeline with RFM target...")
    X_train, X_test, y_train, y_test, feature_names = preprocess_data(
        df=df,
        numerical_cols=numerical_cols,
        categorical_cols=categorical_cols,
        target_col=None,  # Let the function create RFM target
        customer_id_col='CustomerId',
        date_col='TransactionStartTime',
        amount_col='Amount',
        merchant_col='ProviderId',
        test_size=0.2,
        random_state=42,
        drop_original=True,
        create_rfm_target=True,
        rfm_target_col='is_high_risk'
    )
    
    # Display results
    logger.info("\n=== Results ===")
    logger.info(f"Training set shape: {X_train.shape}")
    logger.info(f"Test set shape: {X_test.shape}")
    logger.info(f"Number of features: {len(feature_names)}")
    
    # Check target distribution
    train_high_risk = y_train.mean()
    test_high_risk = y_test.mean()
    
    logger.info("\n=== Target Distribution ===")
    logger.info(f"Training set - High risk: {train_high_risk:.2%}")
    logger.info(f"Test set - High risk: {test_high_risk:.2%}")
    
    # Check if the split is customer-based (no data leakage)
    # Get customer IDs from the original DataFrame using the indices from the split
    train_indices = X_train.index
    test_indices = X_test.index
    
    train_customers = df.loc[train_indices, 'CustomerId'].nunique()
    test_customers = df.loc[test_indices, 'CustomerId'].nunique()
    
    train_customer_set = set(df.loc[train_indices, 'CustomerId'].unique())
    test_customer_set = set(df.loc[test_indices, 'CustomerId'].unique())
    overlap = len(train_customer_set & test_customer_set)
    
    logger.info("\n=== Customer Split ===")
    logger.info(f"Unique customers in train: {train_customers}")
    logger.info(f"Unique customers in test: {test_customers}")
    logger.info(f"Customers in both sets: {overlap} (should be 0 for proper split)")
    
    # Save results for inspection
    output_dir = 'data/processed/rfm_integration'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save processed data
    train_data = X_train.copy()
    train_data['is_high_risk'] = y_train
    train_data.to_csv(f"{output_dir}/train_data.csv", index=False)
    
    test_data = X_test.copy()
    test_data['is_high_risk'] = y_test
    test_data.to_csv(f"{output_dir}/test_data.csv", index=False)
    
    # Save feature names
    with open(f"{output_dir}/feature_names.txt", 'w') as f:
        f.write('\n'.join(feature_names))
    
    logger.info(f"\nResults saved to: {output_dir}")

if __name__ == "__main__":
    main()
