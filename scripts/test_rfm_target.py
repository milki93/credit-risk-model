"""
Test script for RFM-based target engineering.

This script demonstrates how to use the RFM analysis to create a proxy target
variable for credit risk modeling.
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

from src.target_engineering import RFMTargetEngineer, create_proxy_target

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
    
    # Set snapshot date (e.g., end of last month)
    snapshot_date = datetime.now().replace(day=1) - timedelta(days=1)
    
    # Create proxy target using the RFM analysis
    logger.info("Creating proxy target variable using RFM analysis...")
    df_with_target = create_proxy_target(
        df=df,
        customer_id_col='CustomerId',
        date_col='TransactionStartTime',
        amount_col='Amount',
        snapshot_date=snapshot_date,
        n_clusters=3,
        random_state=42
    )
    
    # Display results
    logger.info("\n=== Results ===")
    logger.info(f"Total transactions: {len(df_with_target):,}")
    logger.info(f"High-risk transactions: {df_with_target['is_high_risk'].sum():,} "
               f"({df_with_target['is_high_risk'].mean():.1%})")
    
    # Show sample of high-risk transactions
    logger.info("\nSample of high-risk transactions:")
    high_risk_sample = df_with_target[df_with_target['is_high_risk'] == 1].sample(
        min(5, df_with_target['is_high_risk'].sum()),
        random_state=42
    )
    logger.info(high_risk_sample[['TransactionId', 'CustomerId', 'TransactionStartTime', 
                                'Amount', 'is_high_risk']].to_string())
    
    # Save results for inspection
    output_dir = 'data/processed/rfm_targets'
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = f"{output_dir}/transactions_with_risk_{datetime.now().strftime('%Y%m%d')}.csv"
    df_with_target.to_csv(output_file, index=False)
    logger.info(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()
