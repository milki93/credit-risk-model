import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.feature_engineering import preprocess_data, create_feature_engineering_pipeline

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_sample_data(n_samples=1000):
    """Generate a sample dataset that mimics the Xente dataset structure."""
    np.random.seed(42)
    
    # Generate transaction IDs
    transaction_ids = [f'trx_{i:08d}' for i in range(n_samples)]
    
    # Generate timestamps (last 30 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    timestamps = pd.date_range(start_date, end_date, periods=n_samples)
    
    # Generate customer IDs
    n_customers = n_samples // 20  # ~50 transactions per customer on average
    customer_ids = [f'cust_{i:06d}' for i in range(n_customers)]
    
    # Generate merchant IDs
    n_merchants = n_samples // 50  # ~20 merchants
    merchant_ids = [f'merch_{i:04d}' for i in range(n_merchants)]
    
    # Generate transaction amounts (skewed distribution)
    amounts = np.random.lognormal(mean=3, sigma=1.5, size=n_samples).round(2)
    
    # Create the sample DataFrame
    df = pd.DataFrame({
        'TransactionId': transaction_ids,
        'TransactionStartTime': timestamps,
        'AccountId': np.random.choice(customer_ids, size=n_samples, replace=True),
        'CustomerId': np.random.choice(customer_ids, size=n_samples, replace=True),
        'MerchantId': np.random.choice(merchant_ids, size=n_samples, replace=True),
        'Amount': amounts,
        'PurchaseValue': amounts,  # This is the expected column name
        'Value': (amounts * np.random.uniform(0.9, 1.1, n_samples)).round(2),
        'TransactionAmount': amounts,
        'TransactionAmountUSD': (amounts * np.random.uniform(0.9, 1.1, n_samples)).round(2),
        'TransactionAmountLocal': amounts,
        'CurrencyCode': np.random.choice(['USD', 'EUR', 'GBP', 'KES', 'UGX', 'TZS', 'ZAR'], size=n_samples, p=[0.4, 0.2, 0.1, 0.1, 0.1, 0.05, 0.05]),
        'CountryCode': np.random.choice(['US', 'GB', 'KE', 'UG', 'TZ', 'ZA'], size=n_samples, p=[0.4, 0.2, 0.1, 0.1, 0.1, 0.1]),
        'ProviderId': np.random.choice([f'provider_{i}' for i in range(1, 11)], size=n_samples),
        'ChannelId': np.random.choice(['WEB', 'MOBILE', 'API', 'AGENT'], size=n_samples, p=[0.4, 0.4, 0.1, 0.1]),
        'ProductCategory': np.random.choice(['AIRTIME', 'UTILITY', 'TRANSFER', 'BILL', 'MERCHANT'], size=n_samples, p=[0.3, 0.2, 0.2, 0.2, 0.1]),
        'ProductId': np.random.choice([f'prod_{i:03d}' for i in range(1, 21)], size=n_samples),
        'PricingStrategy': np.random.choice(['FIXED', 'PERCENTAGE', 'TIERED'], size=n_samples, p=[0.6, 0.3, 0.1]),
        'BatchId': [f'batch_{i:04d}' for i in np.random.randint(1, 100, size=n_samples)],
        'SubscriptionId': [f'sub_{i:06d}' if np.random.random() > 0.7 else '' for i in range(n_samples)],
        'AccountCode': [f'acct_{i:04d}' for i in np.random.randint(1, 500, size=n_samples)],
    })
    
    # Generate some synthetic fraud (imbalanced classes)
    df['FraudResult'] = 0
    fraud_indices = np.random.choice(n_samples, size=int(n_samples * 0.02), replace=False)
    df.loc[fraud_indices, 'FraudResult'] = 1
    
    # Make fraud more likely for certain patterns
    df.loc[df['Amount'] > df['Amount'].quantile(0.99), 'FraudResult'] = 1
    df.loc[df['MerchantId'].isin(merchant_ids[:5]), 'FraudResult'] = 1
    
    logger.info(f"Generated sample dataset with {len(df)} records")
    logger.info(f"Fraud rate: {df['FraudResult'].mean():.2%}")
    
    return df

def main():
    # Generate sample data
    n_samples = 1000000  # Increased to 1,000,000 records
    logger.info(f"Generating {n_samples:,} sample records...")
    df = generate_sample_data(n_samples=n_samples)
    
    # Define column types
    numerical_cols = [
        'Amount', 'Value', 'TransactionAmount',
        'TransactionAmountUSD', 'TransactionAmountLocal'
    ]
    
    categorical_cols = [
        'CurrencyCode', 'CountryCode', 'ProviderId',
        'ChannelId', 'ProductCategory', 'ProductId',
        'BatchId', 'AccountCode', 'CustomerId',
        'SubscriptionId', 'MerchantId', 'PricingStrategy'
    ]
    
    # Target column
    target_col = 'FraudResult'
    
    logger.info("Running feature engineering on sample data...")
    start_time = datetime.now()
    
    try:
        # Run the feature engineering pipeline
        X_train, X_test, y_train, y_test, feature_names = preprocess_data(
            df=df,
            numerical_cols=numerical_cols,
            categorical_cols=categorical_cols,
            target_col=target_col,
            test_size=0.2,
            drop_original=True
        )
        
        # Print results
        logger.info("\n=== Feature Engineering Results ===")
        logger.info(f"Training set shape: {X_train.shape}")
        logger.info(f"Test set shape: {X_test.shape}")
        logger.info(f"Number of features: {len(feature_names)}")
        logger.info("\nFirst 20 features:")
        for i, name in enumerate(feature_names[:20], 1):
            logger.info(f"{i}. {name}")
        
        # Save sample output
        output_dir = 'data/processed/sample'
        os.makedirs(output_dir, exist_ok=True)
        
        X_train.head(100).to_csv(f"{output_dir}/X_train_sample.csv", index=False)
        X_test.head(100).to_csv(f"{output_dir}/X_test_sample.csv", index=False)
        y_train.head(100).to_csv(f"{output_dir}/y_train_sample.csv", index=False)
        y_test.head(100).to_csv(f"{output_dir}/y_test_sample.csv", index=False)
        
        logger.info(f"\nSample output saved to '{output_dir}/'")
        
    except Exception as e:
        logger.error(f"Error during feature engineering: {e}", exc_info=True)
        raise
    
    logger.info(f"\nFeature engineering completed in {datetime.now() - start_time}")

if __name__ == "__main__":
    main()
