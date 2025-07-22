import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.feature_engineering import preprocess_data, create_feature_engineering_pipeline
from scripts.download_xente import download_xente_dataset

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(data_dir: str = 'data/raw'):
    """Load the Xente dataset."""
    file_paths = download_xente_dataset(data_dir)
    
    logger.info("Loading dataset files...")
    try:
        # Load transactions data
        transactions = pd.read_csv(file_paths['transactions'])
        
        # Load training labels
        train_labels = pd.read_csv(file_paths['train'])
        
        # Merge transactions with training labels
        df = pd.merge(transactions, train_labels, on='TransactionId', how='left')
        
        # Convert date column to datetime
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
        
        # Sort by transaction time
        df = df.sort_values('TransactionStartTime')
        
        logger.info(f"Loaded dataset with {len(df)} transactions")
        return df
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

def main():
    # Load the data
    try:
        df = load_data()
    except Exception as e:
        logger.error("Failed to load data. Please ensure the dataset files are in the 'data/raw' directory.")
        logger.error("You can download the dataset from: https://www.kaggle.com/competitions/xente-fraud-detection/data")
        return
    
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
    
    # Check if we have the target column (for training data)
    has_target = target_col in df.columns
    
    logger.info("Starting feature engineering...")
    start_time = datetime.now()
    
    try:
        if has_target:
            # For training data (with target)
            X_train, X_test, y_train, y_test, feature_names = preprocess_data(
                df=df,
                numerical_cols=numerical_cols,
                categorical_cols=categorical_cols,
                target_col=target_col,
                test_size=0.2,
                drop_original=True
            )
            
            # Save processed data
            output_dir = 'data/processed'
            os.makedirs(output_dir, exist_ok=True)
            
            X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
            X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
            y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
            y_test.to_csv(f"{output_dir}/y_test.csv", index=False)
            
            logger.info(f"Processed training data shape: {X_train.shape}")
            logger.info(f"Processed test data shape: {X_test.shape}")
            logger.info(f"Number of features: {len(feature_names)}")
            logger.info(f"Feature names: {feature_names[:10]}...")
            
        else:
            # For test data (without target)
            X_processed, feature_names = preprocess_data(
                df=df,
                numerical_cols=numerical_cols,
                categorical_cols=categorical_cols,
                target_col=None,
                drop_original=True
            )
            
            # Save processed data
            output_dir = 'data/processed'
            os.makedirs(output_dir, exist_ok=True)
            X_processed.to_csv(f"{output_dir}/X_processed.csv", index=False)
            
            logger.info(f"Processed data shape: {X_processed.shape}")
            logger.info(f"Number of features: {len(feature_names)}")
            
        logger.info(f"Feature engineering completed in {datetime.now() - start_time}")
        
    except Exception as e:
        logger.error(f"Error during feature engineering: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
