"""
Data processing pipeline for credit risk modeling.
"""
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, Dict, Optional, List
import pyarrow.parquet as pq
import pyarrow as pa

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    """
    A class to handle data loading, cleaning, and feature engineering.
    """
    
    def __init__(self, data_dir: str = 'data'):
        """
        Initialize the DataProcessor.
        
        Args:
            data_dir: Base directory containing raw and processed data
        """
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, 'raw')
        self.processed_dir = os.path.join(data_dir, 'processed')
        self.external_dir = os.path.join(data_dir, 'external')
        
        # Create directories if they don't exist
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.external_dir, exist_ok=True)
    
    def load_raw_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load raw data files from Xente dataset.
        
        Returns:
            Dictionary containing raw data DataFrames
        """
        data = {}
        
        # Define expected files and their loading parameters
        data_files = {
            'transactions.csv': {
                'dtype': {
                    'TransactionId': str,
                    'BatchId': str,
                    'AccountId': str,
                    'SubscriptionId': str,
                    'CustomerId': str,
                    'CurrencyCode': str,
                    'CountryCode': str,
                    'ProviderId': str,
                    'ProductId': str,
                    'ProductCategory': str,
                    'ChannelId': str,
                    'Amount': float,
                    'Value': float,
                    'PricingStrategy': int,
                    'FraudResult': int
                },
                'parse_dates': ['TransactionStartTime']
            },
            'test.csv': {
                'dtype': {
                    'TransactionId': str,
                    'BatchId': str,
                    'AccountId': str,
                    'SubscriptionId': str,
                    'CustomerId': str,
                    'CurrencyCode': str,
                    'CountryCode': str,
                    'ProviderId': str,
                    'ProductId': str,
                    'ProductCategory': str,
                    'ChannelId': str,
                    'Amount': float,
                    'Value': float,
                    'PricingStrategy': int
                },
                'parse_dates': ['TransactionStartTime']
            },
            'sample_submission.csv': {
                'dtype': {
                    'TransactionId': str,
                    'FraudResult': int
                }
            }
        }
        
        # Load each file with appropriate parameters
        for file, load_params in data_files.items():
            file_path = os.path.join(self.raw_dir, file)
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(
                        file_path,
                        dtype=load_params.get('dtype'),
                        parse_dates=load_params.get('parse_dates')
                    )
                    # Store with filename without extension as key
                    key = os.path.splitext(file)[0]
                    data[key] = df
                    logger.info(f"Loaded {file} with shape {df.shape}")
                except Exception as e:
                    logger.error(f"Error loading {file}: {str(e)}")
            else:
                logger.warning(f"File not found: {file}")
                
        # If no data was loaded, try to load any CSV as a fallback
        if not data:
            logger.warning("No standard files found, attempting to load any CSV files...")
            for file in os.listdir(self.raw_dir):
                if file.endswith('.csv'):
                    name = os.path.splitext(file)[0]
                    file_path = os.path.join(self.raw_dir, file)
                    logger.info(f"Loading {file_path}")
                    try:
                        df = pd.read_csv(file_path)
                        data[name] = df
                        logger.info(f"Successfully loaded {file} with shape {df.shape}")
                    except Exception as e:
                        logger.error(f"Error loading {file}: {str(e)}")
        
        return data
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the raw Xente dataset.
        
        Args:
            df: Raw DataFrame from Xente dataset
            
        Returns:
            Cleaned DataFrame
        """
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        logger.info(f"Initial data shape: {df.shape}")
        
        # 1. Convert TransactionStartTime to datetime
        if 'TransactionStartTime' in df.columns:
            df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], utc=True)
            
            # Sort by transaction time for each customer
            if 'CustomerId' in df.columns:
                df = df.sort_values(['CustomerId', 'TransactionStartTime'])
        
        # 2. Handle missing values
        # For numeric columns, fill with median or 0
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                if col in ['Amount', 'Value']:
                    df[col].fillna(0, inplace=True)
                else:
                    df[col].fillna(df[col].median(), inplace=True)
        
        # For categorical columns, fill with 'unknown' or appropriate default
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            if df[col].isnull().sum() > 0:
                if col in ['CurrencyCode', 'CountryCode']:
                    # Use the most common value for these critical categoricals
                    df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'unknown', inplace=True)
                else:
                    df[col].fillna('unknown', inplace=True)
        
        # 3. Clean up string columns (remove any leading/trailing spaces)
        for col in cat_cols:
            df[col] = df[col].astype(str).str.strip()
        
        # 4. Convert boolean columns if they exist
        bool_cols = ['FraudResult'] if 'FraudResult' in df.columns else []
        for col in bool_cols:
            df[col] = df[col].astype(bool)
        
        # 5. Ensure proper data types
        if 'Amount' in df.columns:
            df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
        if 'Value' in df.columns:
            df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
        
        # 6. Add any derived columns that might be useful
        if 'TransactionStartTime' in df.columns:
            df['transaction_date'] = df['TransactionStartTime'].dt.date
            df['transaction_hour'] = df['TransactionStartTime'].dt.hour
            df['transaction_day_of_week'] = df['TransactionStartTime'].dt.dayofweek
            df['transaction_day_of_month'] = df['TransactionStartTime'].dt.day
            df['transaction_month'] = df['TransactionStartTime'].dt.month
            df['transaction_year'] = df['TransactionStartTime'].dt.year
        
        logger.info(f"Cleaned data shape: {df.shape}")
        logger.info(f"Columns after cleaning: {df.columns.tolist()}")
        
        return df
    
    def _create_features_internal(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Internal method to create features from the cleaned data using enhanced feature engineering.
        
        Args:
            df: Cleaned transaction data
            
        Returns:
            DataFrame with additional features
        """
        from src.feature_engineering import XenteFeatures, TimeFeatures
        
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Initialize feature extractors
        xente_features = XenteFeatures(
            customer_id='CustomerId',
            date_col='TransactionStartTime',
            amount_col='Amount',
            fraud_col='FraudResult' if 'FraudResult' in df.columns else None
        )
        
        time_features = TimeFeatures(date_col='TransactionStartTime')
        
        # Fit and transform features
        logger.info("Fitting and transforming features...")
        xente_features.fit(df)
        df_xente = xente_features.transform(df)
        
        time_features.fit(df)
        df_time = time_features.transform(df)
        
        # Combine all features
        df_features = pd.concat([df, df_xente, df_time], axis=1)
        
        # Drop duplicate columns that might have been created
        df_features = df_features.loc[:, ~df_features.columns.duplicated()]
        
        # Add any additional features
        if 'Amount' in df_features.columns:
            # Add transaction direction (credit/debit)
            df_features['is_credit'] = (df_features['Amount'] > 0).astype(int)
            df_features['is_debit'] = (df_features['Amount'] < 0).astype(int)
            
            # Add amount-based features
            df_features['amount_abs'] = df_features['Amount'].abs()
            df_features['amount_log'] = np.log1p(df_features['amount_abs'])
            
            # Add time-based features if not already added
            if 'TransactionStartTime' in df_features.columns:
                df_features['transaction_date'] = df_features['TransactionStartTime'].dt.date
                df_features['transaction_day_of_week'] = df_features['TransactionStartTime'].dt.dayofweek
                df_features['is_weekend'] = df_features['transaction_day_of_week'].isin([5, 6]).astype(int)
        
        logger.info(f"Created {len(df_features.columns) - len(df.columns)} new features")
        return df_features
    
    def create_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create the target variable (is_high_risk) using enhanced RFM analysis.
        
        Args:
            df: DataFrame with transaction data
            
        Returns:
            DataFrame with target variable
        """
        from src.target_engineering import create_proxy_target
        
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Create proxy target using RFM analysis
        if 'CustomerId' in df.columns and 'TransactionStartTime' in df.columns and 'Amount' in df.columns:
            logger.info("Creating proxy target variable using RFM analysis...")
            try:
                # Use our enhanced target engineering
                df_with_target, _ = create_proxy_target(
                    df=df,
                    customer_id='CustomerId',
                    date_col='TransactionStartTime',
                    amount_col='Amount',
                    n_clusters=4,  # Increased clusters for better segmentation
                    risk_threshold=0.2  # Top 20% most risky
                )
                
                # Ensure we have the target column
                if 'is_high_risk' in df_with_target.columns:
                    logger.info(f"Target variable created. High risk ratio: {df_with_target['is_high_risk'].mean():.2%}")
                    return df_with_target
                else:
                    logger.warning("Target column not found in the returned DataFrame")
            except Exception as e:
                logger.error(f"Error creating target variable: {str(e)}")
                # Fall back to simple RFM if advanced method fails
                if 'rfm_score' in df.columns:
                    logger.info("Falling back to simple RFM scoring")
                    threshold = df['rfm_score'].quantile(0.2)
                    df['is_high_risk'] = (df['rfm_score'] <= threshold).astype(int)
                    return df
        
        # If we get here, we couldn't create the target variable
        logger.warning("Could not create target variable - missing required columns")
        return df
    
    def process_data(self, input_file: Optional[str] = None) -> Tuple[pd.DataFrame, Dict]:
        """
        End-to-end data processing pipeline.
        
        Args:
            input_file: Optional path to input file. If None, loads all files from raw_dir.
            
        Returns:
            Tuple of (processed DataFrame, metadata)
        """
        logger.info("Starting data processing pipeline...")
        
        # Load raw data
        if input_file and os.path.exists(input_file):
            # Load single file if specified
            logger.info(f"Loading data from {input_file}")
            df = pd.read_csv(input_file)
            data = {'transactions': df}
        else:
            # Load all data from raw_dir
            logger.info("Loading all data from raw directory")
            data = self.load_raw_data()
            if not data:
                raise ValueError("No data files found in raw directory")
            df = data.get('transactions')
            if df is None:
                raise ValueError("No transactions data found")
        
        # Clean data
        logger.info("Cleaning data...")
        df_cleaned = self.clean_data(df)
        
        # Create features and target variable
        logger.info("Creating features and target variable...")
        df_features = self._create_features_internal(df_cleaned)
        
        # Always try to create target variable if we have the required columns
        required_cols = ['CustomerId', 'TransactionStartTime', 'Amount']
        if all(col in df_features.columns for col in required_cols):
            logger.info("Creating target variable...")
            df_with_target = self.create_target_variable(df_features)
            target_col = 'is_high_risk' if 'is_high_risk' in df_with_target.columns else None
        else:
            logger.warning(f"Missing required columns for target creation. Need: {required_cols}")
            df_with_target = df_features
            target_col = None
        
        # Prepare metadata
        metadata = {
            'num_samples': len(df_with_target),
            'features': [col for col in df_with_target.columns if col != target_col],
            'target': target_col,
            'num_features': len(df_with_target.columns) - (1 if target_col else 0),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info("Data processing pipeline completed successfully")
        return df_with_target, metadata
        
    def save_processed_data(
        self, 
        df: pd.DataFrame, 
        output_file: Optional[str] = None,
        format: str = 'parquet'
    ) -> str:
        """
        Save processed data to disk.
        
        Args:
            df: Processed DataFrame
            output_file: Output file path. If None, generates a default path.
            format: Output format ('parquet' or 'csv')
            
        Returns:
            Path to the saved file
        """
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = os.path.join(
                self.processed_dir, 
                f'processed_data_{timestamp}.{format}'
            )
        
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save in the specified format
        if format.lower() == 'parquet':
            df.to_parquet(output_file, index=False)
        elif format.lower() == 'csv':
            df.to_csv(output_file, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Processed data saved to {output_file}")
        return output_file
        
    def create_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create features and target from the cleaned data.
        
        Args:
            df: Cleaned transaction data
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        # Create features using the internal method
        df_features = self._create_features_internal(df)
        
        # If we have the target column, separate it
        if 'is_high_risk' in df_features.columns:
            X = df_features.drop('is_high_risk', axis=1)
            y = df_features['is_high_risk']
            return X, y
        else:
            # If no target column, return features with None for target
            return df_features, None


def run_data_processing_pipeline(
    input_file: Optional[str] = None,
    output_file: Optional[str] = None,
    data_dir: str = 'data'
) -> Tuple[str, Dict]:
    """
    Run the complete data processing pipeline.
    
    Args:
        input_file: Optional path to input file. If None, loads from raw_dir.
        output_file: Optional path to save processed data. If None, generates a default path.
        data_dir: Base directory containing raw and processed data
        
    Returns:
        Tuple of (output_file_path, metadata)
    """
    # Initialize data processor
    processor = DataProcessor(data_dir=data_dir)
    
    # Process data
    df, metadata = processor.process_data(input_file)
    
    # Save processed data
    output_path = processor.save_processed_data(df, output_file)
    
    return output_path, metadata


if __name__ == "__main__":
    # Example usage
    output_path, metadata = run_data_processing_pipeline()
    print(f"Processing complete. Output saved to: {output_path}")
    print(f"Metadata: {metadata}")
