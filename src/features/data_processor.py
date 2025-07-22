"""
Data processing module for credit risk modeling.
Handles loading, cleaning, and transforming raw transaction data into features.
"""
import os
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple, Union

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Class for processing raw transaction data into features for credit risk modeling.
    """
    
    def __init__(self, data_path: str = "data/raw/transactions.csv", 
                 snapshot_date: Optional[str] = None):
        """
        Initialize the DataProcessor.
        
        Args:
            data_path: Path to the raw transaction data file
            snapshot_date: Reference date for RFM calculation (YYYY-MM-DD format).
                          If None, uses the maximum date in the data.
        """
        self.data_path = data_path
        self.snapshot_date = pd.to_datetime(snapshot_date) if snapshot_date else None
        self.data = None
        self.rfm_features = None
        self.customer_features = None
        self.processed_data = None
        
        # Initialize transformers
        self._init_transformers()
    
    def _init_transformers(self):
        """Initialize data transformers."""
        # Numerical features pipeline
        self.numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Categorical features pipeline
        self.categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
    
    def load_data(self) -> pd.DataFrame:
        """
        Load and preprocess the raw transaction data.
        
        Returns:
            pd.DataFrame: Processed transaction data
        """
        logger.info(f"Loading data from {self.data_path}")
        
        # Load data with appropriate dtypes
        dtypes = {
            'TransactionId': 'str',
            'BatchId': 'str',
            'AccountId': 'str',
            'SubscriptionId': 'str',
            'CustomerId': 'str',
            'CurrencyCode': 'category',
            'CountryCode': 'category',
            'ProviderId': 'category',
            'ProductId': 'category',
            'ProductCategory': 'category',
            'ChannelId': 'category',
            'Amount': 'float32',
            'Value': 'float32',
            'PricingStrategy': 'category',
            'FraudResult': 'int8'
        }
        
        # Parse dates
        date_columns = ['TransactionStartTime']
        
        try:
            self.data = pd.read_csv(
                self.data_path,
                dtype=dtypes,
                parse_dates=date_columns,
                infer_datetime_format=True
            )
            
            # Set snapshot date to max date + 1 day if not provided
            if self.snapshot_date is None:
                self.snapshot_date = self.data['TransactionStartTime'].max() + timedelta(days=1)
                logger.info(f"Using snapshot date: {self.snapshot_date}")
            
            logger.info(f"Loaded {len(self.data)} transactions")
            return self.data
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def clean_data(self) -> pd.DataFrame:
        """
        Clean the loaded transaction data.
        
        Returns:
            pd.DataFrame: Cleaned transaction data
        """
        if self.data is None:
            self.load_data()
            
        logger.info("Cleaning transaction data...")
        
        # Make a copy to avoid modifying the original data
        df = self.data.copy()
        
        # Handle missing values
        df['Amount'] = df['Amount'].fillna(0)
        df['Value'] = df['Value'].fillna(0)
        
        # Ensure Value is always positive (absolute of Amount)
        df['Value'] = df['Value'].abs()
        
        # Add transaction month and year
        df['TransactionMonth'] = df['TransactionStartTime'].dt.to_period('M')
        df['TransactionYear'] = df['TransactionStartTime'].dt.year
        
        # Add day of week and hour of day
        df['DayOfWeek'] = df['TransactionStartTime'].dt.dayofweek
        df['HourOfDay'] = df['TransactionStartTime'].dt.hour
        
        # Add transaction amount categories
        df['AmountCategory'] = pd.cut(
            df['Value'],
            bins=[0, 100, 500, 1000, 5000, float('inf')],
            labels=['0-100', '100-500', '500-1000', '1000-5000', '5000+'],
            right=False
        )
        
        self.data = df
        return df
    
    def calculate_rfm(self) -> pd.DataFrame:
        """
        Calculate RFM (Recency, Frequency, Monetary) features.
        
        Returns:
            pd.DataFrame: RFM features for each customer
        """
        if self.data is None:
            self.clean_data()
            
        logger.info("Calculating RFM features...")
        
        # Calculate RFM metrics
        rfm = self.data.groupby('CustomerId').agg({
            'TransactionStartTime': lambda x: (self.snapshot_date - x.max()).days,  # Recency
            'TransactionId': 'count',  # Frequency
            'Value': ['sum', 'mean', 'max']  # Monetary
        })
        
        # Flatten multi-index columns
        rfm.columns = ['_'.join(col).strip() for col in rfm.columns.values]
        rfm = rfm.rename(columns={
            'TransactionStartTime_<lambda>': 'Recency',
            'TransactionId_count': 'Frequency',
            'Value_sum': 'Monetary',
            'Value_mean': 'AvgTransactionValue',
            'Value_max': 'MaxTransactionValue'
        })
        
        # Calculate additional RFM metrics
        rfm['MonetaryPerFrequency'] = rfm['Monetary'] / rfm['Frequency']
        rfm['RecencyScore'] = pd.qcut(rfm['Recency'], q=5, labels=[5, 4, 3, 2, 1]).astype(int)
        rfm['FrequencyScore'] = pd.qcut(rfm['Frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
        rfm['MonetaryScore'] = pd.qcut(rfm['Monetary'], q=5, labels=[1, 2, 3, 4, 5]).astype(int)
        
        # Calculate RFM Score
        rfm['RFM_Score'] = rfm['RecencyScore'] + rfm['FrequencyScore'] + rfm['MonetaryScore']
        
        # Segment customers based on RFM score
        rfm['RFM_Segment'] = pd.cut(
            rfm['RFM_Score'],
            bins=[0, 4, 8, 12, 15],
            labels=['Low', 'Medium', 'High', 'Top'],
            include_lowest=True
        )
        
        self.rfm_features = rfm.reset_index()
        return self.rfm_features
    
    def calculate_behavioral_features(self) -> pd.DataFrame:
        """
        Calculate behavioral features from transaction data.
        
        Returns:
            pd.DataFrame: Behavioral features for each customer
        """
        if self.data is None:
            self.clean_data()
            
        logger.info("Calculating behavioral features...")
        
        # Group by customer and calculate features
        customer_features = self.data.groupby('CustomerId').agg({
            'TransactionId': 'count',  # Total transactions
            'Value': ['sum', 'mean', 'std', 'min', 'max'],  # Transaction values
            'TransactionStartTime': ['min', 'max', 'nunique'],  # Transaction dates
            'ProductCategory': 'nunique',  # Product diversity
            'ProviderId': 'nunique',  # Provider diversity
            'ChannelId': 'nunique',  # Channel diversity
            'DayOfWeek': lambda x: x.mode()[0] if not x.empty else -1,  # Most active day
            'HourOfDay': lambda x: x.mode()[0] if not x.empty else -1,  # Most active hour
            'AmountCategory': lambda x: x.mode()[0] if not x.empty else 'missing'  # Most common amount category
        })
        
        # Flatten multi-index columns
        customer_features.columns = ['_'.join(col).strip() for col in customer_features.columns.values]
        
        # Rename columns for clarity
        customer_features = customer_features.rename(columns={
            'TransactionId_count': 'TotalTransactions',
            'Value_sum': 'TotalSpend',
            'Value_mean': 'AvgTransactionValue',
            'Value_std': 'StdTransactionValue',
            'Value_min': 'MinTransactionValue',
            'Value_max': 'MaxTransactionValue',
            'TransactionStartTime_min': 'FirstTransactionDate',
            'TransactionStartTime_max': 'LastTransactionDate',
            'TransactionStartTime_nunique': 'UniqueTransactionDays',
            'ProductCategory_nunique': 'UniqueProductCategories',
            'ProviderId_nunique': 'UniqueProviders',
            'ChannelId_nunique': 'UniqueChannels',
            'DayOfWeek_<lambda>': 'MostActiveDay',
            'HourOfDay_<lambda>': 'MostActiveHour',
            'AmountCategory_<lambda>': 'MostCommonAmountCategory'
        })
        
        # Calculate additional features
        customer_features['TransactionFrequency'] = customer_features['TotalTransactions'] / customer_features['UniqueTransactionDays']
        customer_features['CustomerLifetime'] = (customer_features['LastTransactionDate'] - customer_features['FirstTransactionDate']).dt.days
        customer_features['AvgDaysBetweenTransactions'] = customer_features['CustomerLifetime'] / customer_features['TotalTransactions']
        
        # Handle potential division by zero
        customer_features['TransactionFrequency'] = customer_features['TransactionFrequency'].replace([np.inf, -np.inf], 0)
        customer_features['AvgDaysBetweenTransactions'] = customer_features['AvgDaysBetweenTransactions'].replace([np.inf, -np.inf], 0)
        
        # Convert date columns to days since snapshot
        for col in ['FirstTransactionDate', 'LastTransactionDate']:
            customer_features[f'DaysSince{col}'] = (self.snapshot_date - customer_features[col]).dt.days
        
        # Drop original date columns
        customer_features = customer_features.drop(['FirstTransactionDate', 'LastTransactionDate'], axis=1)
        
        self.customer_features = customer_features.reset_index()
        return self.customer_features
    
    def create_target_variable(self, method: str = 'rfm_cluster', n_clusters: int = 3) -> pd.DataFrame:
        """
        Create target variable for credit risk modeling.
        
        Args:
            method: Method to create target variable ('rfm_cluster' or 'fraud_history')
            n_clusters: Number of clusters to create (only used if method='rfm_cluster')
            
        Returns:
            pd.DataFrame: DataFrame with CustomerId and target variable
        """
        if method == 'rfm_cluster':
            if self.rfm_features is None:
                self.calculate_rfm()
                
            logger.info(f"Creating target variable using RFM clustering with {n_clusters} clusters")
            
            # Prepare data for clustering
            rfm_data = self.rfm_features[['Recency', 'Frequency', 'Monetary']].copy()
            
            # Log transform to handle skew
            rfm_data = np.log1p(rfm_data)
            
            # Scale the data
            scaler = StandardScaler()
            rfm_scaled = scaler.fit_transform(rfm_data)
            
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(rfm_scaled)
            
            # Assign cluster labels
            self.rfm_features['Cluster'] = clusters
            
            # Identify high-risk cluster (highest recency, lowest frequency and monetary)
            cluster_means = self.rfm_features.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
            cluster_means['RiskScore'] = (
                cluster_means['Recency'].rank(ascending=False) +  # Higher recency = higher risk
                cluster_means['Frequency'].rank(ascending=True) +  # Lower frequency = higher risk
                cluster_means['Monetary'].rank(ascending=True)     # Lower monetary = higher risk
            )
            
            # The cluster with highest score is the high-risk group
            high_risk_cluster = cluster_means['RiskScore'].idxmax()
            logger.info(f"Identified cluster {high_risk_cluster} as high-risk")
            
            # Create binary target variable
            target = pd.DataFrame({
                'CustomerId': self.rfm_features['CustomerId'],
                'is_high_risk': (self.rfm_features['Cluster'] == high_risk_cluster).astype(int)
            })
            
            return target
            
        elif method == 'fraud_history':
            logger.info("Creating target variable based on fraud history")
            
            # Group by customer and check if they have any fraudulent transactions
            fraud_target = self.data.groupby('CustomerId')['FraudResult'].max().reset_index()
            fraud_target = fraud_target.rename(columns={'FraudResult': 'is_high_risk'})
            
            return fraud_target
            
        else:
            raise ValueError(f"Invalid method: {method}. Choose 'rfm_cluster' or 'fraud_history'.")
    
    def process_features(self, target_method: str = 'rfm_cluster', n_clusters: int = 3) -> pd.DataFrame:
        """
        Process all features and create the final dataset.
        
        Args:
            target_method: Method to create target variable ('rfm_cluster' or 'fraud_history')
            n_clusters: Number of clusters to create (only used if method='rfm_cluster')
            
        Returns:
            pd.DataFrame: Processed dataset with features and target variable
        """
        # Ensure data is loaded and cleaned
        if self.data is None:
            self.clean_data()
        
        # Calculate RFM features if not already done
        if self.rfm_features is None:
            self.calculate_rfm()
        
        # Calculate behavioral features if not already done
        if self.customer_features is None:
            self.calculate_behavioral_features()
        
        # Create target variable
        target = self.create_target_variable(method=target_method, n_clusters=n_clusters)
        
        # Merge all features
        features = pd.merge(
            self.rfm_features,
            self.customer_features,
            on='CustomerId',
            how='left'
        )
        
        # Merge with target
        self.processed_data = pd.merge(
            features,
            target,
            on='CustomerId',
            how='left'
        )
        
        # Drop unnecessary columns
        columns_to_drop = ['Cluster', 'MostCommonAmountCategory']
        self.processed_data = self.processed_data.drop(
            columns=[col for col in columns_to_drop if col in self.processed_data.columns]
        )
        
        # Fill any remaining NaN values
        self.processed_data = self.processed_data.fillna(0)
        
        logger.info(f"Processed dataset shape: {self.processed_data.shape}")
        return self.processed_data
    
    def save_processed_data(self, output_dir: str = "data/processed") -> str:
        """
        Save the processed dataset to disk.
        
        Args:
            output_dir: Directory to save the processed data
            
        Returns:
            str: Path to the saved file
        """
        if self.processed_data is None:
            self.process_features()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"processed_data_{timestamp}.parquet")
        
        # Save to parquet
        self.processed_data.to_parquet(output_path, index=False)
        
        logger.info(f"Processed data saved to {output_path}")
        return output_path
    
    def get_feature_names(self) -> List[str]:
        """
        Get the list of feature names.
        
        Returns:
            List[str]: List of feature names
        """
        if self.processed_data is None:
            self.process_features()
            
        return [col for col in self.processed_data.columns if col not in ['CustomerId', 'is_high_risk']]
    
    def get_numeric_features(self) -> List[str]:
        """
        Get the list of numeric feature names.
        
        Returns:
            List[str]: List of numeric feature names
        """
        if self.processed_data is None:
            self.process_features()
            
        numeric_cols = self.processed_data.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns
        return [col for col in numeric_cols if col not in ['CustomerId', 'is_high_risk']]
    
    def get_categorical_features(self) -> List[str]:
        """
        Get the list of categorical feature names.
        
        Returns:
            List[str]: List of categorical feature names
        """
        if self.processed_data is None:
            self.process_features()
            
        cat_cols = self.processed_data.select_dtypes(include=['category', 'object']).columns
        return [col for col in cat_cols if col not in ['CustomerId', 'is_high_risk']]


def create_processed_dataset(data_path: str = "data/raw/transactions.csv",
                           output_dir: str = "data/processed",
                           snapshot_date: Optional[str] = None,
                           target_method: str = 'rfm_cluster',
                           n_clusters: int = 3) -> str:
    """
    Create a processed dataset from raw transaction data.
    
    Args:
        data_path: Path to the raw transaction data file
        output_dir: Directory to save the processed data
        snapshot_date: Reference date for RFM calculation (YYYY-MM-DD format).
                      If None, uses the maximum date in the data.
        target_method: Method to create target variable ('rfm_cluster' or 'fraud_history')
        n_clusters: Number of clusters to create (only used if method='rfm_cluster')
        
    Returns:
        str: Path to the saved processed data file
    """
    # Initialize data processor
    processor = DataProcessor(data_path=data_path, snapshot_date=snapshot_date)
    
    # Process the data
    processor.process_features(target_method=target_method, n_clusters=n_clusters)
    
    # Save the processed data
    output_path = processor.save_processed_data(output_dir=output_dir)
    
    return output_path


if __name__ == "__main__":
    # Example usage
    output_path = create_processed_dataset(
        data_path="data/raw/transactions.csv",
        output_dir="data/processed",
        snapshot_date=None,  # Will use max date in data
        target_method='rfm_cluster',
        n_clusters=3
    )
    print(f"Processed data saved to: {output_path}")
