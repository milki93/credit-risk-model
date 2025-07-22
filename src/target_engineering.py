"""
Module for creating proxy target variables for credit risk modeling.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional

class RFMTargetEngineer:
    """
    Engineer target variables using RFM (Recency, Frequency, Monetary) analysis
    and clustering to create proxy labels for credit risk.
    """
    
    def __init__(self, 
                 n_clusters: int = 3,
                 recency_weight: float = 0.4,
                 frequency_weight: float = 0.3,
                 monetary_weight: float = 0.3):
        """
        Initialize the RFM target engineer.
        
        Args:
            n_clusters: Number of clusters for K-means
            recency_weight: Weight for recency in RFM score
            frequency_weight: Weight for frequency in RFM score
            monetary_weight: Weight for monetary value in RFM score
        """
        self.n_clusters = n_clusters
        self.recency_weight = recency_weight
        self.frequency_weight = frequency_weight
        self.monetary_weight = monetary_weight
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.feature_columns = ['recency', 'frequency', 'monetary']
    
    def calculate_rfm(self, 
                     df: pd.DataFrame,
                     customer_id: str,
                     date_col: str,
                     amount_col: str,
                     reference_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        """
        Calculate RFM metrics for each customer.
        
        Args:
            df: DataFrame containing transaction data
            customer_id: Name of customer ID column
            date_col: Name of transaction date column
            amount_col: Name of transaction amount column
            reference_date: Reference date for recency calculation. 
                          If None, uses max date in data + 1 day
                          
        Returns:
            DataFrame with RFM metrics per customer
        """
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Convert date column to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col])
        
        # Set reference date if not provided
        if reference_date is None:
            reference_date = df[date_col].max() + pd.Timedelta(days=1)
        
        # Calculate RFM metrics
        rfm = df.groupby(customer_id).agg({
            date_col: lambda x: (reference_date - x.max()).days,  # Recency
            customer_id: 'count',                                # Frequency
            amount_col: 'sum'                                    # Monetary
        }).rename(columns={
            date_col: 'recency',
            customer_id: 'frequency',
            amount_col: 'monetary'
        })
        
        # Calculate RFM score (lower is better for recency, higher better for others)
        rfm['recency_score'] = pd.qcut(rfm['recency'], q=5, labels=[5, 4, 3, 2, 1], duplicates='drop')
        rfm['frequency_score'] = pd.qcut(rfm['frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
        rfm['monetary_score'] = pd.qcut(rfm['monetary'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
        
        # Calculate weighted RFM score
        rfm['rfm_score'] = (
            rfm['recency_score'].astype(float) * self.recency_weight +
            rfm['frequency_score'].astype(float) * self.frequency_weight +
            rfm['monetary_score'].astype(float) * self.monetary_weight
        )
        
        return rfm
    
    def fit(self, 
           df: pd.DataFrame,
           customer_id: str = 'CustomerId',
           date_col: str = 'TransactionStartTime',
           amount_col: str = 'Amount') -> 'RFMTargetEngineer':
        """
        Fit the RFM model and cluster customers.
        
        Args:
            df: DataFrame containing transaction data
            customer_id: Name of customer ID column
            date_col: Name of transaction date column
            amount_col: Name of transaction amount column
            
        Returns:
            Fitted RFMTargetEngineer instance
        """
        # Calculate RFM metrics
        self.rfm_ = self.calculate_rfm(df, customer_id, date_col, amount_col)
        
        # Scale the RFM features
        self.scaler.fit(self.rfm_[self.feature_columns])
        
        # Fit K-means on scaled features
        scaled_features = self.scaler.transform(self.rfm_[self.feature_columns])
        self.kmeans.fit(scaled_features)
        
        # Add cluster labels to RFM data
        self.rfm_['cluster'] = self.kmeans.labels_
        
        # Sort clusters by risk (lower RFM score = higher risk)
        cluster_means = self.rfm_.groupby('cluster')['rfm_score'].mean().sort_values()
        self.risk_rank_ = {old: new for new, old in enumerate(cluster_means.index)}
        self.rfm_['risk_rank'] = self.rfm_['cluster'].map(self.risk_rank_)
        
        return self
    
    def transform(self, 
                 df: pd.DataFrame,
                 customer_id: str = 'CustomerId',
                 date_col: str = 'TransactionStartTime',
                 amount_col: str = 'Amount') -> pd.DataFrame:
        """
        Transform the input data by adding RFM features and risk labels.
        
        Args:
            df: DataFrame containing transaction data
            customer_id: Name of customer ID column
            date_col: Name of transaction date column
            amount_col: Name of transaction amount column
            
        Returns:
            Original DataFrame with added RFM features and risk labels
        """
        # Calculate RFM metrics for the new data
        rfm_new = self.calculate_rfm(df, customer_id, date_col, amount_col)
        
        # Scale features and predict clusters
        scaled_features = self.scaler.transform(rfm_new[self.feature_columns])
        rfm_new['cluster'] = self.kmeans.predict(scaled_features)
        rfm_new['risk_rank'] = rfm_new['cluster'].map(self.risk_rank_)
        
        # Merge with original data
        result = df.merge(
            rfm_new[['risk_rank', 'rfm_score']],
            left_on=customer_id,
            right_index=True,
            how='left'
        )
        
        return result

def create_proxy_target(df: pd.DataFrame,
                       customer_id: str = 'CustomerId',
                       date_col: str = 'TransactionStartTime',
                       amount_col: str = 'Amount',
                       n_clusters: int = 3,
                       risk_threshold: float = 0.8) -> Tuple[pd.DataFrame, 'RFMTargetEngineer']:
    """
    Create a proxy target variable for credit risk using RFM analysis.
    
    Args:
        df: DataFrame containing transaction data
        customer_id: Name of customer ID column
        date_col: Name of transaction date column
        amount_col: Name of transaction amount column
        n_clusters: Number of clusters for risk segmentation
        risk_threshold: Percentile threshold for high-risk classification (0-1)
        
    Returns:
        Tuple of (DataFrame with added target column, fitted RFMTargetEngineer)
    """
    # Initialize and fit the RFM target engineer
    rfm_engineer = RFMTargetEngineer(n_clusters=n_clusters)
    rfm_engineer.fit(df, customer_id, date_col, amount_col)
    
    # Transform the data to get risk ranks
    df_with_risk = rfm_engineer.transform(df, customer_id, date_col, amount_col)
    
    # Create binary target based on risk threshold
    threshold = df_with_risk['risk_rank'].quantile(risk_threshold)
    df_with_risk['is_high_risk'] = (df_with_risk['risk_rank'] >= threshold).astype(int)
    
    # Add RFM score as a feature
    df_with_risk['rfm_score'] = rfm_engineer.rfm_.loc[
        df_with_risk[customer_id], 'rfm_score'
    ].values
    
    return df_with_risk, rfm_engineer
