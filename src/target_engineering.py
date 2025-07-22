"""
Target Engineering Module

This module implements the creation of a proxy target variable for credit risk
using RFM (Recency, Frequency, Monetary) analysis and K-Means clustering.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RFMTargetEngineer:
    """Engineer a proxy target variable using RFM analysis and clustering."""
    
    def __init__(self, snapshot_date=None, n_clusters=3, random_state=42):
        """Initialize the RFM target engineer.
        
        Args:
            snapshot_date (str or datetime, optional): Reference date for RFM calculation.
                If None, uses max date in the data. Defaults to None.
            n_clusters (int, optional): Number of clusters for K-Means. Defaults to 3.
            random_state (int, optional): Random state for reproducibility. Defaults to 42.
        """
        self.snapshot_date = pd.to_datetime(snapshot_date) if snapshot_date else None
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans = KMeans(
            n_clusters=n_clusters, 
            random_state=random_state,
            n_init=10  # Explicitly set n_init to avoid warning
        )
        self.pipeline = Pipeline([
            ('scaler', self.scaler),
            ('kmeans', self.kmeans)
        ])
        self.rfm_scores_ = None
        self.cluster_centers_ = None
    
    def calculate_rfm(self, df, customer_id_col='CustomerId', 
                     date_col='TransactionStartTime', amount_col='Amount'):
        """Calculate RFM metrics for each customer.
        
        Args:
            df (pd.DataFrame): Input transaction data
            customer_id_col (str): Name of customer ID column
            date_col (str): Name of transaction date column
            amount_col (str): Name of transaction amount column
            
        Returns:
            pd.DataFrame: RFM scores for each customer
        """
        logger.info("Calculating RFM metrics...")
        
        # Convert to datetime if not already
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Set snapshot date if not provided
        if self.snapshot_date is None:
            self.snapshot_date = df[date_col].max()
        
        # Calculate RFM metrics
        rfm = df.groupby(customer_id_col).agg({
            date_col: lambda x: (self.snapshot_date - x.max()).days,  # Recency
            customer_id_col: 'count',  # Frequency
            amount_col: 'sum'  # Monetary
        }).rename(columns={
            date_col: 'recency',
            customer_id_col: 'frequency',
            amount_col: 'monetary'
        })
        
        # Store RFM scores
        self.rfm_scores_ = rfm
        return rfm
    
    def calculate_rfm_scores(self, rfm_scores):
        """Calculate RFM scores (1-5 scale) based on quintiles."""
        # Reverse recency (higher is better)
        rfm_scores['recency_score'] = pd.qcut(
            rfm_scores['recency'], 
            q=5, 
            labels=[5, 4, 3, 2, 1]
        ).astype(int)
        
        # Higher frequency is better
        rfm_scores['frequency_score'] = pd.qcut(
            rfm_scores['frequency'].rank(method='first'),
            q=5,
            labels=[1, 2, 3, 4, 5]
        ).astype(int)
        
        # Higher monetary is better
        rfm_scores['monetary_score'] = pd.qcut(
            rfm_scores['monetary'],
            q=5,
            labels=[1, 2, 3, 4, 5]
        ).astype(int)
        
        return rfm_scores
    
    def fit(self, df, customer_id_col='CustomerId', 
           date_col='TransactionStartTime', amount_col='Amount'):
        """Fit the RFM model and identify high-risk customers.
        
        Args:
            df (pd.DataFrame): Input transaction data
            customer_id_col (str): Name of customer ID column
            date_col (str): Name of transaction date column
            amount_col (str): Name of transaction amount column
            
        Returns:
            self: Returns the instance itself
        """
        # Calculate RFM metrics
        rfm_scores = self.calculate_rfm(df, customer_id_col, date_col, amount_col)
        
        # Scale the features
        X = rfm_scores[['recency', 'frequency', 'monetary']]
        
        # Fit the pipeline (scaling + clustering)
        self.pipeline.fit(X)
        
        # Get cluster assignments
        rfm_scores['cluster'] = self.pipeline.predict(X)
        
        # Calculate cluster centers in the original feature space
        self.cluster_centers_ = self.scaler.inverse_transform(
            self.kmeans.cluster_centers_
        )
        
        # Identify the high-risk cluster (highest recency, lowest frequency and monetary)
        # We calculate a composite score where higher is riskier
        cluster_risk_scores = (
            self.cluster_centers_[:, 0] -  # Higher recency is riskier
            self.cluster_centers_[:, 1] -  # Lower frequency is riskier
            self.cluster_centers_[:, 2]    # Lower monetary is riskier
        )
        
        self.high_risk_cluster_ = np.argmax(cluster_risk_scores)
        logger.info(f"Identified high-risk cluster: {self.high_risk_cluster_}")
        
        return self
    
    def transform(self, df, customer_id_col='CustomerId'):
        """Add is_high_risk column to the input DataFrame.
        
        Args:
            df (pd.DataFrame): Input transaction data
            customer_id_col (str): Name of customer ID column
            
        Returns:
            pd.DataFrame: Original DataFrame with added 'is_high_risk' column
        """
        if self.rfm_scores_ is None:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        # Create a mapping from customer ID to high-risk status
        risk_mapping = (self.rfm_scores_['cluster'] == self.high_risk_cluster_).astype(int)
        risk_mapping = risk_mapping.rename('is_high_risk')
        
        # Merge with original data
        result_df = df.merge(
            risk_mapping,
            left_on=customer_id_col,
            right_index=True,
            how='left'
        )
        
        # Fill any missing values with 0 (not high risk)
        result_df['is_high_risk'] = result_df['is_high_risk'].fillna(0).astype(int)
        
        logger.info(f"High-risk customers: {result_df['is_high_risk'].sum():,} "
                   f"({result_df['is_high_risk'].mean():.1%} of total)")
        
        return result_df
    
    def fit_transform(self, df, customer_id_col='CustomerId', 
                     date_col='TransactionStartTime', amount_col='Amount'):
        """Fit the model and transform the data in one step."""
        self.fit(df, customer_id_col, date_col, amount_col)
        return self.transform(df, customer_id_col)

def create_proxy_target(df, customer_id_col='CustomerId',
                       date_col='TransactionStartTime', 
                       amount_col='Amount',
                       snapshot_date=None,
                       n_clusters=3,
                       random_state=42):
    """Convenience function to create proxy target variable.
    
    Args:
        df (pd.DataFrame): Input transaction data
        customer_id_col (str): Name of customer ID column
        date_col (str): Name of transaction date column
        amount_col (str): Name of transaction amount column
        snapshot_date (str or datetime, optional): Reference date for RFM calculation
        n_clusters (int, optional): Number of clusters for K-Means
        random_state (int, optional): Random state for reproducibility
        
    Returns:
        pd.DataFrame: Original DataFrame with added 'is_high_risk' column
    """
    engineer = RFMTargetEngineer(
        snapshot_date=snapshot_date,
        n_clusters=n_clusters,
        random_state=random_state
    )
    
    return engineer.fit_transform(
        df, 
        customer_id_col=customer_id_col,
        date_col=date_col,
        amount_col=amount_col
    )
