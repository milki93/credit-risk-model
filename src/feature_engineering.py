import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Dict, Optional, Union, Any, Tuple
import logging
from functools import reduce

# Import RFM target engineering
from src.target_engineering import RFMTargetEngineer, create_proxy_target
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    FunctionTransformer,
    LabelEncoder,
    KBinsDiscretizer
)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from xverse.transformer import WOE
from typing import List, Dict, Optional, Union, Any, Tuple
import logging
from functools import reduce

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomWOETransformer(BaseEstimator, TransformerMixin):
    """
    Custom Weight of Evidence (WOE) transformer that's compatible with the latest pandas versions.
    This is a simplified version that works with binary classification.
    """
    def __init__(self, target_col: str, min_samples_leaf: int = 20, n_bins: int = 10):
        self.target_col = target_col
        self.min_samples_leaf = min_samples_leaf
        self.n_bins = n_bins
        self.woe_dict = {}
        
    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None):
        """
        Fit the WOE transformer on the training data.
        
        Args:
            X: Input features as a pandas DataFrame
            y: Target values (not used, kept for compatibility)
            
        Returns:
            self: The fitted transformer
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        # If target is not in X, use y if provided
        if self.target_col not in X.columns:
            if y is not None:
                X[self.target_col] = y
            else:
                raise ValueError(f"Target column '{self.target_col}' not found in X and y is None")
        
        # Calculate WOE for each numeric column
        for col in X.select_dtypes(include=['number']).columns:
            if col == self.target_col:
                continue
                
            # Skip if constant
            if X[col].nunique() <= 1:
                continue
                
            # Create bins with equal number of samples
            try:
                X['temp_bin'] = pd.qcut(
                    X[col], 
                    q=self.n_bins, 
                    duplicates='drop',
                    labels=False
                )
                
                # Calculate WOE for each bin
                temp_df = X[[col, 'temp_bin', self.target_col]].copy()
                total_good = temp_df[self.target_col].sum()
                total_bad = len(temp_df) - total_good
                
                woe_values = {}
                for bin_val in temp_df['temp_bin'].unique():
                    bin_data = temp_df[temp_df['temp_bin'] == bin_val]
                    bin_good = bin_data[self.target_col].sum()
                    bin_bad = len(bin_data) - bin_good
                    
                    # Apply Laplace smoothing to avoid division by zero
                    good_ratio = (bin_good + 0.5) / (total_good + 1)
                    bad_ratio = (bin_bad + 0.5) / (total_bad + 1)
                    
                    if good_ratio > 0 and bad_ratio > 0:
                        woe = np.log(good_ratio / bad_ratio)
                        woe_values[bin_val] = woe
                
                self.woe_dict[col] = woe_values
                
            except Exception as e:
                logger.warning(f"Could not calculate WOE for {col}: {str(e)}")
                
        return self
        
    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Transform the input data using the fitted WOE values.
        
        Args:
            X: Input features to transform
            
        Returns:
            np.ndarray: Transformed features
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        X_transformed = X.copy()
        
        # Apply WOE transformation to each column
        for col, woe_values in self.woe_dict.items():
            if col not in X.columns:
                continue
                
            # Create bins using the same quantiles as in fit
            try:
                X_transformed['temp_bin'] = pd.qcut(
                    X[col], 
                    q=self.n_bins, 
                    duplicates='drop',
                    labels=False
                )
                
                # Map bins to WOE values
                X_transformed[col] = X_transformed['temp_bin'].map(woe_values)
                
                # Fill any remaining NaN values with the overall average WOE
                if X_transformed[col].isna().any():
                    overall_woe = np.mean(list(woe_values.values()))
                    X_transformed[col] = X_transformed[col].fillna(overall_woe)
                    
            except Exception as e:
                logger.warning(f"Could not apply WOE transformation to {col}: {str(e)}")
                # If transformation fails, drop the column
                X_transformed = X_transformed.drop(columns=[col], errors='ignore')
        
        # Clean up
        X_transformed = X_transformed.drop(columns=['temp_bin'], errors='ignore')
        
        return X_transformed.values if isinstance(X_transformed, pd.DataFrame) else X_transformed

class XenteFeatures(BaseEstimator, TransformerMixin):
    """
    Compute customer behavior features for Xente dataset.
    Includes RFM (Recency, Frequency, Monetary) and other behavioral features.
    """
    def __init__(self, 
                 customer_id: str = 'AccountId',
                 date_col: str = 'TransactionStartTime',
                 amount_col: str = 'PurchaseValue',
                 fraud_col: str = 'FraudResult'):
        self.customer_id = customer_id
        self.date_col = date_col
        self.amount_col = amount_col
        self.fraud_col = fraud_col
        self.reference_date = None
        self.customer_features_ = None
        self.feature_names_ = [
            'recency_days', 'transaction_count', 'avg_amount',
            'std_amount', 'total_amount', 'fraud_rate'
        ]
        
    def fit(self, X: pd.DataFrame, y=None):
        # Ensure datetime format
        X = X.copy()
        X[self.date_col] = pd.to_datetime(X[self.date_col])
        
        # Set reference date (latest transaction + 1 day)
        self.reference_date = X[self.date_col].max() + pd.Timedelta(days=1)
        
        # Calculate customer features
        agg_funcs = {
            self.date_col: [
                ('recency', lambda x: (self.reference_date - x.max()).days),
                ('frequency', 'count')
            ],
            self.amount_col: [
                ('avg_amount', 'mean'),
                ('std_amount', 'std'),
                ('total_amount', 'sum'),
                ('count_nonzero', lambda x: (x > 0).sum())
            ]
        }
        
        # Add fraud rate if fraud column exists and is in X
        if self.fraud_col and self.fraud_col in X.columns:
            agg_funcs[self.fraud_col] = [('fraud_rate', 'mean')]
        
        # Group by customer and calculate features
        customer_features = X.groupby(self.customer_id).agg(agg_funcs)
        
        # Flatten multi-index columns
        customer_features.columns = ['_'.join(col).strip() for col in customer_features.columns.values]
        
        # Calculate additional features
        customer_features['avg_days_between_transactions'] = \
            customer_features[f'{self.date_col}_frequency'] / 30  # Approximate as 30 days
            
        customer_features['transaction_frequency'] = \
            customer_features[f'{self.date_col}_frequency'] / 30  # Transactions per day
        
        # Store the customer features for transformation
        self.customer_features_ = customer_features
        
        # Update feature names
        self.feature_names_ = customer_features.columns.tolist()
        
        self.customer_features_ = customer_features
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Ensure datetime format
        X = X.copy()
        X[self.date_col] = pd.to_datetime(X[self.date_col])
        
        # If reference_date is not set, set it now
        if self.reference_date is None:
            self.reference_date = X[self.date_col].max() + pd.Timedelta(days=1)
        
        # If features are not calculated, calculate them
        if self.customer_features_ is None:
            self.fit(X)
        
        # Merge features back to the original dataframe
        X_transformed = X.merge(
            self.customer_features_,
            left_on=self.customer_id,
            right_index=True,
            how='left'
        )
        
        # Return only the engineered features
        return X_transformed[self.feature_names_]
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation"""
        return np.array(self.feature_names_)

class TimeFeatures(BaseEstimator, TransformerMixin):
    """
    Extract sophisticated time-based features from transaction timestamps.
    Includes cyclical encoding for time features and business hour indicators.
    """
    def __init__(self, date_col: str = 'TransactionStartTime'):
        self.date_col = date_col
        self.feature_names_ = [
            'hour_sin', 'hour_cos',
            'day_of_week_sin', 'day_of_week_cos',
            'day_of_month_sin', 'day_of_month_cos',
            'month_sin', 'month_cos',
            'is_weekend', 'is_business_hours',
            'is_night', 'is_morning_rush',
            'is_evening_rush', 'season'
        ]
        
    def _cyclical_encoding(self, values: np.ndarray, period: int) -> tuple:
        """Encode cyclical features using sine and cosine transformation."""
        sin_values = np.sin(2 * np.pi * values / period)
        cos_values = np.cos(2 * np.pi * values / period)
        return sin_values, cos_values
        
    def fit(self, X: pd.DataFrame, y=None):
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Create a copy to avoid SettingWithCopyWarning
        X = X.copy()
        
        # Convert to datetime if needed
        if isinstance(X[self.date_col], str) or not pd.api.types.is_datetime64_any_dtype(X[self.date_col]):
            X[self.date_col] = pd.to_datetime(X[self.date_col])
        
        # Initialize output DataFrame
        time_features = pd.DataFrame(index=X.index)
        
        # Extract basic time components
        time_features['hour'] = X[self.date_col].dt.hour
        time_features['day_of_week'] = X[self.date_col].dt.dayofweek
        time_features['day_of_month'] = X[self.date_col].dt.day
        time_features['month'] = X[self.date_col].dt.month
        
        # Cyclical encoding for hour (24-hour cycle)
        time_features['hour_sin'], time_features['hour_cos'] = self._cyclical_encoding(
            time_features['hour'], 24)
        
        # Cyclical encoding for day of week (7-day cycle)
        time_features['day_of_week_sin'], time_features['day_of_week_cos'] = self._cyclical_encoding(
            time_features['day_of_week'], 7)
        
        # Cyclical encoding for day of month (assuming 30-day month)
        time_features['day_of_month_sin'], time_features['day_of_month_cos'] = self._cyclical_encoding(
            time_features['day_of_month'], 30)
        
        # Cyclical encoding for month (12-month cycle)
        time_features['month_sin'], time_features['month_cos'] = self._cyclical_encoding(
            time_features['month'], 12)
        
        # Binary time indicators
        time_features['is_weekend'] = X[self.date_col].dt.dayofweek.isin([5, 6]).astype(int)
        
        # Business hours (9 AM to 5 PM)
        time_features['is_business_hours'] = ((time_features['hour'] >= 9) & 
                                            (time_features['hour'] < 17)).astype(int)
        
        # Time of day indicators
        time_features['is_night'] = ((time_features['hour'] >= 22) | 
                                   (time_features['hour'] < 6)).astype(int)
        
        # Morning rush hour (7-10 AM)
        time_features['is_morning_rush'] = ((time_features['hour'] >= 7) & 
                                          (time_features['hour'] < 10)).astype(int)
        
        # Evening rush hour (4-7 PM)
        time_features['is_evening_rush'] = ((time_features['hour'] >= 16) & 
                                          (time_features['hour'] < 19)).astype(int)
        
        # Season (1:Winter, 2:Spring, 3:Summer, 4:Fall)
        month = X[self.date_col].dt.month
        time_features['season'] = (month % 12 + 3) // 3
        
        # Drop intermediate columns
        time_features = time_features.drop(
            ['hour', 'day_of_week', 'day_of_month', 'month'], 
            axis=1, errors='ignore')
        
        return time_features
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation"""
        return np.array(self.feature_names_)

class MerchantFeatures(BaseEstimator, TransformerMixin):
    """Extract merchant-related features."""
    
    def __init__(self, customer_id: str = 'AccountId', merchant_col: str = 'MerchantId', 
                 amount_col: str = 'PurchaseValue', fraud_col: str = 'FraudResult'):
        self.customer_id = customer_id
        self.merchant_col = merchant_col
        self.amount_col = amount_col
        self.fraud_col = fraud_col
        self.merchant_stats_ = None
        self.merchant_affinity_ = None
        self.feature_names_ = [
            'merchant_avg_amount', 'merchant_std_amount',
            'merchant_tx_count', 'customer_merchant_affinity'
        ]
        
    def fit(self, X: pd.DataFrame, y=None):
        # Calculate merchant statistics
        agg_funcs = {
            self.amount_col: [
                ('merchant_avg_amount', 'mean'),
                ('merchant_std_amount', 'std'),
                ('merchant_tx_count', 'count')
            ]
        }
        
        # Add fraud rate if fraud column exists and is in X
        if self.fraud_col and self.fraud_col in X.columns:
            agg_funcs[self.fraud_col] = [('merchant_fraud_rate', 'mean')]
        
        # Calculate merchant stats
        merchant_stats = X.groupby(self.merchant_col).agg(agg_funcs)
        
        # Flatten multi-index columns
        merchant_stats.columns = [col[1] if col[1] else col[0] for col in merchant_stats.columns]
        merchant_stats = merchant_stats.reset_index()
        
        # Calculate customer-merchant affinity (how often a customer uses this merchant)
        customer_merchant_counts = X.groupby([self.customer_id, self.merchant_col]).size().reset_index(name='tx_count')
        total_tx_per_customer = X.groupby(self.customer_id).size().reset_index(name='total_tx')
        
        # Calculate affinity as percentage of customer's transactions with this merchant
        merchant_affinity = pd.merge(
            customer_merchant_counts,
            total_tx_per_customer,
            on=self.customer_id
        )
        merchant_affinity['affinity'] = merchant_affinity['tx_count'] / merchant_affinity['total_tx']
        
        # Store the merchant stats and affinity for transformation
        self.merchant_stats_ = merchant_stats
        self.merchant_affinity_ = merchant_affinity
        
        # Set feature names
        self.feature_names_ = [
            'merchant_avg_amount',
            'merchant_std_amount',
            'merchant_tx_count'
        ]
        
        # Add fraud rate feature if it exists
        if self.fraud_col and self.fraud_col in X.columns:
            self.feature_names_.append('merchant_fraud_rate')
            
        self.feature_names_.append('customer_merchant_affinity')
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.merchant_stats_ is None:
            self.fit(X)
            
        # Make a copy of X to avoid SettingWithCopyWarning
        X = X.copy()
        
        # Merge merchant stats
        X_transformed = X.merge(
            self.merchant_stats_,
            on=self.merchant_col,
            how='left'
        )
        
        # Check if we have merchant_affinity_ data
        if hasattr(self, 'merchant_affinity_') and self.merchant_affinity_ is not None:
            # Merge merchant affinity using the correct column name 'affinity' which was set in fit()
            X_transformed = X_transformed.merge(
                self.merchant_affinity_[[self.customer_id, self.merchant_col, 'affinity']],
                on=[self.customer_id, self.merchant_col],
                how='left'
            )
            # Rename 'affinity' to 'customer_merchant_affinity' to match feature_names_
            X_transformed = X_transformed.rename(columns={'affinity': 'customer_merchant_affinity'})
        else:
            # If no affinity data, create a column of zeros
            X_transformed['customer_merchant_affinity'] = 0.0
        
        # Fill missing values
        for col in ['merchant_avg_amount', 'merchant_std_amount', 'merchant_tx_count']:
            if col in X_transformed.columns:
                X_transformed[col] = X_transformed[col].fillna(0)
        
        # Ensure all required features are present
        for col in self.feature_names_:
            if col not in X_transformed.columns:
                X_transformed[col] = 0.0
        
        # Select only the new features
        return X_transformed[self.feature_names_].values
    
    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_)


class ColumnDropper(BaseEstimator, TransformerMixin):
    """Drop specified columns from the DataFrame."""
    def __init__(self, columns_to_drop=None):
        self.columns_to_drop = columns_to_drop or []
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.drop(columns=self.columns_to_drop, errors='ignore')

class DataFrameWrapper(BaseEstimator, TransformerMixin):
    """Wrapper to ensure input/output is a pandas DataFrame."""
    def __init__(self, transformer):
        self.transformer = transformer
        self.feature_names_ = None
        
    def _ensure_dataframe(self, X, y=None):
        """Ensure X is a DataFrame and optionally convert y to Series."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if y is not None and not isinstance(y, pd.Series):
            y = pd.Series(y)
        return X, y
        
    def fit(self, X, y=None):
        X_df, y_series = self._ensure_dataframe(X, y)
        self.transformer.fit(X_df, y_series)
        if hasattr(self.transformer, 'feature_names_'):
            self.feature_names_ = self.transformer.feature_names_
        return self
        
    def transform(self, X):
        X_df, _ = self._ensure_dataframe(X)
        result = self.transformer.transform(X_df)
        
        # If the result is a numpy array, convert to DataFrame with column names if available
        if isinstance(result, np.ndarray):
            if hasattr(self, 'feature_names_') and self.feature_names_ is not None:
                return pd.DataFrame(result, columns=self.feature_names_, index=X_df.index)
            return pd.DataFrame(result, index=X_df.index)
        return result
    
    def get_feature_names_out(self, input_features=None):
        if hasattr(self.transformer, 'get_feature_names_out'):
            return self.transformer.get_feature_names_out(input_features)
        elif hasattr(self.transformer, 'feature_names_'):
            return self.transformer.feature_names_
        return None

def create_feature_engineering_pipeline(
    numerical_cols: List[str],
    categorical_cols: List[str],
    target_col: Optional[str] = None,
    customer_id_col: str = 'AccountId',
    date_col: str = 'TransactionStartTime',
    amount_col: str = 'PurchaseValue',
    merchant_col: str = 'MerchantId',
    fraud_col: str = 'FraudResult'
) -> Pipeline:
    """
    Create a comprehensive feature engineering pipeline for the Xente dataset.
    
    Args:
        numerical_cols: List of numerical column names
        categorical_cols: List of categorical column names
        target_col: Name of the target column (for WoE encoding)
        customer_id_col: Name of the customer ID column
        date_col: Name of the date column
        amount_col: Name of the transaction amount column
        merchant_col: Name of the merchant ID column
        fraud_col: Name of the fraud indicator column
        
    Returns:
        A scikit-learn Pipeline with all feature engineering steps
    """
    # 1. Customer behavior features (RFM + more)
    customer_pipeline = Pipeline([
        ('customer_features', XenteFeatures(
            customer_id=customer_id_col,
            date_col=date_col,
            amount_col=amount_col,
            fraud_col=fraud_col
        )),
        ('imputer', SimpleImputer(strategy='median')),
        ('variance_threshold', VarianceThreshold(threshold=0.0)),
        ('scaler', StandardScaler())
    ])
    
    # 2. Time-based features
    time_pipeline = Pipeline([
        ('time_features', TimeFeatures(date_col=date_col)),
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # 3. Merchant features
    merchant_pipeline = Pipeline([
        ('merchant_features', MerchantFeatures(
            merchant_col=merchant_col,
            amount_col=amount_col,
            customer_id=customer_id_col
        )),
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # 4. Numerical features preprocessing
    numerical_pipeline = Pipeline([
        ('selector', ColumnTransformer([
            ('num', 'passthrough', numerical_cols)
        ], remainder='drop')),
        ('imputer', SimpleImputer(strategy='median')),
        ('variance_threshold', VarianceThreshold(threshold=0.0)),
        ('scaler', StandardScaler())
    ])
    
    # 5. Categorical features preprocessing
    categorical_pipeline = Pipeline([
        ('selector', ColumnTransformer([
            ('cat', 'passthrough', categorical_cols)
        ], remainder='drop')),
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(
            handle_unknown='ignore',
            sparse_output=False,
            min_frequency=0.01  # Reduce cardinality
        ))
    ])
    
    # Combine all feature pipelines
    feature_union = FeatureUnion([
        ('customer', customer_pipeline),
        ('time', time_pipeline),
        ('merchant', merchant_pipeline),
        ('numerical', numerical_pipeline),
        ('categorical', categorical_pipeline)
    ])
    
    # Create the full pipeline
    pipeline_steps = [
        ('feature_union', feature_union),
        
        # Convert to DataFrame for WOE transformer
        ('to_dataframe', FunctionTransformer(
            func=lambda x: pd.DataFrame(x, columns=[f'feature_{i}' for i in range(x.shape[1])]),
            validate=False
        ))
    ]
    
    # Add WoE transformation if target is provided
    if target_col is not None:
        # Create a custom transformer to handle the target column
        class TargetExtractor(BaseEstimator, TransformerMixin):
            def __init__(self, target_col):
                self.target_col = target_col
                
            def fit(self, X, y=None):
                return self
                
            def transform(self, X):
                # Handle both pandas DataFrame and numpy array inputs
                if hasattr(X, 'iloc'):  # pandas DataFrame
                    if self.target_col in X.columns:
                        return pd.DataFrame(X[self.target_col], columns=[self.target_col])
                    else:
                        # If target column not found, try to use y if it was passed to fit
                        if hasattr(self, 'y_') and self.y_ is not None:
                            if hasattr(self.y_, 'values'):
                                return pd.DataFrame(self.y_.values, columns=[self.target_col])
                            else:
                                return pd.DataFrame(np.array(self.y_), columns=[self.target_col])
                        else:
                            raise ValueError(f"Target column '{self.target_col}' not found in input data")
                elif hasattr(X, 'shape'):  # numpy array
                    # If it's already a numpy array, assume the target is the last column
                    return pd.DataFrame(X[:, -1], columns=[self.target_col])
                else:
                    raise ValueError("Input must be a pandas DataFrame or numpy array")
                    
            def fit_transform(self, X, y=None, **fit_params):
                if y is not None:
                    self.y_ = y
                return self.fit(X, y).transform(X)
        
        # Add target extraction and WOE transformation
        pipeline_steps.extend([
            ('target_extractor', TargetExtractor(target_col=target_col)),
            ('woe', WOE()),
            # Convert to numpy array for final steps
            ('to_numpy', FunctionTransformer(
                func=lambda x: x.values,
                validate=False
            ))
        ])
    else:
        # If not using WOE, still need to handle the target column
        if target_col is not None:
            pipeline_steps.append(('target_remover', ColumnDropper(columns_to_drop=[target_col])))
    
    return Pipeline(steps=pipeline_steps)

def get_feature_names(pipeline: Pipeline) -> List[str]:
    """
    Extract feature names from the pipeline.
    
    Args:
        pipeline: Fitted feature engineering pipeline
        
    Returns:
        List of feature names
    """
    feature_names = []
    
    # Get feature names from feature engineering steps
    for name, step in pipeline.named_steps.items():
        if hasattr(step, 'get_feature_names'):
            feature_names.extend(step.get_feature_names_out())
        elif hasattr(step, 'named_steps'):
            # Handle nested pipelines
            for sub_name, sub_step in step.named_steps.items():
                if hasattr(sub_step, 'get_feature_names'):
                    feature_names.extend(sub_step.get_feature_names_out())
    
    return feature_names

def preprocess_data(
    df: pd.DataFrame,
    numerical_cols: List[str],
    categorical_cols: List[str],
    target_col: Optional[str] = None,  # Make target_col optional
    customer_id_col: str = 'CustomerId',
    date_col: str = 'TransactionStartTime',
    amount_col: str = 'Amount',
    merchant_col: str = 'ProviderId',
    test_size: float = 0.2,
    random_state: int = 42,
    drop_original: bool = True,
    create_rfm_target: bool = True,  # New parameter to control RFM target creation
    rfm_target_col: str = 'is_high_risk'  # New parameter for RFM target column name
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """
    Preprocess the input data by applying feature engineering and transformations.
    
    This function performs the following steps:
    1. Validates input data and required columns
    2. Applies feature engineering transformations
    3. Handles target encoding using WOE (Weight of Evidence)
    4. Splits data into training and test sets
    5. Returns processed features and targets
    
    Args:
        df: Input DataFrame containing the raw data
        numerical_cols: List of numerical column names
        categorical_cols: List of categorical column names
        target_col: Name of the target column (default: 'FraudResult')
        customer_id_col: Name of the customer ID column (default: 'CustomerId')
        date_col: Name of the datetime column (default: 'TransactionStartTime')
        amount_col: Name of the transaction amount column (default: 'Amount')
        merchant_col: Name of the merchant ID column (default: 'ProviderId')
        test_size: Proportion of data to use for testing (default: 0.2)
        random_state: Random seed for reproducibility (default: 42)
        drop_original: Whether to drop original columns (default: True)
        
    Returns:
        If target_col is provided, returns (X_train, X_test, y_train, y_test, feature_names)
        Otherwise, returns (X_processed, feature_names)
    """
    # Make a copy of the input DataFrame
    df_processed = df.copy()
    
    # Create RFM-based target if requested and no target column is provided
    if create_rfm_target and target_col is None and all(col in df_processed.columns for col in [customer_id_col, date_col, amount_col]):
        logger.info("Creating RFM-based target variable...")
        snapshot_date = df_processed[date_col].max() if date_col in df_processed.columns else None
        
        # Create RFM target
        df_processed = create_proxy_target(
            df=df_processed,
            customer_id_col=customer_id_col,
            date_col=date_col,
            amount_col=amount_col,
            snapshot_date=snapshot_date,
            n_clusters=3,
            random_state=random_state
        )
        
        # Set the target column to the newly created RFM target
        target_col = rfm_target_col
        logger.info(f"Created RFM-based target '{target_col}'. Value counts:")
        logger.info(df_processed[target_col].value_counts(normalize=True).to_string())
    
    # Ensure required columns exist
    required_cols = {
        'customer_id': customer_id_col,
        'date': date_col,
        'amount': amount_col,
        'merchant': merchant_col,
        'fraud': target_col
    }
    
    # Check for missing columns
    missing_cols = [name for name, col in required_cols.items() 
                   if col is not None and col not in df_processed.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Handle missing values in critical columns
    df_processed[date_col] = pd.to_datetime(df_processed[date_col], errors='coerce')
    df_processed[amount_col] = pd.to_numeric(df_processed[amount_col], errors='coerce')
    
    # Drop rows with missing critical values
    critical_cols = [customer_id_col, date_col, amount_col]
    df_processed = df_processed.dropna(subset=critical_cols).copy()
    
    # Sort by customer and date for time-based features
    df_processed = df_processed.sort_values([customer_id_col, date_col])
    
    # Ensure the target column is not in features
    if target_col in numerical_cols:
        numerical_cols = [col for col in numerical_cols if col != target_col]
    if target_col in categorical_cols:
        categorical_cols = [col for col in categorical_cols if col != target_col]
    
    # Prepare features and target
    X = df_processed.copy()
    y = X[target_col].copy() if target_col else None
    
    # First, create a pipeline without WOE transformation for feature engineering
    feature_pipeline = create_feature_engineering_pipeline(
        numerical_cols=numerical_cols,
        categorical_cols=categorical_cols,
        target_col=None,  # Don't include target in feature engineering
        customer_id_col=customer_id_col,
        date_col=date_col,
        amount_col=amount_col,
        merchant_col=merchant_col,
        fraud_col=target_col  # Still use target for fraud-related features
    )
    
    # If target column exists, split the data
    if target_col and target_col in df_processed.columns:
        from sklearn.model_selection import train_test_split
        
        # For RFM target, we need to ensure we have the same split for features and target
        if target_col == rfm_target_col and create_rfm_target:
            # Get the customer-level target
            customer_targets = df_processed[[customer_id_col, target_col]].drop_duplicates()
            
            # Split customers into train and test
            train_customers, test_customers = train_test_split(
                customer_targets[customer_id_col].unique(),
                test_size=test_size,
                random_state=random_state,
                stratify=customer_targets[target_col] if len(customer_targets[target_col].unique()) > 1 else None
            )
            
            # Split the data based on customer IDs
            train_mask = df_processed[customer_id_col].isin(train_customers)
            test_mask = df_processed[customer_id_col].isin(test_customers)
            
            X = df_processed.drop(columns=[target_col] if drop_original and target_col in df_processed.columns else [])
            y = df_processed[target_col]
            
            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]
        else:
            # Original random split for non-RFM targets
            X = df_processed.drop(columns=[target_col] if drop_original and target_col in df_processed.columns else [])
            y = df_processed[target_col]
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=test_size, 
                random_state=random_state, 
                stratify=y if len(y.unique()) > 1 else None
            )
        
        # 1. First, fit and transform the features
        X_train_features = feature_pipeline.fit_transform(X_train)
        X_test_features = feature_pipeline.transform(X_test)
        
        # 2. Apply WOE transformation separately to avoid data leakage
        if target_col is not None:
            # Use our custom WOE transformer
            woe_transformer = CustomWOETransformer(target_col=target_col)
            
            # Fit on training data
            X_train_processed = woe_transformer.fit_transform(
                X_train_features, 
                y_train.values if hasattr(y_train, 'values') else y_train
            )
            
            # Transform test data
            X_test_processed = woe_transformer.transform(X_test_features)
            
            # Generate meaningful feature names
            if hasattr(X_train_features, 'columns'):
                # Use original column names with _woe suffix for transformed features
                feature_names = [f"{col}_woe" for col in X_train_features.columns 
                              if col in woe_transformer.woe_dict]
                
                # Add any non-transformed numeric features
                numeric_cols = X_train_features.select_dtypes(include=['number']).columns
                for col in numeric_cols:
                    if col not in woe_transformer.woe_dict and col != target_col:
                        feature_names.append(col)
            else:
                # Fallback to generic names if no column information is available
                feature_names = [f"feature_{i}" for i in range(X_train_processed.shape[1])]
            
            logger.info(f"Generated {len(feature_names)} features after WOE transformation")
        else:
            X_train_processed = X_train_features
            X_test_processed = X_test_features
            feature_names = get_feature_names(feature_pipeline)
        
        # Convert to DataFrames if not already
        if not isinstance(X_train_processed, pd.DataFrame):
            # Ensure we have the correct number of feature names
            if hasattr(X_train_processed, 'shape') and len(X_train_processed.shape) == 2:
                n_features = X_train_processed.shape[1]
                if len(feature_names) != n_features:
                    logger.warning(f"Mismatch in number of features: expected {n_features}, got {len(feature_names)}. Using default feature names.")
                    feature_names = [f'feature_{i}' for i in range(n_features)]
            
            # Create DataFrames without index to avoid shape mismatches
            X_train_df = pd.DataFrame(
                X_train_processed,
                columns=feature_names[:X_train_processed.shape[1]] if hasattr(X_train_processed, 'shape') and len(X_train_processed.shape) == 2 else None
            )
            
            X_test_df = pd.DataFrame(
                X_test_processed,
                columns=feature_names[:X_test_processed.shape[1]] if hasattr(X_test_processed, 'shape') and len(X_test_processed.shape) == 2 else None
            )
            
            # Set indices if available
            if hasattr(X_train, 'index') and len(X_train.index) == len(X_train_df):
                X_train_df.index = X_train.index
            if hasattr(X_test, 'index') and len(X_test.index) == len(X_test_df):
                X_test_df.index = X_test.index
        else:
            X_train_df = X_train_processed
            X_test_df = X_test_processed
        
        # Add back the target column if needed
        if drop_original:
            return X_train_df, X_test_df, y_train, y_test, feature_names
        else:
            # Add back original columns if needed
            X_train_full = pd.concat([X_train_df, X_train], axis=1)
            X_test_full = pd.concat([X_test_df, X_test], axis=1)
            return X_train_full, X_test_full, y_train, y_test, feature_names
    else:
        # No target column provided, just transform the data
        X_processed = pipeline.fit_transform(X)
        feature_names = get_feature_names(pipeline)
        
        # Convert to DataFrame
        X_df = pd.DataFrame(X_processed, columns=feature_names, index=X.index)
        
        if drop_original:
            return X_df, feature_names
        else:
            # Add back original columns if needed
            return pd.concat([X_df, X], axis=1), feature_names

# Example usage
if __name__ == "__main__":
    # Example data loading (replace with actual data loading)
    # df = pd.read_csv("data/raw/transactions.csv")
    
    # Define column types based on Xente dataset
    NUMERICAL_COLS = [
        'Amount', 'Value', 'TransactionAmount',
        'TransactionAmountUSD', 'TransactionAmountLocal'
    ]
    
    CATEGORICAL_COLS = [
        'CurrencyCode', 'CountryCode', 'ProviderId',
        'ChannelId', 'ProductCategory', 'ProductId',
        'BatchId', 'AccountCode', 'CustomerId',
        'SubscriptionId', 'MerchantId'
    ]
    
    # Target column for supervised learning
    TARGET_COL = 'FraudResult'  # Binary: 0 for non-fraud, 1 for fraud
    
    # Example 1: Preprocess data for model training
    # X_train, X_test, y_train, y_test, feature_names = preprocess_data(
    #     df,
    #     numerical_cols=NUMERICAL_COLS,
    #     categorical_cols=CATEGORICAL_COLS,
    #     target_col=TARGET_COL,
    #     test_size=0.2,
    #     random_state=42
    # )
    # 
    # print(f"Training set shape: {X_train.shape}")
    # print(f"Test set shape: {X_test.shape}")
    # print(f"Number of features: {len(feature_names)}")
    # print(f"Feature names: {feature_names[:10]}...")  # Print first 10 features
    # 
    # # Example 2: Preprocess new data for prediction (no target column)
    # X_new = df.drop(columns=[TARGET_COL] if TARGET_COL in df.columns else [])
    # X_processed, feature_names = preprocess_data(
    #     X_new,
    #     numerical_cols=NUMERICAL_COLS,
    #     categorical_cols=CATEGORICAL_COLS,
    #     target_col=None  # No target for prediction
    # )
    # 
    # print(f"Processed data shape: {X_processed.shape}")
    
    pass
