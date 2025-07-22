"""
Utility functions for data processing, splitting, and transformation.
"""
from typing import Tuple, List, Union, Optional, Dict, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def split_data(
    data: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training and testing sets.
    
    Args:
        data: Input DataFrame containing features and target
        target_col: Name of the target column
        test_size: Proportion of the dataset to include in the test split (default: 0.2)
        random_state: Controls the shuffling applied to the data (default: 42)
        stratify: Whether to perform stratified splitting based on the target (default: True)
        
    Returns:
        Tuple containing X_train, X_test, y_train, y_test
    """
    X = data.drop(columns=[target_col])
    y = data[target_col]
    
    stratify_col = y if stratify else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=stratify_col,
        shuffle=True
    )
    
    return X_train, X_test, y_train, y_test

def get_numeric_categorical_columns(df: pd.DataFrame, 
                                  exclude_cols: Optional[List[str]] = None) -> Tuple[List[str], List[str]]:
    """
    Identify numeric and categorical columns in a DataFrame.
    
    Args:
        df: Input DataFrame
        exclude_cols: List of columns to exclude from the result
        
    Returns:
        Tuple of (numeric_columns, categorical_columns)
    """
    if exclude_cols is None:
        exclude_cols = []
    
    # Get all columns not in exclude_cols
    columns = [col for col in df.columns if col not in exclude_cols]
    
    # Identify numeric and categorical columns
    numeric_cols = df[columns].select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
    categorical_cols = df[columns].select_dtypes(include=['category', 'object', 'bool']).columns.tolist()
    
    # Handle datetime columns (exclude them from both)
    datetime_cols = df[columns].select_dtypes(include=['datetime64', 'timedelta64']).columns.tolist()
    
    # Remove datetime columns from both lists
    numeric_cols = [col for col in numeric_cols if col not in datetime_cols]
    categorical_cols = [col for col in categorical_cols if col not in datetime_cols]
    
    return numeric_cols, categorical_cols

def create_preprocessing_pipeline(
    numeric_cols: List[str],
    categorical_cols: List[str],
    numeric_strategy: str = 'median',
    categorical_strategy: str = 'constant',
    scaling: str = 'standard'
) -> ColumnTransformer:
    """
    Create a preprocessing pipeline for numeric and categorical features.
    
    Args:
        numeric_cols: List of numeric column names
        categorical_cols: List of categorical column names
        numeric_strategy: Strategy for imputing missing numeric values ('mean', 'median', 'constant')
        categorical_strategy: Strategy for imputing missing categorical values ('most_frequent', 'constant')
        scaling: Type of scaling to apply to numeric features ('standard', 'minmax', 'robust', or None)
        
    Returns:
        ColumnTransformer with preprocessing steps
    """
    # Define numeric transformers
    numeric_transformer_steps = [
        ('imputer', SimpleImputer(strategy=numeric_strategy))
    ]
    
    # Add scaling if specified
    if scaling == 'standard':
        numeric_transformer_steps.append(('scaler', StandardScaler()))
    elif scaling == 'minmax':
        numeric_transformer_steps.append(('scaler', MinMaxScaler()))
    elif scaling == 'robust':
        numeric_transformer_steps.append(('scaler', RobustScaler()))
    
    numeric_transformer = Pipeline(numeric_transformer_steps)
    
    # Define categorical transformer
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy=categorical_strategy, fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Create column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='drop'  # Drop other columns
    )
    
    return preprocessor

def convert_data_types(
    df: pd.DataFrame,
    int_cols: Optional[List[str]] = None,
    float_cols: Optional[List[str]] = None,
    category_cols: Optional[List[str]] = None,
    datetime_cols: Optional[Dict[str, str]] = None,
    bool_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Convert columns to specified data types.
    
    Args:
        df: Input DataFrame
        int_cols: Columns to convert to int
        float_cols: Columns to convert to float
        category_cols: Columns to convert to category
        datetime_cols: Dictionary of {column: format} for datetime conversion
        bool_cols: Columns to convert to boolean
        
    Returns:
        DataFrame with converted data types
    """
    df = df.copy()
    
    if int_cols:
        for col in int_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
    
    if float_cols:
        for col in float_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('float32')
    
    if category_cols:
        for col in category_cols:
            if col in df.columns:
                df[col] = df[col].astype('category')
    
    if datetime_cols:
        for col, fmt in datetime_cols.items():
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], format=fmt, errors='coerce')
    
    if bool_cols:
        for col in bool_cols:
            if col in df.columns:
                df[col] = df[col].astype('bool')
    
    return df

def handle_outliers(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = 'clip',
    threshold: float = 3.0
) -> pd.DataFrame:
    """
    Handle outliers in numeric columns using specified method.
    
    Args:
        df: Input DataFrame
        columns: List of columns to process (None for all numeric columns)
        method: Method to handle outliers ('clip', 'remove', or 'winsorize')
        threshold: Number of standard deviations to use for outlier detection
        
    Returns:
        DataFrame with outliers handled
    """
    df = df.copy()
    
    if columns is None:
        numeric_cols = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns
    else:
        numeric_cols = [col for col in columns if col in df.columns and df[col].dtype in ['int64', 'float64', 'int32', 'float32']]
    
    for col in numeric_cols:
        if method == 'clip':
            # Clip values to threshold standard deviations from the mean
            mean = df[col].mean()
            std = df[col].std()
            lower_bound = mean - threshold * std
            upper_bound = mean + threshold * std
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
            
        elif method == 'remove':
            # Remove rows with outliers
            z_scores = (df[col] - df[col].mean()) / df[col].std()
            df = df[abs(z_scores) <= threshold]
            
        elif method == 'winsorize':
            # Winsorize (cap) the outliers
            lower = df[col].quantile(0.01)
            upper = df[col].quantile(0.99)
            df[col] = df[col].clip(lower=lower, upper=upper)
    
    return df

def create_time_based_features(
    df: pd.DataFrame,
    datetime_col: str,
    prefix: Optional[str] = None
) -> pd.DataFrame:
    """
    Create time-based features from a datetime column.
    
    Args:
        df: Input DataFrame
        datetime_col: Name of the datetime column
        prefix: Prefix to use for the new column names (default: datetime_col)
        
    Returns:
        DataFrame with added time-based features
    """
    if prefix is None:
        prefix = datetime_col
    
    df = df.copy()
    
    # Ensure the column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
        df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce')
    
    # Extract time-based features
    df[f'{prefix}_year'] = df[datetime_col].dt.year
    df[f'{prefix}_month'] = df[datetime_col].dt.month
    df[f'{prefix}_day'] = df[datetime_col].dt.day
    df[f'{prefix}_hour'] = df[datetime_col].dt.hour
    df[f'{prefix}_minute'] = df[datetime_col].dt.minute
    df[f'{prefix}_dayofweek'] = df[datetime_col].dt.dayofweek
    df[f'{prefix}_dayofyear'] = df[datetime_col].dt.dayofyear
    df[f'{prefix}_weekofyear'] = df[datetime_col].dt.isocalendar().week
    df[f'{prefix}_is_weekend'] = df[datetime_col].dt.dayofweek.isin([5, 6]).astype(int)
    
    # Add cyclical encoding for periodic features
    for col in [f'{prefix}_month', f'{prefix}_day', f'{prefix}_hour', f'{prefix}_minute', 
                f'{prefix}_dayofweek', f'{prefix}_dayofyear', f'{prefix}_weekofyear']:
        if col in df.columns:
            max_val = df[col].max()
            df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / max_val)
            df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / max_val)
    
    return df

def calculate_feature_importance(
    model: Any,
    feature_names: List[str],
    importance_type: str = 'auto'
) -> pd.DataFrame:
    """
    Calculate and return feature importances from a trained model.
    
    Args:
        model: Trained model with feature_importances_ or coef_ attribute
        feature_names: List of feature names
        importance_type: Type of importance to calculate ('auto', 'permutation', 'shap')
        
    Returns:
        DataFrame with feature names and their importance scores
    """
    if hasattr(model, 'feature_importances_'):
        # Tree-based models
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # Linear models
        if len(model.coef_.shape) == 1:
            importances = np.abs(model.coef_)
        else:
            # Multi-class classification
            importances = np.sum(np.abs(model.coef_), axis=0)
    else:
        raise ValueError("Model does not have feature_importances_ or coef_ attribute")
    
    # Create DataFrame with feature importances
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    
    # Sort by importance
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    return feature_importance
