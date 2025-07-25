"""
Feature engineering utilities for credit risk modeling.
"""
from typing import Tuple, Union, List, Optional
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


def create_feature_pipeline(
    numeric_features: List[str],
    categorical_features: List[str]
) -> Pipeline:
    """
    Create a feature engineering pipeline.
    
    Args:
        numeric_features: List of numeric feature names
        categorical_features: List of categorical feature names
        
    Returns:
        A scikit-learn Pipeline for feature engineering
    """
    # Numeric transformations
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler(with_mean=False))
    ])
    
    # Categorical transformations
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    return preprocessor


def create_target_variable(
    data: pd.DataFrame,
    target_column: str = 'default',
    positive_class: str = 'yes'
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Create feature matrix X and target vector y from input data.
    
    Args:
        data: Input DataFrame
        target_column: Name of the target column
        positive_class: Value indicating positive class
        
    Returns:
        Tuple of (X, y) where X is the feature matrix and y is the target vector
    """
    # Create a copy to avoid modifying the original data
    df = data.copy()
    
    # Create binary target
    y = (df[target_column] == positive_class).astype(int)
    X = df.drop(columns=[target_column])
    
    return X, y
