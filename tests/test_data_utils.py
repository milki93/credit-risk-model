"""
Tests for data utility functions.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.utils.data_utils import (
    split_data,
    get_numeric_categorical_columns,
    create_preprocessing_pipeline,
    convert_data_types,
    handle_outliers,
    create_time_based_features,
    calculate_feature_importance
)

@pytest.fixture
def sample_data():
    """Create a sample DataFrame for testing."""
    np.random.seed(42)
    n_samples = 1000
    
    # Create dates for testing time-based features
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_samples)]
    
    data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(10, 5, n_samples),
        'category1': np.random.choice(['A', 'B', 'C'], size=n_samples),
        'category2': np.random.choice(['X', 'Y'], size=n_samples),
        'date_col': dates,
        'int_col': np.random.randint(1, 100, size=n_samples),
        'float_col': np.random.uniform(0, 1, size=n_samples),
        'target': np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])
    })
    
    # Add some missing values
    for col in data.columns:
        if col not in ['target', 'date_col']:
            mask = np.random.random(n_samples) < 0.1  # 10% missing values
            data.loc[mask, col] = np.nan
    
    return data

# Test split_data function
def test_split_data_basic(sample_data):
    """Test basic functionality of split_data."""
    X_train, X_test, y_train, y_test = split_data(
        sample_data, 
        target_col='target',
        test_size=0.2,
        random_state=42
    )
    
    # Check shapes
    assert len(X_train) + len(X_test) == len(sample_data)
    assert len(y_train) + len(y_test) == len(sample_data)
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    
    # Check if test size is approximately correct
    assert abs(len(X_test) / len(sample_data) - 0.2) < 0.01

def test_split_data_stratification(sample_data):
    """Test if stratification works correctly."""
    # Get class distribution in original data
    original_dist = sample_data['target'].value_counts(normalize=True)
    
    # Split with stratification
    _, _, y_train, y_test = split_data(
        sample_data,
        target_col='target',
        test_size=0.2,
        stratify=True,
        random_state=42
    )
    
    # Check if class distribution is preserved
    train_dist = y_train.value_counts(normalize=True)
    test_dist = y_test.value_counts(normalize=True)
    
    assert abs(original_dist[0] - train_dist[0]) < 0.05
    assert abs(original_dist[1] - train_dist[1]) < 0.05
    assert abs(original_dist[0] - test_dist[0]) < 0.05
    assert abs(original_dist[1] - test_dist[1]) < 0.05

def test_split_data_no_stratification(sample_data):
    """Test splitting without stratification."""
    # Split without stratification
    _, _, y_train, y_test = split_data(
        sample_data,
        target_col='target',
        test_size=0.2,
        stratify=False,
        random_state=42
    )
    
    # Just verify the function runs without errors
    assert len(y_train) > 0
    assert len(y_test) > 0

# Test get_numeric_categorical_columns function
def test_get_numeric_categorical_columns(sample_data):
    """Test identifying numeric and categorical columns."""
    numeric_cols, categorical_cols = get_numeric_categorical_columns(
        sample_data, 
        exclude_cols=['target']
    )
    
    # Check if columns are correctly identified
    expected_numeric = ['feature1', 'feature2', 'int_col', 'float_col']
    expected_categorical = ['category1', 'category2']
    
    assert set(numeric_cols) == set(expected_numeric)
    assert set(categorical_cols) == set(expected_categorical)
    
    # Test with no exclusions
    numeric_cols_all, categorical_cols_all = get_numeric_categorical_columns(sample_data)
    assert 'target' in numeric_cols_all  # target is numeric (0/1)

# Test create_preprocessing_pipeline function
def test_create_preprocessing_pipeline():
    """Test creating a preprocessing pipeline."""
    numeric_cols = ['feature1', 'feature2']
    categorical_cols = ['category1', 'category2']
    
    # Test with standard scaling
    preprocessor = create_preprocessing_pipeline(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        numeric_strategy='median',
        categorical_strategy='most_frequent',
        scaling='standard'
    )
    
    assert preprocessor is not None
    assert hasattr(preprocessor, 'fit_transform')
    assert len(preprocessor.transformers) == 2  # num and cat transformers
    
    # Test with no scaling
    preprocessor_no_scale = create_preprocessing_pipeline(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        scaling=None
    )
    assert len(preprocessor_no_scale.transformers) == 2

# Test convert_data_types function
def test_convert_data_types(sample_data):
    """Test converting data types."""
    # Make a copy to avoid modifying the original
    df = sample_data.copy()
    
    # Convert data types
    converted = convert_data_types(
        df,
        int_cols=['int_col'],
        float_cols=['float_col'],
        category_cols=['category1', 'category2'],
        datetime_cols={'date_col': None},  # Infer format
        bool_cols=[]
    )
    
    # Check data types
    assert pd.api.types.is_integer_dtype(converted['int_col'].dtype)
    assert pd.api.types.is_float_dtype(converted['float_col'].dtype)
    assert pd.api.types.is_categorical_dtype(converted['category1'].dtype)
    assert pd.api.types.is_categorical_dtype(converted['category2'].dtype)
    assert pd.api.types.is_datetime64_any_dtype(converted['date_col'].dtype)

# Test handle_outliers function
def test_handle_outliers():
    """Test handling outliers."""
    # Create a DataFrame with outliers
    data = pd.DataFrame({
        'normal': np.random.normal(0, 1, 1000),
        'with_outliers': np.concatenate([
            np.random.normal(0, 1, 950),  # Normal data
            np.random.normal(20, 1, 50)    # Outliers
        ])
    })
    
    # Clip outliers
    clipped = handle_outliers(data, method='clip', threshold=3.0)
    assert (clipped['with_outliers'] <= data['with_outliers'].quantile(0.99) * 1.5).all()
    
    # Remove outliers
    removed = handle_outliers(data, method='remove', threshold=3.0)
    assert len(removed) < len(data)
    
    # Winsorize
    winsorized = handle_outliers(data, method='winsorize')
    assert winsorized['with_outliers'].max() <= data['with_outliers'].quantile(0.99)

# Test create_time_based_features function
def test_create_time_based_features():
    """Test creating time-based features."""
    # Create a DataFrame with datetime column
    dates = pd.date_range('2023-01-01', periods=365, freq='D')
    df = pd.DataFrame({'date': dates})
    
    # Create time-based features
    df_with_time_features = create_time_based_features(df, 'date', 'test')
    
    # Check if features were created
    expected_columns = [
        'test_year', 'test_month', 'test_day', 'test_hour', 
        'test_minute', 'test_dayofweek', 'test_dayofyear',
        'test_weekofyear', 'test_is_weekend'
    ]
    
    for col in expected_columns:
        assert col in df_with_time_features.columns
    
    # Check cyclical encoding
    assert 'test_month_sin' in df_with_time_features.columns
    assert 'test_month_cos' in df_with_time_features.columns

# Test calculate_feature_importance function
def test_calculate_feature_importance():
    """Test calculating feature importance."""
    # Create sample data
    X = np.random.rand(100, 5)
    y = (X[:, 0] + X[:, 1] * 2 + np.random.normal(0, 0.1, 100)) > 0.5
    feature_names = [f'feature_{i}' for i in range(5)]
    
    # Test with RandomForest
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf.fit(X, y)
    
    rf_importance = calculate_feature_importance(rf, feature_names)
    assert len(rf_importance) == 5
    assert 'feature' in rf_importance.columns
    assert 'importance' in rf_importance.columns
    
    # Test with LogisticRegression
    lr = LogisticRegression(random_state=42)
    lr.fit(X, y)
    
    lr_importance = calculate_feature_importance(lr, feature_names)
    assert len(lr_importance) == 5
    assert 'feature' in lr_importance.columns
    assert 'importance' in lr_importance.columns
    
    # Test with invalid model
    class DummyModel:
        pass
    
    dummy = DummyModel()
    with pytest.raises(ValueError, match="does not have feature_importances_ or coef_ attribute"):
        calculate_feature_importance(dummy, feature_names)
