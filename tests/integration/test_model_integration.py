"""Integration tests for the credit risk model training pipeline."""
import os
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.models.model_trainer import ModelTrainer
from sklearn.model_selection import train_test_split

# Skip these tests if data files are not available
DATA_PATH = Path("data/processed/processed_transactions.parquet")
pytestmark = pytest.mark.skipif(
    not DATA_PATH.exists(),
    reason=f"Test data not found at {DATA_PATH}. Run data processing pipeline first."
)

# Constants
TARGET_COL = 'FraudResult'
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_test_data():
    """Load and prepare test data for integration testing."""
    # Load processed data
    df = pd.read_parquet(DATA_PATH)
    
    # Select features and target
    numeric_cols = [col for col in df.select_dtypes(include=np.number).columns 
                   if col != TARGET_COL]
    cat_cols = [col for col in df.select_dtypes(include=['object', 'category']).columns 
               if col not in ['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId']
               and col != TARGET_COL]
    
    # Drop rows with missing target
    df = df.dropna(subset=[TARGET_COL])
    
    # Convert target to binary (0/1)
    y = (df[TARGET_COL] > 0).astype(int)
    X = df[numeric_cols + cat_cols]
    
    return X, y


def test_end_to_end_model_training():
    """Test the complete model training pipeline with real data."""
    # Load data
    X, y = load_test_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    # Initialize model trainer
    trainer = ModelTrainer(
        model_type='random_forest',
        random_state=RANDOM_STATE,
        cv_folds=3,
        n_jobs=-1
    )
    
    # Set up preprocessing
    numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    trainer.create_preprocessing_pipeline(numeric_cols, cat_cols)
    
    # Train model with basic parameters (no tuning for speed)
    results = trainer.fit(
        X_train, y_train,
        param_search='none',
        refit=True
    )
    
    # Verify training results
    assert 'model' in results
    assert 'training_metrics' in results
    assert 'feature_importances' in results
    
    # Evaluate on test set
    test_metrics = trainer.evaluate(X_test, y_test, set_name='test')
    
    # Basic sanity checks on metrics
    assert 'test_accuracy' in test_metrics
    assert 0 <= test_metrics['test_accuracy'] <= 1
    assert 'test_precision' in test_metrics
    assert 0 <= test_metrics['test_precision'] <= 1
    assert 'test_recall' in test_metrics
    assert 0 <= test_metrics['test_recall'] <= 1
    
    # Verify feature importances
    importances = results['feature_importances']
    # Feature importances can be a DataFrame or dict depending on the model
    if hasattr(importances, 'to_dict'):  # Handle DataFrame
        assert len(importances) > 0
        assert 'feature' in importances.columns
        assert 'importance' in importances.columns
    else:  # Handle dict
        assert isinstance(importances, dict)
        assert len(importances) > 0
        assert all(isinstance(k, str) for k in importances.keys())
        assert all(isinstance(v, (int, float)) for v in importances.values())


def test_model_saving_loading(tmp_path):
    """Test saving and loading the trained model."""
    # Load a small subset of data for faster testing
    X, y = load_test_data()
    X = X.head(1000)
    y = y.head(1000)
    
    # Initialize and train model
    trainer = ModelTrainer(
        model_type='logistic',
        random_state=RANDOM_STATE,
        n_jobs=1
    )
    
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    trainer.create_preprocessing_pipeline(numeric_cols, cat_cols)
    
    trainer.fit(X, y, param_search='none')
    
    # Save model
    model_path = tmp_path / "test_model.joblib"
    trainer.save_model(model_path)
    assert model_path.exists()
    
    # Load model
    loaded_trainer = ModelTrainer.load_model(model_path)
    assert hasattr(loaded_trainer, 'model')
    assert hasattr(loaded_trainer.model, 'predict')
    
    # Test prediction
    y_pred = loaded_trainer.model.predict(X.head())
    assert len(y_pred) == 5
    assert set(y_pred).issubset({0, 1})  # Binary classification


@patch('src.models.model_trainer.logger')
def test_model_training_with_missing_data(mock_logger):
    """Test model training with missing data handling."""
    X, y = load_test_data()
    
    # Add some missing values to test imputation
    X_modified = X.copy()
    numeric_cols = X_modified.select_dtypes(include=np.number).columns
    if len(numeric_cols) > 0:
        X_modified.loc[X_modified.sample(frac=0.1, random_state=RANDOM_STATE).index, 
                      numeric_cols[0]] = np.nan
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_modified, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    # Initialize and train model
    trainer = ModelTrainer(
        model_type='random_forest',
        random_state=RANDOM_STATE,
        n_jobs=1
    )
    
    numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    trainer.create_preprocessing_pipeline(numeric_cols, cat_cols)
    
    # This should handle missing values without errors
    results = trainer.fit(X_train, y_train, param_search='none')
    
    # Verify model was trained successfully
    assert 'model' in results
    assert 'training_metrics' in results
    
    # Test prediction with missing values
    X_test_modified = X_test.copy()
    if len(numeric_cols) > 0:
        X_test_modified[numeric_cols[0]] = np.nan
    
    # Should not raise an error
    y_pred = trainer.model.predict(X_test_modified)
    assert len(y_pred) == len(X_test_modified)
