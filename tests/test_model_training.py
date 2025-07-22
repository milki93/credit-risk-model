"""
Unit tests for model training functionality.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Import the functions to test
from src.model_training import (
    evaluate_model,
    train_random_forest,
    train_gradient_boosting,
    train_xgboost,
    train_lightgbm,
    train_models,
    register_best_model
)

# Set random seed for reproducibility
np.random.seed(42)

@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        weights=[0.8, 0.2],
        random_state=42
    )
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    y = pd.Series(y, name='target')
    return X, y

def test_evaluate_model(sample_data):
    """Test the evaluate_model function."""
    X, y = sample_data
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    # Mock MLflow to avoid actual logging during tests
    with patch('mlflow.start_run'), \
         patch('mlflow.log_params'), \
         patch('mlflow.log_metrics'), \
         patch('mlflow.sklearn.log_model'), \
         patch('mlflow.log_dict'), \
         patch('mlflow.log_figure'):
        
        metrics = evaluate_model(
            model=model,
            X_test=X,
            y_test=y,
            model_name='test_model',
            run_name='test_run'
        )
    
    # Check that all expected metrics are returned
    expected_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'average_precision']
    for metric in expected_metrics:
        assert metric in metrics
        assert isinstance(metrics[metric], float)

def test_train_random_forest(sample_data):
    """Test the train_random_forest function."""
    X, y = sample_data
    
    # Test with grid search
    model, params = train_random_forest(
        X_train=X,
        y_train=y,
        use_grid_search=True,
        n_iter=2
    )
    
    assert isinstance(model, RandomForestClassifier)
    assert 'n_estimators' in params
    assert 'max_depth' in params
    
    # Test without grid search
    model, params = train_random_forest(
        X_train=X,
        y_train=y,
        use_grid_search=False
    )
    
    assert isinstance(model, RandomForestClassifier)
    assert model.n_estimators == 200

def test_train_gradient_boosting(sample_data):
    """Test the train_gradient_boosting function."""
    X, y = sample_data
    
    model, params = train_gradient_boosting(
        X_train=X,
        y_train=y,
        use_grid_search=False
    )
    
    assert isinstance(model, GradientBoostingClassifier)
    assert model.n_estimators == 200
    assert model.learning_rate == 0.1

def test_train_xgboost(sample_data):
    """Test the train_xgboost function."""
    X, y = sample_data
    
    model, params = train_xgboost(
        X_train=X,
        y_train=y,
        use_grid_search=False
    )
    
    assert isinstance(model, XGBClassifier)
    assert model.n_estimators == 200
    assert model.learning_rate == 0.1

def test_train_lightgbm(sample_data):
    """Test the train_lightgbm function."""
    X, y = sample_data
    
    model, params = train_lightgbm(
        X_train=X,
        y_train=y,
        use_grid_search=False
    )
    
    assert isinstance(model, LGBMClassifier)
    assert model.n_estimators == 200
    assert model.learning_rate == 0.1

def test_train_models(sample_data):
    """Test the train_models function."""
    X, y = sample_data
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]
    
    # Mock MLflow and model training to speed up tests
    with patch('mlflow.set_tracking_uri'), \
         patch('mlflow.set_experiment'), \
         patch('mlflow.start_run'), \
         patch('mlflow.log_metric'), \
         patch('mlflow.sklearn.log_model'), \
         patch('src.model_training.train_random_forest') as mock_train_rf, \
         patch('src.model_training.train_gradient_boosting') as mock_train_gb, \
         patch('src.model_training.train_xgboost') as mock_train_xgb, \
         patch('src.model_training.train_lightgbm') as mock_train_lgbm:
        
        # Set up mock return values
        mock_model = MagicMock()
        mock_model.predict.return_value = np.random.randint(0, 2, size=20)
        mock_model.predict_proba.return_value = np.column_stack([
            np.random.rand(20), np.random.rand(20)
        ])
        mock_params = {'param1': 'value1', 'param2': 'value2'}
        
        mock_train_rf.return_value = (mock_model, mock_params)
        mock_train_gb.return_value = (mock_model, mock_params)
        mock_train_xgb.return_value = (mock_model, mock_params)
        mock_train_lgbm.return_value = (mock_model, mock_params)
        
        # Call the function
        results = train_models(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            models=['random_forest', 'gradient_boosting'],
            use_grid_search=False,
            n_iter=1
        )
    
    # Check that results contain expected models
    assert 'random_forest' in results
    assert 'gradient_boosting' in results
    assert 'xgboost' not in results  # Not included in this test
    
    # Check that each model has the expected keys
    for model_name in results:
        assert 'model' in results[model_name]
        assert 'metrics' in results[model_name]
        assert 'params' in results[model_name]

def test_register_best_model():
    """Test the register_best_model function."""
    # Create mock results
    results = {
        'model1': {
            'model': MagicMock(),
            'metrics': {'roc_auc': 0.85, 'accuracy': 0.8},
            'params': {'param1': 'value1'}
        },
        'model2': {
            'model': MagicMock(),
            'metrics': {'roc_auc': 0.9, 'accuracy': 0.75},
            'params': {'param2': 'value2'}
        }
    }
    
    # Mock MLflow
    with patch('mlflow.start_run') as mock_start_run, \
         patch('mlflow.log_metric') as mock_log_metric, \
         patch('mlflow.log_params') as mock_log_params, \
         patch('mlflow.sklearn.log_model') as mock_log_model:
        
        # Call the function
        register_best_model(
            results=results,
            metric='roc_auc',
            model_name='test_model'
        )
        
        # Check that MLflow was called with the correct parameters
        assert mock_start_run.called
        mock_log_metric.assert_called_with('best_roc_auc', 0.9)  # model2 has higher ROC AUC
        mock_log_params.assert_called_with({'param2': 'value2'})
        mock_log_model.assert_called()
