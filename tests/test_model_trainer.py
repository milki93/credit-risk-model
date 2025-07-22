"""
Test script for the ModelTrainer class.
"""
import os
import sys
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.model_trainer import ModelTrainer

@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    # Generate synthetic data
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=5,
        n_redundant=2,
        n_clusters_per_class=1,
        weights=[0.9, 0.1],  # Imbalanced classes
        random_state=42
    )
    
    # Convert to DataFrame with mixed feature types
    numeric_cols = [f'numeric_{i+1}' for i in range(n_features - 2)]
    cat_cols = ['category_1', 'category_2']
    
    X = pd.DataFrame(
        X[:, :-2],  # All but last two columns are numeric
        columns=numeric_cols
    )
    
    # Add categorical columns
    X[cat_cols[0]] = np.random.choice(['A', 'B', 'C'], size=n_samples)
    X[cat_cols[1]] = np.random.choice(['X', 'Y'], size=n_samples)
    
    # Add some missing values
    for col in X.columns:
        mask = np.random.random(n_samples) < 0.1  # 10% missing
        X.loc[mask, col] = np.nan
    
    return X, pd.Series(y)

def test_model_trainer_initialization():
    """Test ModelTrainer initialization with different model types."""
    # Test with different model types
    for model_type in ['random_forest', 'gradient_boosting', 'logistic', 'stacking']:
        trainer = ModelTrainer(model_type=model_type, random_state=42)
        assert trainer.model_type == model_type
        assert trainer.random_state == 42
        assert trainer.cv_folds == 5
        assert trainer.scoring == 'roc_auc'

def test_preprocessing_pipeline(sample_data):
    """Test the preprocessing pipeline creation."""
    X, _ = sample_data
    numeric_cols = [col for col in X.columns if col.startswith('numeric')]
    cat_cols = [col for col in X.columns if col.startswith('category')]
    
    # Initialize trainer
    trainer = ModelTrainer(model_type='random_forest', random_state=42)
    
    # Create preprocessing pipeline
    trainer.create_preprocessing_pipeline(numeric_cols, cat_cols)
    
    # Check if preprocessor is created
    assert trainer.preprocessor is not None
    assert hasattr(trainer.preprocessor, 'transform')
    
    # Check if feature names are stored
    assert hasattr(trainer, 'numeric_features_')
    assert hasattr(trainer, 'categorical_features_')
    assert set(numeric_cols) == set(trainer.numeric_features_)
    assert set(cat_cols) == set(trainer.categorical_features_)

def test_model_training(sample_data):
    """Test model training and evaluation."""
    X, y = sample_data
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Initialize and configure trainer with a simpler model for testing
    trainer = ModelTrainer(
        model_type='logistic',  # Use logistic regression for faster testing
        random_state=42,
        cv_folds=2,  # Fewer folds for faster testing
        n_jobs=1,  # Avoid potential issues with parallel processing in tests
    )
    
    # Set up preprocessing
    numeric_cols = [col for col in X.columns if col.startswith('numeric')]
    cat_cols = [col for col in X.columns if col.startswith('category')]
    
    # Create preprocessing pipeline
    trainer.create_preprocessing_pipeline(numeric_cols, cat_cols)
    
    # Test training with no parameter search first
    results = trainer.fit(
        X_train, y_train,
        param_search='none',  # Skip parameter search for this test
        refit=True
    )
    
    # Check results
    assert 'model' in results
    assert 'best_params' in results
    assert 'training_metrics' in results
    assert 'feature_importances' in results
    
    # Test evaluation
    test_metrics = trainer.evaluate(X_test, y_test, set_name='test')
    assert 'test_accuracy' in test_metrics
    assert 'test_precision' in test_metrics
    assert 'test_recall' in test_metrics
    assert 'test_f1' in test_metrics
    
    # Test feature importances
    feature_importances = trainer.get_feature_importances()
    assert not feature_importances.empty
    assert 'feature' in feature_importances.columns
    assert 'importance' in feature_importances.columns

def test_model_saving_loading(tmp_path, sample_data):
    """Test model saving and loading."""
    X, y = sample_data
    
    # Use a small subset for faster testing
    X = X.head(100)
    y = y.head(100)
    
    # Initialize and train a model
    trainer = ModelTrainer(
        model_type='logistic',  # Use logistic regression for faster testing
        random_state=42,
        n_jobs=1
    )
    
    # Identify column types
    numeric_cols = [col for col in X.columns if col.startswith('numeric')]
    cat_cols = [col for col in X.columns if col.startswith('category')]
    
    # Create preprocessing pipeline
    trainer.create_preprocessing_pipeline(numeric_cols, cat_cols)
    
    # Fit with DataFrame
    trainer.fit(X, y, param_search='none')
    
    # Save the model
    model_path = os.path.join(tmp_path, 'test_model.joblib')
    trainer.save_model(model_path)
    assert os.path.exists(model_path)
    
    # Load the model
    loaded_trainer = ModelTrainer.load_model(model_path)
    assert hasattr(loaded_trainer, 'model')
    assert hasattr(loaded_trainer.model, 'predict')
    
    # Test prediction with DataFrame input
    y_pred = loaded_trainer.model.predict(X.head())
    assert len(y_pred) == 5
    assert y_pred.dtype in (int, np.int64, np.int32)
    assert set(y_pred).issubset({0, 1})  # Binary classification

if __name__ == "__main__":
    pytest.main([__file__, '-v'])
