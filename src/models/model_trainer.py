"""
Model training and evaluation pipeline for credit risk prediction.
"""
import os
import logging
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime

# Model imports
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    cross_val_score,
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)

# Feature processing
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Type hints
from pandas import DataFrame, Series
from numpy.typing import ArrayLike

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    A class to handle model training, evaluation, and persistence for credit risk prediction.
    """
    
    def __init__(
        self,
        model_type: str = 'random_forest',
        random_state: int = 42,
        cv_folds: int = 5,
        scoring: str = 'roc_auc',
        n_jobs: int = -1,
        verbose: int = 1
    ) -> None:
        """
        Initialize the ModelTrainer.
        
        Args:
            model_type: Type of model to train ('random_forest', 'gradient_boosting', 'logistic', 'stacking')
            random_state: Random seed for reproducibility
            cv_folds: Number of cross-validation folds
            scoring: Scoring metric for model evaluation
            n_jobs: Number of CPU cores to use (-1 for all available)
            verbose: Verbosity level
        """
        self.model_type = model_type
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        # Initialize model and parameters
        self.model = None
        self.best_params_ = None
        self.feature_importances_ = None
        self.feature_names_ = None
        self.classes_ = None
        
        # Initialize preprocessing pipeline
        self.preprocessor = None
        
        # Set up model and parameter grid based on model type
        self._setup_model()
    
    def _setup_model(self) -> None:
        """Set up the model and parameter grid based on model type."""
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(random_state=self.random_state)
            self.param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'class_weight': ['balanced', 'balanced_subsample', None]
            }
            
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(random_state=self.random_state)
            self.param_grid = {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'subsample': [0.8, 1.0]
            }
            
        elif self.model_type == 'logistic':
            self.model = LogisticRegression(
                random_state=self.random_state,
                class_weight='balanced',
                max_iter=1000,
                n_jobs=self.n_jobs
            )
            self.param_grid = {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
            
        elif self.model_type == 'stacking':
            # Base models
            rf = RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=self.n_jobs
            )
            gb = GradientBoostingClassifier(
                n_estimators=100,
                random_state=self.random_state
            )
            
            # Meta model
            meta_model = LogisticRegression(
                random_state=self.random_state,
                class_weight='balanced'
            )
            
            self.model = StackingClassifier(
                estimators=[
                    ('rf', rf),
                    ('gb', gb)
                ],
                final_estimator=meta_model,
                n_jobs=self.n_jobs
            )
            
            # Simplified parameter grid for stacking
            self.param_grid = {
                'rf__n_estimators': [100, 200],
                'rf__max_depth': [None, 10],
                'gb__n_estimators': [100, 200],
                'gb__learning_rate': [0.01, 0.1]
            }
            
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def create_preprocessing_pipeline(
        self,
        numeric_features: List[str],
        categorical_features: List[str]
    ) -> None:
        """
        Create a preprocessing pipeline for the model.
        
        Args:
            numeric_features: List of numeric feature names
            categorical_features: List of categorical feature names
        """
        # Numeric transformations
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Categorical transformations
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Combine preprocessing steps
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='drop'  # Drop columns that are not explicitly transformed
        )
        
        # Store feature names for later use
        self.numeric_features_ = numeric_features
        self.categorical_features_ = categorical_features
    
    def fit(
        self,
        X_train: Union[DataFrame, ArrayLike],
        y_train: Union[Series, ArrayLike],
        param_search: str = 'random',
        n_iter: int = 20,
        refit: bool = True
    ) -> Dict[str, Any]:
        """
        Train the model with optional hyperparameter tuning.
        
        Args:
            X_train: Training features (DataFrame or numpy array)
            y_train: Training target (Series or numpy array)
            param_search: Type of hyperparameter search ('grid', 'random', or 'none')
            n_iter: Number of parameter settings to sample (for random search)
            refit: Whether to refit the best model on the full training set
            
        Returns:
            Dictionary containing training results and model information
        """
        start_time = datetime.now()
        
        # Store whether input is a DataFrame
        self._is_dataframe = isinstance(X_train, DataFrame)
        
        # Convert to numpy arrays if not already
        if self._is_dataframe:
            self.feature_names_ = X_train.columns.tolist()
            X_train_values = X_train.values
        else:
            X_train_values = X_train
            
        if isinstance(y_train, (pd.Series, pd.DataFrame)):
            y_train = y_train.values
        
        # Store class labels
        self.classes_ = np.unique(y_train)
        
        # Create model pipeline with appropriate input handling
        if self._is_dataframe:
            model_pipeline = Pipeline([
                ('preprocessor', self.preprocessor),
                ('model', self.model)
            ])
        else:
            # If input is numpy array, create a simpler pipeline without column names
            numeric_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            
            categorical_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            
            # Get indices of numeric and categorical columns
            numeric_indices = [i for i, col in enumerate(self.feature_names_) 
                             if col.startswith('numeric')]
            categorical_indices = [i for i, col in enumerate(self.feature_names_)
                                 if col.startswith('category')]
            
            preprocessor = ColumnTransformer([
                ('num', numeric_transformer, numeric_indices),
                ('cat', categorical_transformer, categorical_indices)
            ])
            
            model_pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', self.model)
            ])
        
        # Hyperparameter tuning
        if param_search == 'grid':
            search = GridSearchCV(
                estimator=model_pipeline,
                param_grid=self.param_grid,
                cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state),
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                refit=refit
            )
        elif param_search == 'random':
            search = RandomizedSearchCV(
                estimator=model_pipeline,
                param_distributions=self.param_grid,
                n_iter=n_iter,
                cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state),
                scoring=self.scoring,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                refit=refit
            )
        else:  # No parameter search
            search = model_pipeline
        
        # Fit the model
        logger.info(f"Training {self.model_type} model...")
        search.fit(X_train, y_train)
        
        # Store the best model and parameters
        if param_search in ['grid', 'random']:
            self.model = search.best_estimator_
            self.best_params_ = search.best_params_
            cv_results = search.cv_results_
        else:
            self.model = search
            self.best_params_ = None
            cv_results = {}
        
        # Extract feature importances if available
        self._extract_feature_importances()
        
        # Generate predictions on training set
        y_pred = self.model.predict(X_train)
        y_proba = self.model.predict_proba(X_train)[:, 1] if hasattr(self.model, 'predict_proba') else None
        
        # Calculate training metrics
        train_metrics = self._calculate_metrics(y_train, y_pred, y_proba, prefix='train_')
        
        # Log training summary
        training_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Prepare results
        results = {
            'model': self.model,
            'best_params': self.best_params_,
            'cv_results': cv_results,
            'training_metrics': train_metrics,
            'training_time_seconds': training_time,
            'feature_importances': self.feature_importances_,
            'feature_names': self.feature_names_
        }
        
        return results
    
    def _extract_feature_importances(self) -> None:
        """Extract and store feature importances from the trained model."""
        self.feature_importances_ = None
        self.feature_names_ = None
        
        try:
            # Get the model from the pipeline
            if hasattr(self.model, 'named_steps'):
                model = self.model.named_steps['model']
                preprocessor = self.model.named_steps['preprocessor']
                
                # Get feature names after one-hot encoding
                if hasattr(preprocessor, 'get_feature_names_out'):
                    self.feature_names_ = preprocessor.get_feature_names_out()
                
                # Get feature importances based on model type
                if hasattr(model, 'feature_importances_'):
                    # Tree-based models
                    self.feature_importances_ = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    # Linear models
                    if len(model.coef_.shape) == 1:
                        self.feature_importances_ = np.abs(model.coef_)
                    else:
                        # Multi-class classification
                        self.feature_importances_ = np.sum(np.abs(model.coef_), axis=0)
                
                # For stacking classifier, use the meta-model's coefficients
                elif hasattr(model, 'final_estimator_') and hasattr(model.final_estimator_, 'coef_'):
                    self.feature_importances_ = np.abs(model.final_estimator_.coef_[0])
            
            # Create a DataFrame with feature importances
            if self.feature_importances_ is not None and self.feature_names_ is not None:
                self.feature_importances_ = pd.DataFrame({
                    'feature': self.feature_names_,
                    'importance': self.feature_importances_
                }).sort_values('importance', ascending=False)
                
        except Exception as e:
            logger.warning(f"Could not extract feature importances: {str(e)}")
    
    def evaluate(
        self,
        X: Union[DataFrame, ArrayLike],
        y_true: Union[Series, ArrayLike],
        set_name: str = 'test'
    ) -> Dict[str, float]:
        """
        Evaluate the model on the given dataset.
        
        Args:
            X: Input features (DataFrame or numpy array)
            y_true: True labels
            set_name: Name of the dataset (for logging)
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Ensure y_true is a numpy array
        if isinstance(y_true, (pd.Series, pd.DataFrame)):
            y_true = y_true.values
            
        # For prediction, we can use the model directly since it was trained with the pipeline
        # This maintains the DataFrame structure if that's what was used during training
        if hasattr(self, 'model') and self.model is not None:
            y_pred = self.model.predict(X)
            y_proba = self.model.predict_proba(X)[:, 1] if hasattr(self.model, 'predict_proba') else None
        else:
            raise ValueError("No trained model found. Please train the model first.")
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_true, y_pred, y_proba, prefix=f'{set_name}_')
        
        # Log evaluation results
        logger.info(f"\n{set_name.capitalize()} Evaluation:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        
        return metrics
    
    def _calculate_metrics(
        self,
        y_true: ArrayLike,
        y_pred: ArrayLike,
        y_proba: Optional[ArrayLike] = None,
        prefix: str = ''
    ) -> Dict[str, float]:
        """
        Calculate evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (for probability-based metrics)
            prefix: Prefix for metric names
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Basic classification metrics
        metrics[f'{prefix}accuracy'] = accuracy_score(y_true, y_pred)
        metrics[f'{prefix}precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics[f'{prefix}recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics[f'{prefix}f1'] = f1_score(y_true, y_pred, zero_division=0)
        
        # Probability-based metrics
        if y_proba is not None:
            try:
                metrics[f'{prefix}roc_auc'] = roc_auc_score(y_true, y_proba)
                metrics[f'{prefix}pr_auc'] = average_precision_score(y_true, y_proba)
            except Exception as e:
                logger.warning(f"Could not calculate probability-based metrics: {str(e)}")
        
        return metrics
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # Save the model
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'ModelTrainer':
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            ModelTrainer instance with the loaded model
        """
        # Create a new instance
        instance = cls()
        
        # Load the model
        instance.model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
        
        return instance
    
    def get_feature_importances(self, top_n: Optional[int] = None) -> pd.DataFrame:
        """
        Get feature importances from the trained model.
        
        Args:
            top_n: Number of top features to return (None for all)
            
        Returns:
            DataFrame with feature importances
        """
        if self.feature_importances_ is None:
            self._extract_feature_importances()
        
        if self.feature_importances_ is not None:
            if top_n is not None and top_n > 0:
                return self.feature_importances_.head(top_n)
            return self.feature_importances_
        
        return pd.DataFrame()  # Return empty DataFrame if no importances available
