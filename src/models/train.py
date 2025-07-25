"""
Model training module for credit risk modeling.
"""
import os
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, fbeta_score,
    roc_auc_score, average_precision_score, confusion_matrix, roc_curve, 
    precision_recall_curve, classification_report, precision_recall_fscore_support,
    log_loss
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import json
import joblib
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import shap
from typing import Dict, Any, Tuple, List, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy.typing as npt
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, Any, Tuple, List, Optional
import joblib

# Import custom modules
from ..features.feature_engineering import create_feature_pipeline, create_target_variable

# Set up logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class ModelTrainer:
    """
    A simplified class to handle credit risk model training and evaluation.
    """
    
    def __init__(
        self, 
        experiment_name: str = "credit_risk_modeling",
        random_state: int = 42,
        n_jobs: int = -1
    ) -> None:
        """
        Initialize the ModelTrainer.
        
        Args:
            experiment_name: Name for the MLflow experiment
            random_state: Random seed for reproducibility
            n_jobs: Number of jobs to run in parallel (-1 uses all available cores)
        """
        self.experiment_name = experiment_name
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.best_model = None
        self.best_score = -np.inf
        self.best_model_name = None
        
        # Set up MLflow
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment(experiment_name)
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        self.metrics = {}
        
        # Define models and their parameter grids
        self.models = {
            'logistic_regression': {
                'model': LogisticRegression(
                    random_state=random_state,
                    class_weight='balanced',
                    max_iter=1000,
                    n_jobs=n_jobs
                ),
                'params': {
                    'model__C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'model__penalty': ['l1', 'l2'],
                    'model__solver': ['liblinear', 'saga']
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(
                    random_state=random_state,
                    class_weight='balanced',
                    n_jobs=n_jobs
                ),
                'params': {
                    'model__n_estimators': [100, 200, 300],
                    'model__max_depth': [None, 10, 20, 30],
                    'model__min_samples_split': [2, 5, 10],
                    'model__min_samples_leaf': [1, 2, 4]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(
                    random_state=random_state
                ),
                'params': {
                    'model__n_estimators': [100, 200],
                    'model__learning_rate': [0.01, 0.1, 0.2],
                    'model__max_depth': [3, 5, 7],
                    'model__min_samples_split': [2, 5],
                    'model__min_samples_leaf': [1, 2]
                }
            }
        }
        # Define model parameters for hyperparameter tuning
        self.model_params = self._get_model_params()
        
        # Log initialization parameters
        self.logger.info(
            f"Initialized ModelTrainer with experiment_name={experiment_name}"
        )
    
    def _get_model_params(self) -> Dict[str, Dict]:
        """
        Get model parameters and hyperparameter grids for different algorithms.
        
        Returns:
            Dictionary containing model configurations
        """
        return {
            'logistic_regression': {
                'model': LogisticRegression(
                    random_state=self.random_state,
                    class_weight='balanced',
                    max_iter=1000,
                    n_jobs=self.n_jobs
                ),
                'params': {
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(
                    random_state=self.random_state,
                    class_weight='balanced',
                    n_jobs=self.n_jobs
                ),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 1.0]
                }
            }
        }
    
    def _get_cv_strategy(self, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> Any:
        """
        Get cross-validation strategy based on configuration.
        
        Args:
            X: Features
            y: Target variable
            n_splits: Number of CV folds
            
        Returns:
            Cross-validation strategy object
        """
        if self.cv_strategy == 'time_series':
            from sklearn.model_selection import TimeSeriesSplit
            return TimeSeriesSplit(n_splits=n_splits)
        else:  # Default to stratified k-fold
            from sklearn.model_selection import StratifiedKFold
            return StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=self.random_state
            )
    
    def train_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame = None,
        y_test: pd.Series = None,
        models: Optional[List[str]] = None,
        cv: int = 5,
        n_iter: int = 20,
        scoring: Optional[Union[str, callable]] = None,
        refit: bool = True,
        return_train_score: bool = False
    ) -> Dict[str, Any]:
        """
        Train and evaluate multiple models with hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features (optional)
            y_test: Test target (optional)
            models: List of model names to train (default: all available)
            cv: Number of cross-validation folds
            n_iter: Number of parameter settings to sample
            scoring: Scoring metric (default: self.scoring)
            refit: Whether to refit the best model on the full training set
            return_train_score: Whether to include train scores in results
            
        Returns:
            Dictionary containing trained models and their metrics
        """
        if scoring is None:
            scoring = self.scoring
            
        if models is None:
            models = list(self.model_params.keys())
            
        # Store data for later use (e.g., SHAP explanations)
        self.X_train_ = X_train.copy()
        self.X_test_ = X_test.copy() if X_test is not None else None
        self.y_train_ = y_train.copy()
        self.y_test_ = y_test.copy() if y_test is not None else None
            
        results = {}
        
        for model_name in models:
            if model_name not in self.model_params:
                self.logger.warning(f"Model {model_name} not found in model parameters. Skipping...")
                continue
                
            with mlflow.start_run(run_name=f"{model_name}_tuning", nested=True):
                self.logger.info(f"\n{'='*50}")
                self.logger.info(f"Training {model_name}...")
                
                # Get model and parameters
                model_info = self.model_params[model_name]
                model = model_info['model']
                params = model_info['params']
                
                # Create pipeline with preprocessing
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('model', model)
                ])
                
                # Get CV strategy
                cv_strategy = self._get_cv_strategy(X_train, y_train, cv)
                
                # Hyperparameter tuning with RandomizedSearchCV
                search = RandomizedSearchCV(
                    estimator=pipeline,
                    param_distributions=params,
                    n_iter=n_iter,
                    cv=cv_strategy,
                    scoring=scoring,
                    refit=refit,
                    n_jobs=self.n_jobs,
                    random_state=self.random_state,
                    return_train_score=return_train_score,
                    verbose=1
                )
                
                # Train model
                self.logger.info(f"Starting hyperparameter search for {model_name}...")
                search.fit(X_train, y_train)
                
                # Get CV results
                cv_results = pd.DataFrame(search.cv_results_)
                self.cv_scores_[model_name] = cv_results
                
                # Log parameters and metrics
                mlflow.log_params({
                    f"best_{k}": v for k, v in search.best_params_.items()
                })
                mlflow.log_metric(f"best_cv_{scoring}", search.best_score_)
                
                # Store model and results
                model_results = {
                    'model': search.best_estimator_,
                    'best_params': search.best_params_,
                    'cv_results': cv_results,
                    'best_score': search.best_score_,
                    'scorer': search.scorer_,
                    'refit_time': search.refit_time_
                }
                
                # Evaluate on test set if available
                if X_test is not None and y_test is not None:
                    self.logger.info("Evaluating on test set...")
                    
                    # Make predictions
                    y_pred = search.predict(X_test)
                    y_pred_proba = search.predict_proba(X_test)[:, 1]
                    
                    # Calculate metrics
                    metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
                    
                    # Log metrics
                    for metric_name, metric_value in metrics.items():
                        mlflow.log_metric(f"test_{metric_name}", metric_value)
                    
                    # Update model results
                    model_results.update({
                        'test_metrics': metrics,
                        'y_true': y_test,
                        'y_pred': y_pred,
                        'y_pred_proba': y_pred_proba
                    })
                    
                    # Update best model
                    if metrics.get(scoring, 0) > self.best_score:
                        self.best_score = metrics[scoring]
                        self.best_model = search.best_estimator_
                        self.best_model_name = model_name
                        
                        # Store feature importances if available
                        if hasattr(search.best_estimator_.named_steps['model'], 'feature_importances_'):
                            self.feature_importances_ = pd.Series(
                                search.best_estimator_.named_steps['model'].feature_importances_,
                                index=X_train.columns
                            )
                        
                        self.logger.info(
                            f"New best model: {model_name} with {scoring}={metrics[scoring]:.4f}"
                        )
                
                # Log model
                mlflow.sklearn.log_model(
                    sk_model=search.best_estimator_,
                    artifact_path=model_name,
                    registered_model_name=f"{self.experiment_name}_{model_name}"
                )
                
                # Log artifacts
                self._log_training_artifacts(
                    model_name=model_name,
                    model=search.best_estimator_,
                    cv_results=cv_results,
                    X_test=X_test,
                    y_test=y_test
                )
                
                results[model_name] = model_results
                
                # Log completion
                self.logger.info(f"Completed training for {model_name}")
                self.logger.info(f"Best CV {scoring}: {search.best_score_:.4f}")
                if X_test is not None and y_test is not None:
                    self.logger.info(f"Test {scoring}: {metrics[scoring]:.4f}")
                self.logger.info(f"Best parameters: {search.best_params_}")
        
        return results
    
    def _get_model_configs(self, random_state: int) -> Dict[str, Any]:
        """
        Get model configurations and parameter grids.
        
        Args:
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary of model configurations
        """
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from xgboost import XGBClassifier
        from lightgbm import LGBMClassifier
        
        return {
            'random_forest': (
                RandomForestClassifier,
                {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'class_weight': ['balanced', 'balanced_subsample', None],
                    'random_state': [random_state]
                }
            ),
            'gradient_boosting': (
                GradientBoostingClassifier,
                {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2],
                    'subsample': [0.8, 0.9, 1.0],
                    'random_state': [random_state]
                }
            ),
            'xgboost': (
                XGBClassifier,
                {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 5, 7],
                    'min_child_weight': [1, 3, 5],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0],
                    'random_state': [random_state],
                    'use_label_encoder': [False],
                    'eval_metric': ['logloss']
                }
            ),
            'lightgbm': (
                LGBMClassifier,
                {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [-1, 3, 5, 7],
                    'num_leaves': [31, 50, 100],
                    'min_child_samples': [20, 50, 100],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0],
                    'random_state': [random_state],
                    'class_weight': ['balanced', None]
                }
            )
        }
    
    def train_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None,
        cv: int = 5,
        n_iter: int = 10,
        scoring: str = 'roc_auc'
    ) -> Dict[str, Any]:
        """
        Train and evaluate multiple models with hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features (optional)
            y_test: Test target (optional)
            cv: Number of cross-validation folds
            n_iter: Number of parameter settings to sample
            scoring: Scoring metric
            
        Returns:
            Dictionary containing trained models and their metrics
        """
        results = {}
        
        for model_name, model_info in self.models.items():
            with mlflow.start_run(run_name=f"{model_name}_tuning"):
                self.logger.info(f"\n{'='*50}")
                self.logger.info(f"Training {model_name}...")
                
                # Create pipeline with preprocessing
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('model', model_info['model'])
                ])
                
                # Hyperparameter tuning with RandomizedSearchCV
                search = RandomizedSearchCV(
                    estimator=pipeline,
                    param_distributions=model_info['params'],
                    n_iter=n_iter,
                    cv=cv,
                    scoring=scoring,
                    n_jobs=self.n_jobs,
                    random_state=self.random_state,
                    verbose=1
                )
                
                # Train model
                search.fit(X_train, y_train)
                
                # Log parameters and metrics
                mlflow.log_params({
                    f"best_{k}": v for k, v in search.best_params_.items()
                })
                
                # Store results
                results[model_name] = {
                    'model': search.best_estimator_,
                    'cv_score': search.best_score_,
                    'best_params': search.best_params_
                }
                
                # Evaluate on test set if available
                if X_test is not None and y_test is not None:
                    y_pred = search.best_estimator_.predict(X_test)
                    y_pred_proba = search.best_estimator_.predict_proba(X_test)[:, 1]
                    
                    # Calculate metrics
                    metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
                    results[model_name]['test_metrics'] = metrics
                    
                    # Log metrics
                    mlflow.log_metrics({
                        f'test_{k}': v for k, v in metrics.items()
                    })
                    
                    # Update best model
                    if metrics[scoring] > self.best_score:
                        self.best_score = metrics[scoring]
                        self.best_model = search.best_estimator_
                        self.best_model_name = model_name
                
                # Log model
                mlflow.sklearn.log_model(
                    sk_model=search.best_estimator_,
                    artifact_path=model_name
                )
                
                self.logger.info(f"Completed {model_name}")
                self.logger.info(f"Best CV {scoring}: {search.best_score_:.4f}")
                if 'test_metrics' in results[model_name]:
                    self.logger.info(f"Test {scoring}: {results[model_name]['test_metrics'][scoring]:.4f}")
        
        return results
    

    
    def save_best_model(self, output_dir: str = 'models') -> str:
        """
        Save the best model to disk.
        
        Args:
            output_dir: Directory to save the model
            
        Returns:
            Path to the saved model
        """
        if self.best_model is None:
            raise ValueError("No model has been trained yet. Call train_models() first.")
            
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the model
        model_path = os.path.join(output_dir, "best_model.pkl")
        joblib.dump(self.best_model, model_path)
        
        # Try to save feature importances if available
        try:
            if hasattr(self.best_model, 'feature_importances_'):
                # For tree-based models
                feature_importances = pd.Series(
                    self.best_model.feature_importances_,
                    index=getattr(self, 'feature_names_', [f'feature_{i}' for i in range(len(self.best_model.feature_importances_))])
                )
                # Save feature importances to CSV
                feature_importances.sort_values(ascending=False, inplace=True)
                feature_importances.to_csv(os.path.join(output_dir, "feature_importances.csv"))
                
            elif hasattr(self.best_model, 'coef_'):
                # For linear models
                coef = self.best_model.coef_
                if len(coef.shape) > 1:  # For multi-class
                    coef = coef[0]  # Take first class for binary classification
                feature_importances = pd.Series(
                    coef,
                    index=getattr(self, 'feature_names_', [f'feature_{i}' for i in range(len(coef))])
                )
                # Save feature importances to CSV
                feature_importances.sort_values(ascending=False, inplace=True)
                feature_importances.to_csv(os.path.join(output_dir, "feature_importances.csv"))
                
            self.logger.info(f"Feature importances saved to {os.path.join(output_dir, 'feature_importances.csv')}")
                
        except Exception as e:
            self.logger.warning(f"Could not save feature importances: {str(e)}")
            
        self.logger.info(f"Best model saved to {model_path}")
        return model_path
            
    
    def _log_training_artifacts(
        self,
        model_name: str,
        model: Any,
        cv_results: pd.DataFrame,
        X_train: pd.DataFrame = None,
        y_train: pd.Series = None,
        X_test: pd.DataFrame = None,
        y_test: pd.Series = None,
        output_dir: str = None
    ) -> None:
        """
        Log training artifacts including metrics, plots, and model explanations.
        
        Args:
            model_name: Name of the model
            model: Trained model
            cv_results: Cross-validation results
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            output_dir: Directory to save artifacts
        """
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        # Log cross-validation results
        if cv_results is not None:
            cv_results_df = pd.DataFrame(cv_results)
            if output_dir:
                cv_results_path = os.path.join(output_dir, 'cv_results.csv')
                cv_results_df.to_csv(cv_results_path, index=False)
                mlflow.log_artifact(cv_results_path)
        
        # 3. Log model-specific artifacts
        if hasattr(model.named_steps['model'], 'feature_importances_'):
            self._log_feature_importances(model, X_test if X_test is not None else X_train, tmp_dir)
        
        # 4. Generate SHAP explanations if training data is available
        if X_train is not None and len(X_train) > 0:
            try:
                self._log_shap_explanations(model, X_train, X_test, output_dir)
            except Exception as e:
                self.logger.error(f"Error generating SHAP explanations: {str(e)}")
                import traceback
                self.logger.debug(traceback.format_exc())
        
        # Customize the plot
        plt.xlabel('Iteration')
        plt.ylabel('Score')
        plt.title('Learning Curves')
        plt.legend(loc='best')
        
        # Save the plot
        learning_curve_file = os.path.join(output_dir, 'learning_curves.png')
        plt.savefig(learning_curve_file, bbox_inches='tight', dpi=300)
        plt.close()
        
        # Log the artifact
        mlflow.log_artifact(learning_curve_file)
    
    def _log_feature_importances(
        self, 
        model: Any, 
        X: pd.DataFrame,
        output_dir: str
    ) -> None:
        """
        Log feature importances for the model.
        
        Args:
            model: Trained model with feature_importances_ attribute
            X: Feature matrix (for column names)
            output_dir: Directory to save the plot
        """
        import matplotlib.pyplot as plt
        
        # Get feature importances
        importances = model.named_steps['model'].feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Plot feature importances
        plt.figure(figsize=(12, 8))
        plt.title("Feature Importances")
        plt.bar(
            range(X.shape[1]),
            importances[indices],
            color="r",
            align="center"
        )
        plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
        plt.xlim([-1, X.shape[1]])
        plt.tight_layout()
        
        # Save the plot
        importance_file = os.path.join(output_dir, 'feature_importances.png')
        plt.savefig(importance_file, bbox_inches='tight', dpi=300)
        plt.close()
        
        # Log the artifact
        mlflow.log_artifact(importance_file)
        
        # Save feature importances to CSV
        importances_df = pd.DataFrame({
            'feature': X.columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        importance_csv = os.path.join(output_dir, 'feature_importances.csv')
        importances_df.to_csv(importance_csv, index=False)
        mlflow.log_artifact(importance_csv)
    
    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray = None
    ) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics for credit risk model.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional)
            
        Returns:
            Dictionary of metrics including:
            - Binary classification metrics (accuracy, precision, recall, f1)
            - Probability metrics (roc_auc, average_precision)
            - Business metrics (cost of errors, profit curve)
        """
        metrics = {}
        
        # Binary classification metrics
        metrics.update({
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'f2': fbeta_score(y_true, y_pred, beta=2, zero_division=0)  # Higher weight on recall
        })
        
        # Probability metrics
        if y_prob is not None:
            try:
                metrics.update({
                    'roc_auc': roc_auc_score(y_true, y_prob),
                    'average_precision': average_precision_score(y_true, y_prob),
                    'log_loss': log_loss(y_true, y_prob)
                })
            except Exception as e:
                self.logger.warning(f"Error calculating probability metrics: {str(e)}")
            
            # Calculate metrics at different thresholds
            precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
            
            # Find best threshold based on F1 score
            best_idx = np.argmax(f1_scores)
            metrics.update({
                'best_threshold': float(thresholds[best_idx]),
                'best_f1': float(f1_scores[best_idx])
            })
            
            # Calculate cost of errors (assuming higher cost for false negatives in credit risk)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            cost_fn = fn * 10  # Higher cost for false negatives (missing a risky customer)
            cost_fp = fp * 2   # Lower cost for false positives (rejecting a good customer)
            metrics.update({
                'cost_of_errors': cost_fn + cost_fp,
                'false_negatives': int(fn),
                'false_positives': int(fp),
                'true_positives': int(tp),
                'true_negatives': int(tn)
            })
            
            # Calculate KS statistic
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            metrics['ks_statistic'] = np.max(tpr - fpr)
            
        # Calculate class distribution
        class_dist = np.bincount(y_true.astype(int))
        metrics.update({
            'class_ratio': float(class_dist[1] / len(y_true)) if len(class_dist) > 1 else 0.0,
            'n_samples': len(y_true)
        })
        
        return metrics
    
    def _log_model(
        self, 
        model: Any, 
        model_name: str, 
        metrics: Dict[str, float],
        params: Dict[str, Any],
        feature_names: List[str]
    ) -> None:
        """
        Log model and metrics to MLflow.
        
        Args:
            model: Trained model
            model_name: Name of the model
            metrics: Dictionary of evaluation metrics
            params: Model parameters
            feature_names: List of feature names
        """
        # Log parameters
        mlflow.log_params(params)
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Log model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=model_name,
            registered_model_name=f"{model_name}_credit_risk"
        )
        
        # Log feature importance if available
        if hasattr(model, 'feature_importances_') and hasattr(model, 'feature_names_in_'):
            try:
                # Create a DataFrame with feature importances
                importance_df = pd.DataFrame({
                    'feature': model.feature_names_in_,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)

                # Log feature importance as a table
                mlflow.log_table(
                    data=importance_df.to_dict('list'),
                    artifact_file=f"feature_importance_{model_name}.json"
                )

                # Log feature importance plot if we have features
                if not importance_df.empty:
                    plt.figure(figsize=(10, 6))
                    sns.barplot(
                        x='importance',
                        y='feature',
                        data=importance_df.head(20)
                    )
                    plt.title(f'Feature Importance - {model_name}')
                    plt.tight_layout()
                    mlflow.log_figure(plt.gcf(), f"feature_importance_{model_name}.png")
                    plt.close()
            except Exception as e:
                logger.warning(f"Failed to log feature importance: {str(e)}")
        
        # Skip confusion matrix logging in this version to simplify testing
        # This can be re-enabled later when we have proper test data
    
    def plot_feature_importance(self, top_n: int = 20, figsize: tuple = (12, 8)) -> None:
        """
        Plot feature importance for the best model.
        
        Args:
            top_n: Number of top features to display
            figsize: Figure size
        """
        if not hasattr(self, 'feature_importances_') or self.feature_importances_ is None:
            logger.warning("No feature importances available for the current model.")
            return
            
        # Get top N features
        top_features = self.feature_importances_.sort_values(ascending=False).head(top_n)
        
        # Create plot
        plt.figure(figsize=figsize)
        sns.barplot(x=top_features.values, y=top_features.index, palette='viridis')
        plt.title(f"Top {top_n} Most Important Features - {self.best_model_name}")
        plt.xlabel('Importance Score')
        plt.ylabel('Features')
        plt.tight_layout()
        
        # Save and log to MLflow
        plt.savefig('feature_importance.png')
        mlflow.log_artifact('feature_importance.png')
        plt.close()
    
    def plot_roc_curve(self, X_test: pd.DataFrame, y_test: pd.Series) -> None:
        """
        Plot ROC curve for the best model.
        
        Args:
            X_test: Test features
            y_test: True labels for test set
        """
        if self.best_model is None:
            raise ValueError("No model has been trained yet.")
            
        # Get predicted probabilities
        y_prob = self.best_model.predict_proba(X_test)[:, 1]
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        
        # Save and log to MLflow
        plt.savefig('roc_curve.png')
        mlflow.log_metric('roc_auc', roc_auc)
        mlflow.log_artifact('roc_curve.png')
        plt.close()
    
    def plot_precision_recall_curve(self, X_test: pd.DataFrame, y_test: pd.Series) -> None:
        """
        Plot precision-recall curve for the best model.
        
        Args:
            X_test: Test features
            y_test: True labels for test set
        """
        if self.best_model is None:
            raise ValueError("No model has been trained yet.")
            
        # Get predicted probabilities
        y_prob = self.best_model.predict_proba(X_test)[:, 1]
        
        # Calculate precision-recall curve
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        avg_precision = average_precision_score(y_test, y_prob)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        plt.step(recall, precision, where='post', label=f'Precision-Recall (AP = {avg_precision:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower left')
        
        # Save and log to MLflow
        plt.savefig('precision_recall_curve.png')
        mlflow.log_metric('average_precision', avg_precision)
        mlflow.log_artifact('precision_recall_curve.png')
        plt.close()
    
    def explain_model(
        self, 
        X: 'pd.DataFrame', 
        n_samples: int = 1000,
        show_plots: bool = True,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate SHAP explanations for the model's predictions.
        
        Args:
            X: Input features for explanation
            n_samples: Number of samples to use for SHAP value calculation
            show_plots: Whether to display the plots
            output_dir: Directory to save the plots (if None, plots won't be saved)
            
        Returns:
            Dictionary containing SHAP values and explanations
        """
        if self.best_model is None:
            raise ValueError("No model has been trained yet.")
            
        # Sample data if too large for efficient computation
        if len(X) > n_samples:
            X_sample = X.sample(n=min(n_samples, len(X)), random_state=42)
        else:
            X_sample = X.copy()
            
        # Get the underlying model from the pipeline
        model = self.best_model.named_steps['model']
        
        # Initialize explainer based on model type
        if hasattr(model, 'predict_proba'):
            explainer = shap.Explainer(
                model=model.predict_proba,
                masker=X_sample,
                feature_names=X_sample.columns.tolist(),
                output_names=['Low Risk', 'High Risk']
            )
        else:
            explainer = shap.Explainer(
                model=model.predict,
                masker=X_sample,
                feature_names=X_sample.columns.tolist()
            )
            
        # Calculate SHAP values
        shap_values = explainer(X_sample)
        
        # Generate and save plots
        if show_plots or output_dir:
            self._generate_shap_plots(shap_values, X_sample, output_dir)
            
        # Return explanation results
        return {
            'shap_values': shap_values,
            'expected_value': explainer.expected_value,
            'feature_importances': pd.Series(
                np.abs(shap_values.values).mean(axis=0),
                index=X_sample.columns
            ).sort_values(ascending=False).to_dict()
        }
        
    def _generate_shap_plots(
        self, 
        shap_values: 'shap.Explanation',
        X: 'pd.DataFrame',
        output_dir: Optional[str] = None
    ) -> None:
        """Generate and save SHAP plots."""
        # Create output directory if needed
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        # Summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X, show=False, plot_size=(12, 8))
        plt.tight_layout()
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'shap_summary.png'), 
                       bbox_inches='tight', dpi=300)
        plt.close()
        
        # Bar plot
        plt.figure(figsize=(12, 8))
        shap.plots.bar(shap_values, show=False)
        plt.tight_layout()
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'shap_bar.png'), 
                       bbox_inches='tight', dpi=300)
        plt.close()
        
        # Force plot for a single prediction (first instance)
        plt.figure()
        shap.plots.force(shap_values[0], matplotlib=True, show=False)
        plt.tight_layout()
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'shap_force_plot.png'), 
                       bbox_inches='tight', dpi=300)
        plt.close()
        
        # Decision plot
        plt.figure(figsize=(12, 8))
        shap.decision_plot(
            shap_values.base_values[0], 
            shap_values.values[0],
            X.columns,
            show=False
        )
        plt.tight_layout()
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'shap_decision_plot.png'), 
                       bbox_inches='tight', dpi=300)
        plt.close()
    
    def save_model(self, output_dir: str = 'models') -> str:
        """
        Save the best model to disk with metadata and SHAP explanations.
        
        Args:
            output_dir: Directory to save the model
            
        Returns:
            Path to the saved model
        """
        if self.best_model is None:
            raise ValueError("No model has been trained yet.")
            
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = os.path.join(output_dir, f"{self.best_model_name}_{timestamp}")
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(model_dir, "model.joblib")
        joblib.dump(self.best_model, model_path)
        
        # Save metadata
        metadata = {
            'model_name': self.best_model_name,
            'training_date': timestamp,
            'metrics': self.metrics,
            'feature_importances': self.feature_importances_.to_dict() if hasattr(self, 'feature_importances_') and self.feature_importances_ is not None else None,
            'model_type': type(self.best_model).__name__
        }
        
        with open(os.path.join(model_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Generate and save SHAP explanations if we have feature importances
        if hasattr(self, 'X_train_') and hasattr(self, 'X_test_'):
            try:
                # Use a sample of the training data for SHAP values
                X_sample = self.X_train_.sample(
                    n=min(1000, len(self.X_train_)), 
                    random_state=42
                )
                
                # Generate explanations
                explanations = self.explain_model(
                    X_sample,
                    show_plots=False,
                    output_dir=os.path.join(model_dir, 'explanations')
                )
                
                # Save SHAP values
                with open(os.path.join(model_dir, 'shap_values.pkl'), 'wb') as f:
                    joblib.dump(explanations['shap_values'], f)
                    
                # Update metadata with SHAP-based feature importances
                metadata.update({
                    'shap_feature_importances': explanations['feature_importances'],
                    'shap_expected_value': float(explanations['expected_value'])
                })
                
                # Save updated metadata
                with open(os.path.join(model_dir, 'metadata.json'), 'w') as f:
                    json.dump(metadata, f, indent=2)
                    
            except Exception as e:
                logger.warning(f"Failed to generate SHAP explanations: {str(e)}")
        
        logger.info(f"Saved model and metadata to {model_dir}")
        
        return model_path
    


def train_pipeline(
    data_path: str,
    experiment_name: str = "credit_risk_modeling",
    output_dir: str = 'models',
    random_state: int = 42
) -> Dict[str, Any]:
    """
    End-to-end training pipeline.
    
    Args:
        data_path: Path to the processed data
        experiment_name: Name of the MLflow experiment
        output_dir: Directory to save the model
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary containing training results
    """
    # Initialize MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(experiment_name)
    
    # Load data
    logger.info(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path)
    
    # Split features and target
    X = df.drop(columns=['is_high_risk', 'CustomerId'], errors='ignore')
    y = df['is_high_risk']
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    
    # Initialize and train models
    trainer = ModelTrainer(experiment_name=experiment_name)
    results = trainer.train_models(
        X_train, y_train, X_test, y_test,
        models=['random_forest', 'gradient_boosting', 'xgboost', 'lightgbm'],
        use_grid_search=True,
        n_iter=10,
        random_state=random_state
    )
    
    # Save the best model
    trainer.save_best_model(output_dir=output_dir)
    
    return results
