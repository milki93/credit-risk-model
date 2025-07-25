#!/usr/bin/env python3
"""
Script to run the credit risk model training pipeline with enhanced ModelTrainer.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.train import ModelTrainer
from src.features.data_processor import DataProcessor

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train credit risk models')
    parser.add_argument('--data-dir', type=str, default='data/raw',
                       help='Directory containing the raw data files')
    parser.add_argument('--output-dir', type=str, default='models',
                       help='Directory to save trained models')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Proportion of data to use for testing')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--n-iter', type=int, default=20,
                       help='Number of parameter settings to sample for hyperparameter tuning')
    parser.add_argument('--cv', type=int, default=5,
                       help='Number of cross-validation folds')
    parser.add_argument('--scoring', type=str, default='roc_auc',
                       help='Scoring metric for model selection')
    parser.add_argument('--sample-size', type=int, default=None,
                       help='Number of rows to sample from data for testing (default: all)')
    return parser.parse_args()

def load_and_process_data(data_dir: str, test_size: float = 0.2, random_state: int = 42, sample_size: int = None) -> tuple:
    """
    Load and preprocess the dataset for training.
    
    Args:
        data_dir: Directory containing the raw data files
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        sample_size: Number of rows to sample from data for testing (default: all)
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, feature_names, metadata)
    """
    print("Loading and processing data...")
    
    # Initialize data processor
    processor = DataProcessor()
    
    # Load and process data
    data_file = os.path.join(data_dir, 'transactions.csv')
    data, metadata = processor.process_data(input_file=data_file, sample_size=sample_size)
    
    # Check if target variable exists
    if 'is_high_risk' not in data.columns:
        raise ValueError("Target variable 'is_high_risk' not found in the processed data.")
    
    # Drop identifier columns from features
    ID_COLS = ['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId']
    X = data.drop(columns=['is_high_risk'] + ID_COLS, errors='ignore')
    y = data['is_high_risk']
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y  # Maintain class distribution in splits
    )

    # Apply feature engineering pipeline: encode categoricals, scale numerics
    from src.features.feature_engineering import create_feature_pipeline
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
    feature_pipeline = create_feature_pipeline(numeric_cols, categorical_cols)
    X_train = feature_pipeline.fit_transform(X_train)
    X_test = feature_pipeline.transform(X_test)

    print(f"Training set: {X_train.shape[0]:,} samples ({y_train.mean():.2%} positive)")
    print(f"Test set: {X_test.shape[0]:,} samples ({y_test.mean():.2%} positive)")
    print(f"Number of features: {X_train.shape[1]}")

    return X_train, X_test, y_train, y_test, numeric_cols + categorical_cols, metadata

def main():
    import mlflow
    mlflow.end_run()
    """Main function to run the training pipeline."""
    # Parse command line arguments
    args = parse_args()
    
    # Set random seeds for reproducibility
    np.random.seed(args.random_state)
    
    # Set up MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    experiment_name = f"credit_risk_modeling_{datetime.now().strftime('%Y%m%d')}"
    mlflow.set_experiment(experiment_name)
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test, feature_names, metadata = load_and_process_data(
        args.data_dir, 
        test_size=args.test_size,
        random_state=args.random_state,
        sample_size=args.sample_size
    )
    
    # Initialize model trainer
    trainer = ModelTrainer(
        experiment_name=experiment_name,
        random_state=args.random_state
    )
    
    # Train models
    print("\nStarting model training...")
    results = trainer.train_models(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        cv=args.cv,
        n_iter=args.n_iter,
        scoring=args.scoring
    )
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    trainer.plot_feature_importance()
    trainer.plot_roc_curve(X_test, y_test)
    trainer.plot_precision_recall_curve(X_test, y_test)
    
    # Save the best model
    print("\nSaving the best model...")
    model_path = trainer.save_model(args.output_dir)
    
    # Log metadata and feature names
    mlflow.end_run()
    with mlflow.start_run():
        mlflow.log_params({
            'n_features': len(feature_names),
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test),
            'class_ratio': y_train.mean(),
            'random_state': args.random_state
        })
        
        # Log feature names as an artifact
        with open('feature_names.txt', 'w') as f:
            f.write('\n'.join(feature_names))
        mlflow.log_artifact('feature_names.txt')
        
        # Log processing metadata
        mlflow.log_dict(metadata, 'processing_metadata.json')
    
    print("\nTraining completed successfully!")
    print(f"Best model: {trainer.best_model_name} (Score: {trainer.best_score:.4f})")
    print(f"Model saved to: {model_path}")
    print("\nTo view results, run:")
    print("  mlflow ui --backend-store-uri file:./mlruns")

if __name__ == "__main__":
    main()
