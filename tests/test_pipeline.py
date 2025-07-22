import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_engineering import preprocess_data, create_feature_engineering_pipeline

# Set random seed for reproducibility
np.random.seed(42)

def generate_sample_data(n_samples=1000):
    """Generate sample transaction data for testing."""
    # Generate customer IDs (fewer unique customers to ensure some have multiple transactions)
    n_customers = max(10, n_samples // 20)  # Ensure at least 10 customers
    customer_ids = np.random.randint(1, n_customers + 1, size=n_samples)
    
    # Generate transaction amounts (positive values)
    amounts = np.round(np.random.uniform(10, 1000, size=n_samples), 2)
    values = amounts * np.random.uniform(0.8, 1.2, size=n_samples)  # Slightly different from amount
    
    # Generate transaction dates within the last 90 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
    # Generate timestamps with some clustering by customer
    dates = []
    for _ in range(n_samples):
        # Add some randomness but keep transactions from same customer closer in time
        customer_id = np.random.choice(customer_ids)
        customer_days = np.random.normal(30, 10)  # Around 30 days from now
        days_ago = max(0, min(90, customer_days + np.random.normal(0, 5)))
        
        dates.append(
            end_date - timedelta(days=days_ago) + 
            timedelta(hours=np.random.randint(0, 24), 
                     minutes=np.random.randint(0, 60))
        )
    
    # Generate some categorical features
    categories = np.random.choice(['A', 'B', 'C'], size=n_samples, p=[0.5, 0.3, 0.2])
    channels = np.random.choice(['Web', 'App', 'In-Store'], size=n_samples, p=[0.4, 0.4, 0.2])
    currencies = np.random.choice(['USD', 'EUR', 'GBP'], size=n_samples, p=[0.7, 0.2, 0.1])
    
    # Generate some binary target (0/1) with some correlation to features
    fraud_prob = 0.05 + 0.1 * (amounts > 800) + 0.1 * (channels == 'Web')
    fraud_prob = np.clip(fraud_prob, 0, 1)
    fraud_result = np.random.binomial(1, fraud_prob)
    
    # Create DataFrame
    df = pd.DataFrame({
        'CustomerId': customer_ids,
        'Amount': amounts,
        'Value': np.round(values, 2),
        'CurrencyCode': currencies,
        'ProductCategory': categories,
        'ChannelId': channels,
        'TransactionStartTime': dates,
        'FraudResult': fraud_result
    })
    
    return df

if __name__ == "__main__":
    # Generate sample data
    print("Generating sample data...")
    df = generate_sample_data(1000)
    
    # Define column types
    numerical_cols = ['Amount', 'Value']
    categorical_cols = ['CurrencyCode', 'ProductCategory', 'ChannelId']
    date_col = 'TransactionStartTime'
    target_col = 'FraudResult'
    
    # Print sample of the data
    print("\nOriginal DataFrame:")
    print(df.head())
    
    print("\nDataFrame info:")
    print(df.info())
    
    print("\nValue counts for categorical columns:")
    for col in categorical_cols:
        print(f"\n{col}:")
        print(df[col].value_counts())
    
    # Test the preprocessing pipeline without WOE first
    print("\nTesting preprocessing pipeline (without WOE)...")
    try:
        processed_df = preprocess_data(
            df=df,
            numerical_cols=numerical_cols,
            categorical_cols=categorical_cols,
            target_col=None,  # Skip WOE transformation for now
            drop_original=True
        )
        
        # Print results
        print("\nPreprocessing complete!")
        print("\nProcessed DataFrame shape:", processed_df.shape)
        print("\nFirst few rows of processed data:")
        print(processed_df.head())
        
        print("\nProcessed DataFrame info:")
        print(processed_df.info())
        
    except Exception as e:
        print(f"\nError during preprocessing: {str(e)}")
        import traceback
        traceback.print_exc()
