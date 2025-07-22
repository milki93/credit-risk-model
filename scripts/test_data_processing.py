"""
Test script for the enhanced data processing pipeline.
"""
import sys
import os
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_processing import DataProcessor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_data_processing():
    """Test the enhanced data processing pipeline."""
    logger.info("Testing enhanced data processing pipeline...")
    
    # Initialize data processor
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    processor = DataProcessor(data_dir=data_dir)
    
    # Process the data
    logger.info("Processing data...")
    df_processed, metadata = processor.process_data()
    
    # Log results
    logger.info(f"Processed data shape: {df_processed.shape}")
    logger.info(f"Features: {len(metadata['features'])}")
    logger.info(f"Target variable: {metadata['target']}")
    
    if 'is_high_risk' in df_processed.columns:
        risk_ratio = df_processed['is_high_risk'].mean()
        logger.info(f"High risk ratio: {risk_ratio:.2%}")
    
    # Save processed data
    output_path = os.path.join(data_dir, 'processed', 'test_processed_data.parquet')
    processor.save_processed_data(df_processed, output_path, format='parquet')
    logger.info(f"Saved processed data to {output_path}")
    
    return df_processed, metadata

if __name__ == "__main__":
    test_data_processing()
