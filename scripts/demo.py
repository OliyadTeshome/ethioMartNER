#!/usr/bin/env python3
"""
EthioMart NER Pipeline Demo

This script demonstrates the complete pipeline functionality using sample data.
Perfect for testing the setup and understanding the workflow.
"""

import logging
import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.preprocess import EnhancedDataPreprocessor
from vendor.analytics_engine import EnhancedVendorAnalytics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_data():
    """Create sample e-commerce data for demonstration."""
    logger.info("Creating sample e-commerce data...")
    
    # Sample Amharic e-commerce messages
    sample_messages = [
        {
            "message_id": 1,
            "date": (datetime.now() - timedelta(days=1)).isoformat(),
            "text": "ስልክ ሽያጭ አለ 5000 ብር አዲስ አበባ ውስጥ ነው +251912345678",
            "cleaned_text": "ስልክ ሽያጭ አለ 5000 ብር አዲስ አበባ ውስጥ ነው +251912345678",
            "channel": "ethiopian_marketplace",
            "views": 150,
            "forwards": 5,
            "replies": 3,
            "is_ecommerce": True,
            "sender_id": "user1",
            "sender_username": "vendor1",
            "sender_first_name": "Abebe",
            "sender_last_name": "Kebede",
            "has_media": False
        },
        {
            "message_id": 2,
            "date": (datetime.now() - timedelta(days=2)).isoformat(),
            "text": "ላፕቶፕ ግዛ 15000 ብር ትግራይ ክልል ውስጥ ነው",
            "cleaned_text": "ላፕቶፕ ግዛ 15000 ብር ትግራይ ክልል ውስጥ ነው",
            "channel": "addis_deals",
            "views": 200,
            "forwards": 8,
            "replies": 6,
            "is_ecommerce": True,
            "sender_id": "user2",
            "sender_username": "vendor2",
            "sender_first_name": "Kebede",
            "sender_last_name": "Alemayehu",
            "has_media": True
        },
        {
            "message_id": 3,
            "date": (datetime.now() - timedelta(days=3)).isoformat(),
            "text": "መኪና ሽያጭ አለ 250000 ብር ኦሮሚያ ክልል ውስጥ ነው +251987654321",
            "cleaned_text": "መኪና ሽያጭ አለ 250000 ብር ኦሮሚያ ክልል ውስጥ ነው +251987654321",
            "channel": "ethio_shopping",
            "views": 300,
            "forwards": 12,
            "replies": 8,
            "is_ecommerce": True,
            "sender_id": "user3",
            "sender_username": "vendor3",
            "sender_first_name": "Tigist",
            "sender_last_name": "Haile",
            "has_media": True
        }
    ]
    
    # Create DataFrame
    df = pd.DataFrame(sample_messages)
    
    # Save sample data
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    sample_file = data_dir / "sample_ecommerce_data.json"
    df.to_json(sample_file, orient="records", indent=2)
    
    logger.info(f"✓ Sample data saved to {sample_file}")
    return str(sample_file)


def run_demo_pipeline():
    """Run the complete demo pipeline."""
    logger.info("=" * 60)
    logger.info("ETHIOMART NER PIPELINE DEMO")
    logger.info("=" * 60)
    
    # Step 1: Create sample data
    logger.info("\nStep 1: Creating sample data...")
    sample_file = create_sample_data()
    
    # Step 2: Data preprocessing
    logger.info("\nStep 2: Running data preprocessing...")
    preprocessor = EnhancedDataPreprocessor()
    
    # Prepare data for NER training
    df = pd.read_json(sample_file)
    df = preprocessor.prepare_for_ner_training(df)
    
    # Save processed data
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    processed_file = processed_dir / "sample_processed_data.json"
    df.to_json(processed_file, orient="records", indent=2)
    
    logger.info(f"✓ Processed data saved to {processed_file}")
    
    # Step 3: Generate CONLL dataset
    logger.info("\nStep 3: Generating CONLL dataset...")
    labelled_dir = Path("data/labelled")
    labelled_dir.mkdir(parents=True, exist_ok=True)
    
    preprocessor.save_conll_dataset(df, "sample_ner_dataset.conll")
    logger.info("✓ CONLL dataset generated")
    
    # Step 4: Vendor analytics
    logger.info("\nStep 4: Running vendor analytics...")
    analytics = EnhancedVendorAnalytics()
    
    results = analytics.run_comprehensive_analytics(str(processed_file))
    
    logger.info(f"✓ Vendor analytics completed")
    logger.info(f"  - Scorecard: {results['scorecard_file']}")
    logger.info(f"  - Vendors analyzed: {len(results['scorecard'])}")
    
    # Step 5: Display results
    logger.info("\nStep 5: Demo Results Summary")
    logger.info("-" * 40)
    
    # Display sample data info
    logger.info(f"Sample messages: {len(df)}")
    logger.info(f"Total entities: {df['entity_count'].sum()}")
    
    # Display vendor scorecard
    scorecard = results['scorecard']
    if not scorecard.empty:
        logger.info(f"\nVendor Scorecard (Top {min(3, len(scorecard))}):")
        for i, (_, vendor) in enumerate(scorecard.head(3).iterrows()):
            logger.info(f"  {i+1}. {vendor['vendor_name']} - Score: {vendor['lending_score']:.1f} ({vendor['risk_category']})")
    
    logger.info("\n" + "=" * 60)
    logger.info("DEMO COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)
    logger.info("Generated files:")
    logger.info(f"  - Sample data: {sample_file}")
    logger.info(f"  - Processed data: {processed_file}")
    logger.info(f"  - CONLL dataset: data/labelled/sample_ner_dataset.conll")
    logger.info(f"  - Vendor scorecard: {results['scorecard_file']}")
    logger.info("\nNext steps:")
    logger.info("1. Review generated files")
    logger.info("2. Run complete pipeline with real data")
    logger.info("3. Customize parameters for your use case")


if __name__ == "__main__":
    run_demo_pipeline() 