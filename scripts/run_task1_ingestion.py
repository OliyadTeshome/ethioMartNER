#!/usr/bin/env python3
"""
Task 1: Amharic Data Ingestion & Preprocessing Script

This script runs the complete data ingestion pipeline for EthioMart NER:
1. Scrape data from Ethiopian Telegram e-commerce channels
2. Extract comprehensive metadata (text, media, sender info)
3. Preprocess and clean Amharic text
4. Save structured results in multiple formats
"""

import asyncio
import logging
import click
from pathlib import Path
import pandas as pd
from datetime import datetime
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.telegram_scraper import EnhancedTelegramScraper
from data.preprocess import EnhancedDataPreprocessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/task1_ingestion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@click.command()
@click.option("--channels", "-c", multiple=True, 
              default=["ethiopian_marketplace", "addis_deals", "ethio_shopping", "addis_tech_market", "ethiopia_electronics"],
              help="Telegram channels to scrape")
@click.option("--limit", "-l", default=200, help="Messages per channel")
@click.option("--days", "-d", default=30, help="Days back to scrape")
@click.option("--skip-scraping", is_flag=True, help="Skip scraping and use existing data")
def main(channels, limit, days, skip_scraping):
    """Run Task 1: Complete data ingestion and preprocessing pipeline."""
    
    logger.info("=" * 60)
    logger.info("TASK 1: AMHARIC DATA INGESTION & PREPROCESSING")
    logger.info("=" * 60)
    
    # Step 1: Data Ingestion
    if not skip_scraping:
        logger.info("Step 1: Scraping Telegram channels...")
        raw_df, filtered_df = asyncio.run(run_scraping(channels, limit, days))
        
        if raw_df.empty:
            logger.error("No data was scraped. Please check your API credentials and channel names.")
            return
        
        logger.info(f"✓ Scraped {len(raw_df)} total messages")
        logger.info(f"✓ Filtered to {len(filtered_df)} e-commerce messages")
    else:
        logger.info("Step 1: Skipping scraping, using existing data...")
        # Load existing data
        data_dir = Path("data/raw")
        json_files = list(data_dir.glob("ecommerce_filtered_*.json"))
        
        if not json_files:
            logger.error("No existing filtered data found. Please run without --skip-scraping first.")
            return
        
        # Load most recent file
        latest_file = max(json_files, key=lambda x: x.stat().st_mtime)
        filtered_df = pd.read_json(latest_file)
        logger.info(f"✓ Loaded {len(filtered_df)} messages from {latest_file}")
    
    # Step 2: Data Preprocessing
    logger.info("\nStep 2: Preprocessing and preparing data for NER...")
    preprocessor = EnhancedDataPreprocessor()
    
    # Save filtered data for preprocessing
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_file = Path("data/raw") / f"temp_for_preprocessing_{timestamp}.json"
    filtered_df.to_json(temp_file, orient="records", indent=2)
    
    # Run preprocessing pipeline
    data_splits = preprocessor.run_comprehensive_preprocessing_pipeline(str(temp_file))
    
    # Clean up temp file
    temp_file.unlink()
    
    logger.info("✓ Preprocessing completed successfully")
    
    # Step 3: Generate Summary Report
    logger.info("\nStep 3: Generating summary report...")
    generate_summary_report(filtered_df, data_splits)
    
    logger.info("\n" + "=" * 60)
    logger.info("TASK 1 COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)
    logger.info("Next steps:")
    logger.info("1. Review the generated CONLL dataset in data/labelled/")
    logger.info("2. Manually annotate additional samples if needed")
    logger.info("3. Run Task 2: Label Data in CoNLL Format")


async def run_scraping(channels, limit, days):
    """Run the Telegram scraping pipeline."""
    async with EnhancedTelegramScraper() as scraper:
        raw_df, filtered_df = await scraper.run_comprehensive_scraping_pipeline(
            list(channels), limit_per_channel=limit, days_back=days
        )
        return raw_df, filtered_df


def generate_summary_report(raw_df, data_splits):
    """Generate a comprehensive summary report."""
    report_file = Path("data/processed") / f"task1_summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("ETHIOMART NER PIPELINE - TASK 1 SUMMARY REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        # Raw data statistics
        f.write("RAW DATA STATISTICS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total messages scraped: {len(raw_df)}\n")
        f.write(f"E-commerce messages: {len(raw_df[raw_df['is_ecommerce'] == True])}\n")
        f.write(f"Messages with media: {len(raw_df[raw_df['has_media'] == True])}\n")
        f.write(f"Unique channels: {raw_df['channel'].nunique()}\n")
        f.write(f"Date range: {raw_df['date'].min()} to {raw_df['date'].max()}\n\n")
        
        # Channel distribution
        f.write("CHANNEL DISTRIBUTION:\n")
        f.write("-" * 30 + "\n")
        channel_counts = raw_df['channel'].value_counts()
        for channel, count in channel_counts.items():
            f.write(f"{channel}: {count} messages\n")
        f.write("\n")
        
        # Processed data statistics
        f.write("PROCESSED DATA STATISTICS:\n")
        f.write("-" * 30 + "\n")
        for split_name, df in data_splits.items():
            f.write(f"\n{split_name.upper()} SET:\n")
            f.write(f"  Samples: {len(df)}\n")
            f.write(f"  Avg text length: {df['text_length'].mean():.1f}\n")
            f.write(f"  Avg entities per text: {df['entity_count'].mean():.1f}\n")
            
            # Entity distribution
            entity_counts = {}
            for entities in df['entities']:
                for entity_type, entity_list in entities.items():
                    entity_counts[entity_type] = entity_counts.get(entity_type, 0) + len(entity_list)
            
            f.write(f"  Entity distribution:\n")
            for entity_type, count in entity_counts.items():
                f.write(f"    {entity_type}: {count}\n")
        
        # File locations
        f.write("\nGENERATED FILES:\n")
        f.write("-" * 30 + "\n")
        f.write("Raw data: data/raw/\n")
        f.write("Processed data: data/processed/\n")
        f.write("CONLL dataset: data/labelled/ner_dataset.conll\n")
        f.write("Summary report: data/processed/task1_summary_report_*.txt\n")
    
    logger.info(f"✓ Summary report saved to {report_file}")


if __name__ == "__main__":
    main() 