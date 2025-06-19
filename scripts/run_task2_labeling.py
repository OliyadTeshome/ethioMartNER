#!/usr/bin/env python3
"""
Task 2: Label Data in CoNLL Format Script

This script helps with converting processed data to CONLL format for NER training:
1. Load processed data from Task 1
2. Apply regex-based entity extraction
3. Convert to CONLL format
4. Support manual annotation workflow
"""

import logging
import click
from pathlib import Path
import pandas as pd
from datetime import datetime
import sys
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.preprocess import EnhancedDataPreprocessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/task2_labeling.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@click.command()
@click.option("--data-file", "-d", default=None, help="Path to processed data file")
@click.option("--output-file", "-o", default="data/labelled/ner_dataset.conll", help="Output CONLL file")
@click.option("--min-entities", "-m", default=1, help="Minimum entities required per text")
@click.option("--max-texts", "-n", default=1000, help="Maximum number of texts to process")
@click.option("--manual-review", is_flag=True, help="Enable manual review workflow")
def main(data_file, output_file, min_entities, max_texts, manual_review):
    """Run Task 2: Convert data to CONLL format for NER training."""
    
    logger.info("=" * 60)
    logger.info("TASK 2: LABEL DATA IN CONLL FORMAT")
    logger.info("=" * 60)
    
    # Step 1: Find data file if not specified
    if data_file is None:
        data_file = find_latest_processed_data()
        if data_file is None:
            logger.error("No processed data found. Please run Task 1 first.")
            return
    
    logger.info(f"Using data file: {data_file}")
    
    # Step 2: Load and prepare data
    logger.info("Step 1: Loading processed data...")
    df = pd.read_json(data_file)
    logger.info(f"Loaded {len(df)} samples")
    
    # Step 3: Initialize preprocessor
    logger.info("Step 2: Initializing preprocessor...")
    preprocessor = EnhancedDataPreprocessor()
    
    # Step 4: Prepare data for NER training
    logger.info("Step 3: Preparing data for NER training...")
    df = preprocessor.prepare_for_ner_training(df)
    logger.info(f"Prepared {len(df)} samples for NER training")
    
    # Step 5: Filter by minimum entities
    df = df[df['entity_count'] >= min_entities].copy()
    logger.info(f"Filtered to {len(df)} samples with >= {min_entities} entities")
    
    # Step 6: Limit number of texts
    if len(df) > max_texts:
        df = df.head(max_texts).copy()
        logger.info(f"Limited to {len(df)} samples")
    
    # Step 7: Generate CONLL format
    logger.info("Step 4: Generating CONLL format...")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    preprocessor.save_conll_dataset(df, output_path.name)
    
    # Step 8: Generate statistics
    logger.info("Step 5: Generating statistics...")
    generate_labeling_statistics(df, output_path.parent)
    
    # Step 9: Manual review if requested
    if manual_review:
        logger.info("Step 6: Starting manual review workflow...")
        run_manual_review(df, output_path.parent)
    
    logger.info("\n" + "=" * 60)
    logger.info("TASK 2 COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)
    logger.info(f"CONLL dataset saved to: {output_file}")
    logger.info(f"Total samples: {len(df)}")
    logger.info(f"Total entities: {df['entity_count'].sum()}")
    logger.info("Next steps:")
    logger.info("1. Review the CONLL dataset")
    logger.info("2. Manually annotate additional samples if needed")
    logger.info("3. Run Task 3: Fine-tune NER model")


def find_latest_processed_data() -> str:
    """Find the latest processed data file."""
    processed_dir = Path("data/processed")
    if not processed_dir.exists():
        return None
    
    # Look for processed data files
    json_files = list(processed_dir.glob("*_train_*.json"))
    if not json_files:
        return None
    
    # Return the most recent file
    latest_file = max(json_files, key=lambda x: x.stat().st_mtime)
    return str(latest_file)


def generate_labeling_statistics(df: pd.DataFrame, output_dir: Path):
    """Generate statistics about the labeled dataset."""
    stats_file = output_dir / f"labeling_statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Calculate entity type distribution
    entity_counts = {}
    for entities in df['entities']:
        for entity_type, entity_list in entities.items():
            entity_counts[entity_type] = entity_counts.get(entity_type, 0) + len(entity_list)
    
    statistics = {
        "total_samples": len(df),
        "total_entities": df['entity_count'].sum(),
        "avg_entities_per_sample": df['entity_count'].mean(),
        "entity_distribution": entity_counts,
        "text_length_stats": {
            "min": int(df['text_length'].min()),
            "max": int(df['text_length'].max()),
            "mean": float(df['text_length'].mean()),
            "median": float(df['text_length'].median())
        },
        "samples_with_entities": int((df['entity_count'] > 0).sum()),
        "samples_without_entities": int((df['entity_count'] == 0).sum())
    }
    
    with open(stats_file, 'w') as f:
        json.dump(statistics, f, indent=2)
    
    logger.info(f"✓ Labeling statistics saved to {stats_file}")
    
    # Print summary
    logger.info("\nLABELING STATISTICS:")
    logger.info(f"  Total samples: {statistics['total_samples']}")
    logger.info(f"  Total entities: {statistics['total_entities']}")
    logger.info(f"  Avg entities per sample: {statistics['avg_entities_per_sample']:.2f}")
    logger.info("  Entity distribution:")
    for entity_type, count in entity_counts.items():
        logger.info(f"    {entity_type}: {count}")


def run_manual_review(df: pd.DataFrame, output_dir: Path):
    """Run manual review workflow for CONLL data."""
    logger.info("Manual review workflow not yet implemented.")
    logger.info("Please manually review the generated CONLL file:")
    logger.info(f"  {output_dir}/ner_dataset.conll")
    logger.info("You can use tools like:")
    logger.info("  - Brat annotation tool")
    logger.info("  - Label Studio")
    logger.info("  - Custom annotation interface")


if __name__ == "__main__":
    main() 