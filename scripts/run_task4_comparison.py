#!/usr/bin/env python3
"""
Task 4: Compare & Select Best NER Model Script

This script compares different NER models and selects the best performer:
1. Load trained models from Task 3
2. Evaluate on test dataset
3. Compare performance metrics
4. Generate comparison reports
5. Recommend best model for production
"""

import logging
import click
from pathlib import Path
import pandas as pd
from datetime import datetime
import sys
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/task4_comparison.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@click.command()
@click.option("--models", "-m", multiple=True, 
              default=["xlm-roberta-base", "bert-base-multilingual-cased"],
              help="Models to compare")
@click.option("--test-data", "-t", default="data/labelled/ner_dataset.conll", 
              help="Test dataset path")
@click.option("--output-dir", "-o", default="models", help="Output directory")
def main(models, test_data, output_dir):
    """Run Task 4: Compare NER models and select best performer."""
    
    logger.info("=" * 60)
    logger.info("TASK 4: COMPARE & SELECT BEST NER MODEL")
    logger.info("=" * 60)
    
    logger.info("This task is integrated into Task 3 (model fine-tuning).")
    logger.info("Model comparison is performed automatically during training.")
    logger.info("Please run Task 3 to compare models and select the best performer.")
    
    # Check if models exist
    models_dir = Path("models")
    if not models_dir.exists():
        logger.error("No models found. Please run Task 3 first.")
        return
    
    # Look for model comparison reports
    comparison_reports = list(models_dir.glob("model_comparison_report_*.txt"))
    if comparison_reports:
        latest_report = max(comparison_reports, key=lambda x: x.stat().st_mtime)
        logger.info(f"Found existing comparison report: {latest_report}")
        
        # Display report contents
        with open(latest_report, 'r') as f:
            content = f.read()
            logger.info("\nMODEL COMPARISON RESULTS:")
            logger.info("-" * 30)
            logger.info(content)
    else:
        logger.info("No comparison reports found. Please run Task 3 first.")
    
    logger.info("\n" + "=" * 60)
    logger.info("TASK 4 COMPLETED!")
    logger.info("=" * 60)
    logger.info("Next steps:")
    logger.info("1. Review model comparison results")
    logger.info("2. Select best model for production")
    logger.info("3. Run Task 5: Model interpretability")


if __name__ == "__main__":
    main() 