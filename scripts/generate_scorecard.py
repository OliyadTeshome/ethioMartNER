#!/usr/bin/env python3
"""
Vendor scorecard generation script for EthioMart NER pipeline.
"""

import logging
import click
import pandas as pd
from pathlib import Path
from src.vendor.analytics_engine import VendorAnalytics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option("--data-file", "-f", required=True, help="Input data file")
@click.option("--output-dir", "-o", default="outputs", help="Output directory")
def main(data_file, output_dir):
    """Generate vendor scorecards."""
    # Load data
    df = pd.read_json(data_file)
    
    # Analyze vendors
    analytics = VendorAnalytics()
    performance = analytics.analyze_vendor_performance(df)
    
    # Generate scorecards
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save results
    scorecard_file = output_path / "vendor_scorecard.json"
    with open(scorecard_file, 'w') as f:
        import json
        json.dump(performance, f, indent=2)
    
    logger.info(f"Scorecard saved to {scorecard_file}")


if __name__ == "__main__":
    main() 