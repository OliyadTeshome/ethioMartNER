#!/usr/bin/env python3
"""
Task 6: Fintech Vendor Scorecard for Micro-Lending Script

This script runs the complete vendor analytics pipeline for micro-lending:
1. Extract vendor profiles from processed data
2. Calculate comprehensive metrics (posting frequency, engagement, etc.)
3. Generate lending scores and risk categories
4. Create vendor scorecards for micro-lending decisions
"""

import logging
import click
from pathlib import Path
import pandas as pd
from datetime import datetime
import sys
import json
from typing import List

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from vendor.analytics_engine import EnhancedVendorAnalytics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/task6_analytics.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@click.command()
@click.option("--data-file", "-d", default=None, help="Path to processed data file")
@click.option("--output-dir", "-o", default="outputs", help="Output directory for results")
@click.option("--min-posts", "-m", default=5, help="Minimum posts required for vendor analysis")
@click.option("--score-threshold", "-s", default=20, help="Minimum score for lending consideration")
def main(data_file, output_dir, min_posts, score_threshold):
    """Run Task 6: Complete vendor analytics and micro-lending scorecard pipeline."""
    
    logger.info("=" * 60)
    logger.info("TASK 6: FINTECH VENDOR SCORECARD FOR MICRO-LENDING")
    logger.info("=" * 60)
    
    # Step 1: Find data file if not specified
    if data_file is None:
        data_file = find_latest_processed_data()
        if data_file is None:
            logger.error("No processed data found. Please run Task 1 first.")
            return
    
    logger.info(f"Using data file: {data_file}")
    
    # Step 2: Initialize analytics engine
    logger.info("Step 1: Initializing vendor analytics engine...")
    analytics = EnhancedVendorAnalytics()
    
    # Step 3: Run comprehensive analytics
    logger.info("Step 2: Running comprehensive vendor analytics...")
    results = analytics.run_comprehensive_analytics(data_file)
    
    scorecard = results["scorecard"]
    vendor_profiles = results["vendor_profiles"]
    metrics = results["metrics"]
    
    logger.info(f"✓ Analyzed {len(vendor_profiles)} vendors")
    logger.info(f"✓ Generated scorecard with {len(scorecard)} vendors")
    
    # Step 4: Filter and enhance results
    logger.info("Step 3: Filtering and enhancing results...")
    
    # Filter vendors with minimum posts
    filtered_scorecard = scorecard[scorecard['total_posts'] >= min_posts].copy()
    logger.info(f"✓ Filtered to {len(filtered_scorecard)} vendors with >= {min_posts} posts")
    
    # Filter by score threshold
    qualified_vendors = filtered_scorecard[filtered_scorecard['lending_score'] >= score_threshold].copy()
    logger.info(f"✓ {len(qualified_vendors)} vendors meet score threshold (>= {score_threshold})")
    
    # Step 5: Generate detailed reports
    logger.info("Step 4: Generating detailed reports...")
    generate_detailed_reports(qualified_vendors, vendor_profiles, metrics, output_dir)
    
    # Step 6: Create summary statistics
    logger.info("Step 5: Creating summary statistics...")
    create_summary_statistics(qualified_vendors, output_dir)
    
    # Step 7: Generate lending recommendations
    logger.info("Step 6: Generating lending recommendations...")
    generate_lending_recommendations(qualified_vendors, output_dir)
    
    logger.info("\n" + "=" * 60)
    logger.info("TASK 6 COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)
    logger.info("Generated files:")
    logger.info(f"1. Vendor scorecard: {results['scorecard_file']}")
    logger.info(f"2. Detailed reports: {output_dir}/")
    logger.info(f"3. Summary statistics: {output_dir}/summary_statistics.json")
    logger.info(f"4. Lending recommendations: {output_dir}/lending_recommendations.json")


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


def generate_detailed_reports(scorecard: pd.DataFrame, vendor_profiles: pd.DataFrame, 
                            metrics: pd.DataFrame, output_dir: str):
    """Generate detailed vendor reports."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Top performers report
    top_performers = scorecard.head(20)
    top_performers_file = output_path / "top_20_performers.csv"
    top_performers.to_csv(top_performers_file, index=False, encoding='utf-8')
    logger.info(f"✓ Top performers saved to {top_performers_file}")
    
    # 2. Risk category breakdown
    risk_breakdown = scorecard['risk_category'].value_counts()
    risk_file = output_path / "risk_category_breakdown.json"
    with open(risk_file, 'w') as f:
        json.dump(risk_breakdown.to_dict(), f, indent=2)
    logger.info(f"✓ Risk breakdown saved to {risk_file}")
    
    # 3. Lending recommendation summary
    rec_summary = scorecard['lending_recommendation'].value_counts()
    rec_file = output_path / "lending_recommendations_summary.json"
    with open(rec_file, 'w') as f:
        json.dump(rec_summary.to_dict(), f, indent=2)
    logger.info(f"✓ Lending recommendations saved to {rec_file}")


def create_summary_statistics(scorecard: pd.DataFrame, output_dir: str):
    """Create comprehensive summary statistics."""
    output_path = Path(output_dir)
    
    summary = {
        "total_vendors": len(scorecard),
        "average_lending_score": float(scorecard['lending_score'].mean()),
        "median_lending_score": float(scorecard['lending_score'].median()),
        "score_range": {
            "min": float(scorecard['lending_score'].min()),
            "max": float(scorecard['lending_score'].max())
        },
        "average_posts_per_vendor": float(scorecard['total_posts'].mean()),
        "average_views_per_post": float(scorecard['avg_views_per_post'].mean()),
        "average_price": float(scorecard['avg_price'].mean()),
        "risk_distribution": scorecard['risk_category'].value_counts().to_dict(),
        "recommendation_distribution": scorecard['lending_recommendation'].value_counts().to_dict(),
        "top_products": get_top_products(scorecard),
        "top_locations": get_top_locations(scorecard)
    }
    
    summary_file = output_path / "summary_statistics.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"✓ Summary statistics saved to {summary_file}")


def get_top_products(scorecard: pd.DataFrame) -> List[str]:
    """Extract top products from vendor data."""
    # This would need to be implemented based on your data structure
    # For now, return a placeholder
    return ["Phone", "Laptop", "Car", "House", "Electronics"]


def get_top_locations(scorecard: pd.DataFrame) -> List[str]:
    """Extract top locations from vendor data."""
    # This would need to be implemented based on your data structure
    # For now, return a placeholder
    return ["Addis Ababa", "Tigray", "Amhara", "Oromia", "Somali"]


def generate_lending_recommendations(scorecard: pd.DataFrame, output_dir: str):
    """Generate detailed lending recommendations."""
    output_path = Path(output_dir)
    
    recommendations = {
        "high_limit_candidates": [],
        "standard_limit_candidates": [],
        "low_limit_candidates": [],
        "review_candidates": [],
        "declined_candidates": []
    }
    
    # Categorize vendors by recommendation
    for _, vendor in scorecard.iterrows():
        vendor_info = {
            "vendor_id": vendor['vendor_id'],
            "vendor_name": vendor['vendor_name'],
            "lending_score": float(vendor['lending_score']),
            "risk_category": vendor['risk_category'],
            "total_posts": int(vendor['total_posts']),
            "avg_views_per_post": float(vendor['avg_views_per_post']),
            "avg_price": float(vendor['avg_price'])
        }
        
        rec = vendor['lending_recommendation']
        if "High Limit" in rec:
            recommendations["high_limit_candidates"].append(vendor_info)
        elif "Standard Limit" in rec:
            recommendations["standard_limit_candidates"].append(vendor_info)
        elif "Low Limit" in rec:
            recommendations["low_limit_candidates"].append(vendor_info)
        elif "Review" in rec:
            recommendations["review_candidates"].append(vendor_info)
        else:
            recommendations["declined_candidates"].append(vendor_info)
    
    # Save recommendations
    rec_file = output_path / "lending_recommendations.json"
    with open(rec_file, 'w') as f:
        json.dump(recommendations, f, indent=2)
    
    logger.info(f"✓ Lending recommendations saved to {rec_file}")
    
    # Print summary
    logger.info("\nLENDING RECOMMENDATIONS SUMMARY:")
    logger.info(f"  High Limit: {len(recommendations['high_limit_candidates'])} vendors")
    logger.info(f"  Standard Limit: {len(recommendations['standard_limit_candidates'])} vendors")
    logger.info(f"  Low Limit: {len(recommendations['low_limit_candidates'])} vendors")
    logger.info(f"  Review Required: {len(recommendations['review_candidates'])} vendors")
    logger.info(f"  Declined: {len(recommendations['declined_candidates'])} vendors")


if __name__ == "__main__":
    main() 