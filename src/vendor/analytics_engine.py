"""
Enhanced vendor analytics engine for EthioMart micro-lending scorecards.

Provides comprehensive vendor analysis and scoring for fintech micro-lending decisions
based on extracted NER entities and post metadata.
"""

import logging
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class EnhancedVendorAnalytics:
    """Enhanced analytics engine for vendor performance analysis and micro-lending scorecards."""

    def __init__(self, data_dir: str = "data"):
        """Initialize the enhanced analytics engine."""
        self.data_dir = Path(data_dir)
        self.output_dir = Path("outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Scoring weights for different metrics
        self.scoring_weights = {
            "posting_frequency": 0.25,
            "engagement_rate": 0.20,
            "price_consistency": 0.15,
            "product_diversity": 0.15,
            "location_coverage": 0.10,
            "response_time": 0.10,
            "media_quality": 0.05
        }

    def extract_vendor_profiles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract vendor profiles from raw data."""
        logger.info("Extracting vendor profiles...")
        
        vendor_profiles = []
        
        for sender_id, group in df.groupby('sender_id'):
            if pd.isna(sender_id):
                continue
                
            profile = {
                'vendor_id': str(sender_id),
                'vendor_username': group['sender_username'].iloc[0] if not pd.isna(group['sender_username'].iloc[0]) else f"vendor_{sender_id}",
                'vendor_name': self._get_vendor_name(group),
                'total_posts': len(group),
                'first_post_date': group['date'].min(),
                'last_post_date': group['date'].max(),
                'active_days': (pd.to_datetime(group['date'].max()) - pd.to_datetime(group['date'].min())).days + 1,
                'channels': list(group['channel'].unique()),
                'total_views': group['views'].sum(),
                'total_forwards': group['forwards'].sum(),
                'total_replies': group['replies'].sum(),
                'media_posts': len(group[group['has_media'] == True]),
                'avg_views_per_post': group['views'].mean(),
                'avg_forwards_per_post': group['forwards'].mean(),
                'avg_replies_per_post': group['replies'].mean(),
            }
            
            # Extract entities from all posts
            all_entities = []
            for entities in group['entities']:
                if isinstance(entities, dict):
                    all_entities.extend(entities.values())
            
            # Analyze entities
            entity_analysis = self._analyze_entities(all_entities)
            profile.update(entity_analysis)
            
            vendor_profiles.append(profile)
        
        return pd.DataFrame(vendor_profiles)

    def _get_vendor_name(self, group: pd.DataFrame) -> str:
        """Extract vendor name from group data."""
        first_names = group['sender_first_name'].dropna().unique()
        last_names = group['sender_last_name'].dropna().unique()
        
        if len(first_names) > 0 and len(last_names) > 0:
            return f"{first_names[0]} {last_names[0]}"
        elif len(first_names) > 0:
            return first_names[0]
        elif len(last_names) > 0:
            return last_names[0]
        else:
            return f"Vendor_{group['sender_id'].iloc[0]}"

    def _analyze_entities(self, all_entities: List[List[Dict]]) -> Dict[str, Any]:
        """Analyze entities for vendor profile."""
        flat_entities = []
        for entity_list in all_entities:
            if isinstance(entity_list, list):
                flat_entities.extend(entity_list)
        
        # Extract different entity types
        prices = [e['text'] for e in flat_entities if isinstance(e, dict) and e.get('text') and 'PRICE' in str(e)]
        phones = [e['text'] for e in flat_entities if isinstance(e, dict) and e.get('text') and 'PHONE' in str(e)]
        locations = [e['text'] for e in flat_entities if isinstance(e, dict) and e.get('text') and 'LOCATION' in str(e)]
        products = [e['text'] for e in flat_entities if isinstance(e, dict) and e.get('text') and 'PRODUCT' in str(e)]
        
        # Analyze prices
        price_values = []
        for price in prices:
            try:
                import re
                numeric_value = re.findall(r'\d+(?:,\d{3})*(?:\.\d{2})?', str(price))
                if numeric_value:
                    price_values.append(float(numeric_value[0].replace(',', '')))
            except:
                continue
        
        return {
            'total_entities': len(flat_entities),
            'unique_prices': len(set(prices)),
            'unique_phones': len(set(phones)),
            'unique_locations': len(set(locations)),
            'unique_products': len(set(products)),
            'avg_price': np.mean(price_values) if price_values else 0,
            'min_price': np.min(price_values) if price_values else 0,
            'max_price': np.max(price_values) if price_values else 0,
            'price_std': np.std(price_values) if len(price_values) > 1 else 0,
            'top_locations': list(set(locations))[:5],
            'top_products': list(set(products))[:5],
        }

    def calculate_vendor_metrics(self, vendor_profiles: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive vendor metrics for scoring."""
        logger.info("Calculating vendor metrics...")
        
        metrics_df = vendor_profiles.copy()
        
        # Posting frequency (posts per week)
        metrics_df['posting_frequency'] = metrics_df['total_posts'] / (metrics_df['active_days'] / 7)
        
        # Engagement rate (views + forwards + replies per post)
        metrics_df['engagement_rate'] = (
            metrics_df['total_views'] + metrics_df['total_forwards'] + metrics_df['total_replies']
        ) / metrics_df['total_posts']
        
        # Price consistency (lower std = more consistent)
        metrics_df['price_consistency'] = 1 / (1 + metrics_df['price_std'])
        
        # Product diversity (more unique products = higher score)
        metrics_df['product_diversity'] = metrics_df['unique_products'] / metrics_df['total_posts']
        
        # Location coverage (more locations = higher score)
        metrics_df['location_coverage'] = metrics_df['unique_locations'] / metrics_df['total_posts']
        
        # Response time (estimated from post frequency)
        metrics_df['response_time'] = 1 / (1 + metrics_df['posting_frequency'])
        
        # Media quality (percentage of posts with media)
        metrics_df['media_quality'] = metrics_df['media_posts'] / metrics_df['total_posts']
        
        # Normalize metrics to 0-1 scale
        scaler = StandardScaler()
        metric_columns = [
            'posting_frequency', 'engagement_rate', 'price_consistency',
            'product_diversity', 'location_coverage', 'response_time', 'media_quality'
        ]
        
        metrics_df[metric_columns] = scaler.fit_transform(metrics_df[metric_columns])
        
        # Ensure all metrics are positive (0-1 scale)
        for col in metric_columns:
            metrics_df[col] = (metrics_df[col] - metrics_df[col].min()) / (metrics_df[col].max() - metrics_df[col].min())
        
        return metrics_df

    def calculate_lending_score(self, metrics_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate lending score for each vendor."""
        logger.info("Calculating lending scores...")
        
        scored_df = metrics_df.copy()
        
        # Calculate weighted lending score
        score_components = []
        for metric, weight in self.scoring_weights.items():
            if metric in scored_df.columns:
                score_components.append(scored_df[metric] * weight)
        
        scored_df['lending_score'] = sum(score_components)
        
        # Normalize lending score to 0-100 scale
        scored_df['lending_score'] = (scored_df['lending_score'] - scored_df['lending_score'].min()) / \
                                   (scored_df['lending_score'].max() - scored_df['lending_score'].min()) * 100
        
        # Assign risk categories
        scored_df['risk_category'] = pd.cut(
            scored_df['lending_score'],
            bins=[0, 30, 60, 80, 100],
            labels=['High Risk', 'Medium Risk', 'Low Risk', 'Excellent']
        )
        
        # Assign lending recommendations
        scored_df['lending_recommendation'] = scored_df['lending_score'].apply(self._get_lending_recommendation)
        
        return scored_df

    def _get_lending_recommendation(self, score: float) -> str:
        """Get lending recommendation based on score."""
        if score >= 80:
            return "Approve - High Limit"
        elif score >= 60:
            return "Approve - Standard Limit"
        elif score >= 40:
            return "Approve - Low Limit"
        elif score >= 20:
            return "Review Required"
        else:
            return "Decline"

    def generate_scorecard(self, scored_df: pd.DataFrame) -> pd.DataFrame:
        """Generate final vendor scorecard."""
        logger.info("Generating vendor scorecard...")
        
        # Select relevant columns for scorecard
        scorecard_columns = [
            'vendor_id', 'vendor_name', 'vendor_username', 'total_posts',
            'active_days', 'avg_views_per_post', 'avg_price', 'unique_products',
            'unique_locations', 'lending_score', 'risk_category', 'lending_recommendation'
        ]
        
        scorecard = scored_df[scorecard_columns].copy()
        
        # Sort by lending score (descending)
        scorecard = scorecard.sort_values('lending_score', ascending=False)
        
        # Add rank
        scorecard['rank'] = range(1, len(scorecard) + 1)
        
        return scorecard

    def save_scorecard(self, scorecard: pd.DataFrame, filename: str = None):
        """Save scorecard to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"vendor_scorecard_{timestamp}.csv"
        
        output_file = self.output_dir / filename
        scorecard.to_csv(output_file, index=False, encoding='utf-8')
        
        logger.info(f"✓ Scorecard saved to {output_file}")
        return str(output_file)

    def run_comprehensive_analytics(self, data_file: str) -> Dict[str, Any]:
        """Run complete vendor analytics pipeline."""
        logger.info("Starting comprehensive vendor analytics pipeline...")
        
        # Load data
        df = pd.read_json(data_file)
        logger.info(f"Loaded {len(df)} messages for analysis")
        
        # Extract vendor profiles
        vendor_profiles = self.extract_vendor_profiles(df)
        logger.info(f"Extracted {len(vendor_profiles)} vendor profiles")
        
        # Calculate metrics
        metrics_df = self.calculate_vendor_metrics(vendor_profiles)
        
        # Calculate lending scores
        scored_df = self.calculate_lending_score(metrics_df)
        
        # Generate scorecard
        scorecard = self.generate_scorecard(scored_df)
        
        # Save results
        scorecard_file = self.save_scorecard(scorecard)
        
        return {
            "scorecard": scorecard,
            "vendor_profiles": vendor_profiles,
            "metrics": metrics_df,
            "scorecard_file": scorecard_file
        } 