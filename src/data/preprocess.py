"""
Enhanced data preprocessing module for EthioMart NER pipeline.

Handles comprehensive text cleaning, normalization, and preparation for NER training
with advanced Amharic text processing capabilities.
"""

import re
import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime
import unicodedata

logger = logging.getLogger(__name__)


class EnhancedDataPreprocessor:
    """Enhanced preprocessor for Amharic e-commerce text data with NER preparation."""

    def __init__(self, data_dir: str = "data"):
        """Initialize the enhanced preprocessor.
        
        Args:
            data_dir: Directory containing raw and processed data
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.labelled_dir = self.data_dir / "labelled"
        
        # Create directories
        for dir_path in [self.processed_dir, self.labelled_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def normalize_amharic_text(self, text: str) -> str:
        """Normalize Amharic text for better processing.
        
        Args:
            text: Raw text to normalize
            
        Returns:
            Normalized Amharic text
        """
        if not isinstance(text, str):
            return ""
        
        # Unicode normalization
        text = unicodedata.normalize('NFC', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove phone numbers (Ethiopian format)
        text = re.sub(r'(\+251|0)?[0-9]{9}', '', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep Amharic, English, numbers, and basic punctuation
        text = re.sub(r'[^\w\s\u1200-\u137F\u1380-\u139F\u2D80-\u2DDF\uAB00-\uAB2F.,!?]', '', text)
        
        return text.strip()

    def extract_entities_with_regex(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Extract entities using regex patterns for Amharic text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with entity types and their positions
        """
        entities = {
            "PRICE": [],
            "PHONE": [],
            "LOCATION": [],
            "PRODUCT": []
        }
        
        # Price patterns (Ethiopian Birr)
        price_patterns = [
            (r'(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:ብር|birr|ETB)', 'PRICE'),
            (r'(?:ብር|birr|ETB)\s*(\d+(?:,\d{3})*(?:\.\d{2})?)', 'PRICE'),
        ]
        
        for pattern, entity_type in price_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entities[entity_type].append({
                    "text": match.group(1) if match.group(1) else match.group(0),
                    "start": match.start(),
                    "end": match.end(),
                    "confidence": 0.9
                })
        
        # Phone number patterns
        phone_patterns = [
            (r'(\+251[0-9]{9})', 'PHONE'),
            (r'(0[0-9]{9})', 'PHONE'),
        ]
        
        for pattern, entity_type in phone_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                entities[entity_type].append({
                    "text": match.group(1),
                    "start": match.start(),
                    "end": match.end(),
                    "confidence": 0.8
                })
        
        # Location patterns (common Ethiopian locations)
        location_keywords = [
            "አዲስ አበባ", "ትግራይ", "አማራ", "ኦሮሚያ", "ሶማሌ", "ቤኒሻንጉል", 
            "ጋምቤላ", "ሀረሪ", "ድሬዳዋ", "ጅማ", "ሀዋሳ", "ባህር ዳር"
        ]
        
        for location in location_keywords:
            if location in text:
                start = text.find(location)
                entities["LOCATION"].append({
                    "text": location,
                    "start": start,
                    "end": start + len(location),
                    "confidence": 0.7
                })
        
        return entities

    def prepare_for_ner_training(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for NER training with entity annotations.
        
        Args:
            df: DataFrame with text data
            
        Returns:
            DataFrame ready for NER training
        """
        # Clean text
        df['normalized_text'] = df['cleaned_text'].apply(self.normalize_amharic_text)
        
        # Remove empty texts
        df = df[df['normalized_text'].str.len() > 10].copy()
        
        # Extract entities
        df['entities'] = df['normalized_text'].apply(self.extract_entities_with_regex)
        
        # Add metadata for training
        df['text_length'] = df['normalized_text'].str.len()
        df['entity_count'] = df['entities'].apply(lambda x: sum(len(v) for v in x.values()))
        
        # Filter messages with at least some entities for better training
        df = df[df['entity_count'] > 0].copy()
        
        return df

    def create_conll_format(self, text: str, entities: Dict[str, List[Dict[str, Any]]]) -> List[Tuple[str, str]]:
        """Convert text and entities to CONLL format.
        
        Args:
            text: Input text
            entities: Dictionary of entities with positions
            
        Returns:
            List of (token, label) tuples in CONLL format
        """
        # Simple tokenization (split by whitespace)
        tokens = text.split()
        labels = ['O'] * len(tokens)
        
        # Map entity positions to token positions
        for entity_type, entity_list in entities.items():
            for entity in entity_list:
                entity_text = entity['text']
                entity_start = entity['start']
                entity_end = entity['end']
                
                # Find tokens that overlap with entity
                token_start = 0
                for i, token in enumerate(tokens):
                    token_end = token_start + len(token)
                    
                    # Check if token overlaps with entity
                    if (token_start <= entity_start < token_end) or \
                       (token_start < entity_end <= token_end) or \
                       (entity_start <= token_start and entity_end >= token_end):
                        
                        # Assign BIO labels
                        if i == 0 or labels[i-1] == 'O':
                            labels[i] = f'B-{entity_type}'
                        else:
                            labels[i] = f'I-{entity_type}'
                    
                    token_start = token_end + 1  # +1 for space
        
        return list(zip(tokens, labels))

    def save_conll_dataset(self, df: pd.DataFrame, filename: str = "ner_dataset.conll"):
        """Save dataset in CONLL format.
        
        Args:
            df: DataFrame with text and entities
            filename: Output filename
        """
        output_file = self.labelled_dir / filename
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for _, row in df.iterrows():
                text = row['normalized_text']
                entities = row['entities']
                
                # Convert to CONLL format
                conll_data = self.create_conll_format(text, entities)
                
                # Write tokens and labels
                for token, label in conll_data:
                    f.write(f"{token}\t{label}\n")
                
                # Add empty line between sentences
                f.write("\n")
        
        logger.info(f"Saved CONLL dataset to {output_file}")

    def split_data(self, df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1) -> Dict[str, pd.DataFrame]:
        """Split data into train/validation/test sets with stratification.
        
        Args:
            df: DataFrame to split
            test_size: Proportion for test set
            val_size: Proportion for validation set
            
        Returns:
            Dictionary with train, validation, and test DataFrames
        """
        # Create stratification column based on entity types
        df['entity_types'] = df['entities'].apply(
            lambda x: '_'.join(sorted([k for k, v in x.items() if len(v) > 0]))
        )
        
        # Split into train and temp
        train_df, temp_df = train_test_split(
            df, test_size=test_size + val_size, 
            stratify=df['entity_types'], random_state=42
        )
        
        # Split temp into validation and test
        val_ratio = val_size / (test_size + val_size)
        val_df, test_df = train_test_split(
            temp_df, test_size=1 - val_ratio,
            stratify=temp_df['entity_types'], random_state=42
        )
        
        return {
            "train": train_df,
            "validation": val_df,
            "test": test_df
        }

    def save_processed_data(self, data_dict: Dict[str, pd.DataFrame], prefix: str = "ner_data"):
        """Save processed data to files.
        
        Args:
            data_dict: Dictionary with train/val/test DataFrames
            prefix: Prefix for output files
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for split_name, df in data_dict.items():
            # Save as JSON
            json_file = self.processed_dir / f"{prefix}_{split_name}_{timestamp}.json"
            df.to_json(json_file, orient="records", indent=2)
            
            # Save as CSV
            csv_file = self.processed_dir / f"{prefix}_{split_name}_{timestamp}.csv"
            df.to_csv(csv_file, index=False, encoding='utf-8')
            
            logger.info(f"Saved {split_name} data ({len(df)} samples) to {json_file} and {csv_file}")

    def run_comprehensive_preprocessing_pipeline(self, input_file: str) -> Dict[str, pd.DataFrame]:
        """Run the complete comprehensive preprocessing pipeline.
        
        Args:
            input_file: Path to input JSON file
            
        Returns:
            Dictionary with processed train/val/test DataFrames
        """
        logger.info(f"Starting comprehensive preprocessing pipeline for {input_file}")
        
        # Load data
        df = pd.read_json(input_file)
        logger.info(f"Loaded {len(df)} samples")
        
        # Prepare for NER
        df = self.prepare_for_ner_training(df)
        logger.info(f"Prepared {len(df)} samples for NER training")
        
        # Create CONLL dataset
        self.save_conll_dataset(df)
        
        # Split data
        data_splits = self.split_data(df)
        
        # Save processed data
        self.save_processed_data(data_splits)
        
        # Print statistics
        self.print_dataset_statistics(data_splits)
        
        return data_splits

    def print_dataset_statistics(self, data_splits: Dict[str, pd.DataFrame]):
        """Print comprehensive dataset statistics.
        
        Args:
            data_splits: Dictionary with train/val/test DataFrames
        """
        logger.info("Dataset Statistics:")
        logger.info("=" * 50)
        
        for split_name, df in data_splits.items():
            logger.info(f"\n{split_name.upper()} SET:")
            logger.info(f"  Samples: {len(df)}")
            logger.info(f"  Avg text length: {df['text_length'].mean():.1f}")
            logger.info(f"  Avg entities per text: {df['entity_count'].mean():.1f}")
            
            # Entity type distribution
            entity_counts = {}
            for entities in df['entities']:
                for entity_type, entity_list in entities.items():
                    entity_counts[entity_type] = entity_counts.get(entity_type, 0) + len(entity_list)
            
            logger.info(f"  Entity distribution:")
            for entity_type, count in entity_counts.items():
                logger.info(f"    {entity_type}: {count}")
        
        logger.info("=" * 50) 