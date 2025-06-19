"""
Data preprocessing module for EthioMart NER pipeline.

Handles text cleaning, normalization, and preparation for NER training.
"""

import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocessor for Amharic e-commerce text data."""

    def __init__(self, data_dir: str = "data"):
        """Initialize the preprocessor.
        
        Args:
            data_dir: Directory containing raw and processed data
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def clean_text(self, text: str) -> str:
        """Clean and normalize Amharic text.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove phone numbers
        text = re.sub(r'(\+251|0)?[0-9]{9}', '', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep Amharic
        text = re.sub(r'[^\w\s\u1200-\u137F\u1380-\u139F\u2D80-\u2DDF\uAB00-\uAB2F]', '', text)
        
        return text.strip()

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract potential entities from text using regex patterns.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of entity types and their values
        """
        entities = {
            "PRICE": [],
            "PHONE": [],
            "LOCATION": [],
            "PRODUCT": []
        }
        
        # Price patterns (Ethiopian Birr)
        price_patterns = [
            r'(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:ብር|birr|ETB)',
            r'(?:ብር|birr|ETB)\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',
        ]
        
        for pattern in price_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities["PRICE"].extend(matches)
        
        # Phone number patterns
        phone_patterns = [
            r'(\+251[0-9]{9})',
            r'(0[0-9]{9})',
        ]
        
        for pattern in phone_patterns:
            matches = re.findall(pattern, text)
            entities["PHONE"].extend(matches)
        
        return entities

    def prepare_ner_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for NER training.
        
        Args:
            df: DataFrame with text data
            
        Returns:
            DataFrame ready for NER training
        """
        # Clean text
        df['cleaned_text'] = df['text'].apply(self.clean_text)
        
        # Remove empty texts
        df = df[df['cleaned_text'].str.len() > 10].copy()
        
        # Extract entities
        df['entities'] = df['cleaned_text'].apply(self.extract_entities)
        
        return df

    def split_data(self, df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1) -> Dict[str, pd.DataFrame]:
        """Split data into train/validation/test sets.
        
        Args:
            df: DataFrame to split
            test_size: Proportion for test set
            val_size: Proportion for validation set
            
        Returns:
            Dictionary with train, validation, and test DataFrames
        """
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
        train_df, val_df = train_test_split(train_df, test_size=val_size, random_state=42)
        
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
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        for split_name, df in data_dict.items():
            output_file = self.processed_dir / f"{prefix}_{split_name}_{timestamp}.json"
            df.to_json(output_file, orient="records", indent=2)
            logger.info(f"Saved {split_name} data ({len(df)} samples) to {output_file}")

    def run_preprocessing_pipeline(self, input_file: str) -> Dict[str, pd.DataFrame]:
        """Run the complete preprocessing pipeline.
        
        Args:
            input_file: Path to input JSON file
            
        Returns:
            Dictionary with processed train/val/test DataFrames
        """
        logger.info(f"Starting preprocessing pipeline for {input_file}")
        
        # Load data
        df = pd.read_json(input_file)
        logger.info(f"Loaded {len(df)} samples")
        
        # Prepare for NER
        df = self.prepare_ner_data(df)
        logger.info(f"Prepared {len(df)} samples for NER")
        
        # Split data
        data_splits = self.split_data(df)
        
        # Save processed data
        self.save_processed_data(data_splits)
        
        return data_splits 