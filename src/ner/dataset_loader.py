"""
Dataset loader for NER training and evaluation.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class NERDatasetLoader:
    """Loader for NER datasets in various formats."""

    def __init__(self, model_name: str = "bert-base-multilingual-cased"):
        """Initialize the dataset loader.
        
        Args:
            model_name: Hugging Face model name for tokenizer
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.label2id = {
            "O": 0,
            "B-PRICE": 1,
            "I-PRICE": 2,
            "B-PHONE": 3,
            "I-PHONE": 4,
            "B-LOCATION": 5,
            "I-LOCATION": 6,
            "B-PRODUCT": 7,
            "I-PRODUCT": 8,
        }
        self.id2label = {v: k for k, v in self.label2id.items()}

    def load_json_data(self, file_path: str) -> pd.DataFrame:
        """Load data from JSON file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            DataFrame with loaded data
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return pd.DataFrame(data)

    def tokenize_and_align_labels(self, examples: Dict) -> Dict:
        """Tokenize text and align labels with tokens.
        
        Args:
            examples: Dictionary with text and labels
            
        Returns:
            Tokenized examples with aligned labels
        """
        tokenized_inputs = self.tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # For now, return basic tokenization
        # In a full implementation, you'd align labels with subword tokens
        return tokenized_inputs

    def create_dataset(self, df: pd.DataFrame) -> Dataset:
        """Create HuggingFace Dataset from DataFrame.
        
        Args:
            df: DataFrame with text and entity information
            
        Returns:
            HuggingFace Dataset
        """
        # Convert DataFrame to list of dictionaries
        data = df.to_dict('records')
        
        # Create dataset
        dataset = Dataset.from_list(data)
        
        return dataset

    def load_datasets(self, data_dir: str) -> DatasetDict:
        """Load train/validation/test datasets.
        
        Args:
            data_dir: Directory containing processed data files
            
        Returns:
            DatasetDict with train/validation/test splits
        """
        data_path = Path(data_dir)
        
        # Find the most recent processed files
        train_files = list(data_path.glob("ner_data_train_*.json"))
        val_files = list(data_path.glob("ner_data_validation_*.json"))
        test_files = list(data_path.glob("ner_data_test_*.json"))
        
        if not train_files or not val_files or not test_files:
            raise FileNotFoundError("Could not find processed data files")
        
        # Load most recent files
        train_df = self.load_json_data(str(max(train_files, key=lambda x: x.stat().st_mtime)))
        val_df = self.load_json_data(str(max(val_files, key=lambda x: x.stat().st_mtime)))
        test_df = self.load_json_data(str(max(test_files, key=lambda x: x.stat().st_mtime)))
        
        # Create datasets
        train_dataset = self.create_dataset(train_df)
        val_dataset = self.create_dataset(val_df)
        test_dataset = self.create_dataset(test_df)
        
        return DatasetDict({
            "train": train_dataset,
            "validation": val_dataset,
            "test": test_dataset
        }) 