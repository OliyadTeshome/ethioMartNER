"""
Enhanced NER model fine-tuning module for Amharic text.

Supports multiple transformer models with proper tokenization and label alignment
for Amharic NER tasks.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    EvalPrediction
)
from datasets import Dataset, DatasetDict
import evaluate
from sklearn.metrics import classification_report, confusion_matrix

logger = logging.getLogger(__name__)


class EnhancedNERModelTrainer:
    """Enhanced trainer for NER model fine-tuning with multiple model support."""

    def __init__(
        self,
        model_name: str = "xlm-roberta-base",
        output_dir: str = "models",
        num_labels: int = 9,
        max_length: int = 512,
        device: Optional[str] = None
    ):
        """Initialize the enhanced trainer.
        
        Args:
            model_name: Base model name (xlm-roberta-base, bert-base-multilingual-cased, etc.)
            output_dir: Directory to save models
            num_labels: Number of NER labels
            max_length: Maximum sequence length
            device: Device to use (cuda, cpu, or auto-detect)
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_labels = num_labels
        self.max_length = max_length
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Label mapping
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
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name, 
            num_labels=num_labels,
            id2label=self.id2label,
            label2id=self.label2id
        )
        
        # Add special tokens for Amharic if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"Initialized trainer with model: {model_name}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Labels: {list(self.label2id.keys())}")

    def load_conll_data(self, conll_file: str) -> DatasetDict:
        """Load data from CONLL format file.
        
        Args:
            conll_file: Path to CONLL file
            
        Returns:
            DatasetDict with train/validation/test splits
        """
        logger.info(f"Loading CONLL data from {conll_file}")
        
        sentences = []
        current_sentence = {"tokens": [], "labels": []}
        
        with open(conll_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line == "":
                    if current_sentence["tokens"]:
                        sentences.append(current_sentence)
                        current_sentence = {"tokens": [], "labels": []}
                else:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        token, label = parts[0], parts[1]
                        current_sentence["tokens"].append(token)
                        current_sentence["labels"].append(label)
        
        # Add last sentence if not empty
        if current_sentence["tokens"]:
            sentences.append(current_sentence)
        
        logger.info(f"Loaded {len(sentences)} sentences")
        
        # Convert to DataFrame
        data = []
        for sentence in sentences:
            data.append({
                "tokens": sentence["tokens"],
                "labels": sentence["labels"],
                "text": " ".join(sentence["tokens"])
            })
        
        df = pd.DataFrame(data)
        
        # Split data
        train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
        
        # Create datasets
        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)
        test_dataset = Dataset.from_pandas(test_df)
        
        return DatasetDict({
            "train": train_dataset,
            "validation": val_dataset,
            "test": test_dataset
        })

    def tokenize_and_align_labels(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        """Tokenize text and align labels with subword tokens.
        
        Args:
            examples: Dictionary with tokens and labels
            
        Returns:
            Tokenized examples with aligned labels
        """
        tokenized_inputs = self.tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        labels = []
        for i, label in enumerate(examples["labels"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(self.label2id.get(label[word_idx], 0))
                else:
                    # Handle subword tokens
                    label_ids.append(-100)
                previous_word_idx = word_idx
            
            labels.append(label_ids)
        
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def compute_metrics(self, eval_pred: EvalPrediction) -> Dict[str, float]:
        """Compute evaluation metrics.
        
        Args:
            eval_pred: Evaluation predictions
            
        Returns:
            Dictionary with metrics
        """
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)
        
        # Remove ignored index (special tokens)
        true_predictions = [
            [self.id2label[p] for (p, l) in zip(pred, label) if l != -100]
            for pred, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.id2label[l] for (p, l) in zip(pred, label) if l != -100]
            for pred, label in zip(predictions, labels)
        ]
        
        # Load seqeval metric
        seqeval_metric = evaluate.load("seqeval")
        results = seqeval_metric.compute(predictions=true_predictions, references=true_labels)
        
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    def train(
        self,
        datasets: DatasetDict,
        training_args: Dict[str, Any] = None,
        model_name_suffix: str = ""
    ) -> Trainer:
        """Train the NER model with comprehensive configuration.
        
        Args:
            datasets: DatasetDict with train/validation splits
            training_args: Training arguments
            model_name_suffix: Suffix for model name
            
        Returns:
            Trained Trainer instance
        """
        # Default training arguments
        default_args = {
            "output_dir": str(self.output_dir / f"{self.model_name.replace('/', '_')}{model_name_suffix}"),
            "num_train_epochs": 5,
            "per_device_train_batch_size": 8,
            "per_device_eval_batch_size": 8,
            "warmup_steps": 500,
            "weight_decay": 0.01,
            "logging_dir": str(self.output_dir / "logs"),
            "logging_steps": 10,
            "evaluation_strategy": "steps",
            "eval_steps": 500,
            "save_steps": 1000,
            "load_best_model_at_end": True,
            "metric_for_best_model": "f1",
            "greater_is_better": True,
            "save_total_limit": 3,
            "fp16": torch.cuda.is_available(),
            "dataloader_num_workers": 4,
            "remove_unused_columns": False,
        }
        
        if training_args:
            default_args.update(training_args)
        
        args = TrainingArguments(**default_args)
        
        # Tokenize datasets
        tokenized_datasets = datasets.map(
            self.tokenize_and_align_labels,
            batched=True,
            remove_columns=datasets["train"].column_names
        )
        
        # Data collator
        data_collator = DataCollatorForTokenClassification(self.tokenizer)
        
        # Callbacks
        callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            callbacks=callbacks,
        )
        
        # Train model
        logger.info("Starting model training...")
        trainer.train()
        
        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_results = trainer.evaluate(tokenized_datasets["test"])
        logger.info(f"Test results: {test_results}")
        
        return trainer

    def save_model(self, trainer: Trainer, model_name: str = None):
        """Save the trained model and tokenizer.
        
        Args:
            trainer: Trained Trainer instance
            model_name: Name for the saved model
        """
        if model_name is None:
            model_name = f"{self.model_name.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        model_path = self.output_dir / model_name
        trainer.save_model(str(model_path))
        self.tokenizer.save_pretrained(str(model_path))
        
        # Save training configuration
        config = {
            "model_name": self.model_name,
            "num_labels": self.num_labels,
            "max_length": self.max_length,
            "label2id": self.label2id,
            "id2label": self.id2label,
            "training_date": datetime.now().isoformat(),
        }
        
        config_file = model_path / "training_config.json"
        import json
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Model saved to {model_path}")
        return str(model_path)

    def generate_training_report(self, trainer: Trainer, test_results: Dict[str, float], model_path: str):
        """Generate comprehensive training report.
        
        Args:
            trainer: Trained Trainer instance
            test_results: Test set evaluation results
            model_path: Path where model was saved
        """
        report_file = Path(model_path) / "training_report.txt"
        
        with open(report_file, 'w') as f:
            f.write("ETHIOMART NER MODEL TRAINING REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Device: {self.device}\n\n")
            
            f.write("TEST SET RESULTS:\n")
            f.write("-" * 20 + "\n")
            for metric, value in test_results.items():
                f.write(f"{metric}: {value:.4f}\n")
            
            f.write(f"\nModel saved to: {model_path}\n")
        
        logger.info(f"Training report saved to {report_file}")


def compare_models(
    model_configs: List[Dict[str, Any]],
    datasets: DatasetDict,
    output_dir: str = "models"
) -> Dict[str, Any]:
    """Compare multiple models and return results.
    
    Args:
        model_configs: List of model configurations
        datasets: DatasetDict with train/validation/test splits
        output_dir: Directory to save models
        
    Returns:
        Dictionary with comparison results
    """
    results = {}
    
    for config in model_configs:
        model_name = config["model_name"]
        logger.info(f"Training model: {model_name}")
        
        trainer = EnhancedNERModelTrainer(
            model_name=model_name,
            output_dir=output_dir,
            num_labels=config.get("num_labels", 9),
            max_length=config.get("max_length", 512)
        )
        
        # Train model
        trained_trainer = trainer.train(datasets, config.get("training_args", {}))
        
        # Evaluate on test set
        test_results = trained_trainer.evaluate()
        
        # Save model
        model_path = trainer.save_model(trained_trainer, f"{model_name.replace('/', '_')}")
        
        # Generate report
        trainer.generate_training_report(trained_trainer, test_results, model_path)
        
        results[model_name] = {
            "test_results": test_results,
            "model_path": model_path,
            "config": config
        }
    
    return results 