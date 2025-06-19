"""
NER model fine-tuning module.
"""

import logging
from pathlib import Path
from typing import Dict, Any
import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from datasets import DatasetDict

logger = logging.getLogger(__name__)


class NERModelTrainer:
    """Trainer for NER model fine-tuning."""

    def __init__(
        self,
        model_name: str = "bert-base-multilingual-cased",
        output_dir: str = "models",
        num_labels: int = 9
    ):
        """Initialize the trainer.
        
        Args:
            model_name: Base model name
            output_dir: Directory to save models
            num_labels: Number of NER labels
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_labels = num_labels
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name, num_labels=num_labels
        )

    def train(
        self,
        datasets: DatasetDict,
        training_args: Dict[str, Any] = None
    ) -> Trainer:
        """Train the NER model.
        
        Args:
            datasets: DatasetDict with train/validation splits
            training_args: Training arguments
            
        Returns:
            Trained Trainer instance
        """
        default_args = {
            "output_dir": str(self.output_dir),
            "num_train_epochs": 3,
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
        }
        
        if training_args:
            default_args.update(training_args)
        
        args = TrainingArguments(**default_args)
        
        data_collator = DataCollatorForTokenClassification(self.tokenizer)
        
        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=datasets["train"],
            eval_dataset=datasets["validation"],
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        trainer.train()
        return trainer

    def save_model(self, trainer: Trainer, model_name: str = "ethiomart-ner"):
        """Save the trained model.
        
        Args:
            trainer: Trained Trainer instance
            model_name: Name for the saved model
        """
        model_path = self.output_dir / model_name
        trainer.save_model(str(model_path))
        self.tokenizer.save_pretrained(str(model_path))
        logger.info(f"Model saved to {model_path}") 