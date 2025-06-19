"""
NER (Named Entity Recognition) module for EthioMart pipeline.

This module handles dataset loading, model fine-tuning, and evaluation
for Amharic NER tasks.
"""

from .dataset_loader import NERDatasetLoader
from .model_finetune import NERModelTrainer
from .evaluate import NEREvaluator

__all__ = ["NERDatasetLoader", "NERModelTrainer", "NEREvaluator"] 