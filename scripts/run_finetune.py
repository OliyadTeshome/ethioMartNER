#!/usr/bin/env python3
"""
Model fine-tuning script for EthioMart NER pipeline.
"""

import logging
import click
from pathlib import Path
from src.ner.dataset_loader import NERDatasetLoader
from src.ner.model_finetune import NERModelTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option("--data-dir", "-d", default="data/processed", help="Data directory")
@click.option("--model-name", "-m", default="bert-base-multilingual-cased", help="Base model")
@click.option("--epochs", "-e", default=3, help="Training epochs")
def main(data_dir, model_name, epochs):
    """Run NER model fine-tuning."""
    # Load datasets
    loader = NERDatasetLoader(model_name)
    datasets = loader.load_datasets(data_dir)
    
    # Train model
    trainer = NERModelTrainer(model_name)
    training_args = {"num_train_epochs": epochs}
    
    trained_trainer = trainer.train(datasets, training_args)
    trainer.save_model(trained_trainer)
    
    logger.info("Model training completed")


if __name__ == "__main__":
    main() 