#!/usr/bin/env python3
"""
Task 3: Fine-tune Amharic NER Model Script

This script runs the complete NER model fine-tuning pipeline:
1. Load CONLL format data
2. Fine-tune multiple transformer models
3. Evaluate performance
4. Save best model and generate reports
"""

import logging
import click
from pathlib import Path
import pandas as pd
from datetime import datetime
import sys
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from ner.model_finetune import EnhancedNERModelTrainer, compare_models
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/task3_finetune.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@click.command()
@click.option("--conll-file", "-c", default="data/labelled/ner_dataset.conll", 
              help="Path to CONLL format dataset")
@click.option("--models", "-m", multiple=True, 
              default=["xlm-roberta-base", "bert-base-multilingual-cased"],
              help="Models to fine-tune")
@click.option("--epochs", "-e", default=5, help="Number of training epochs")
@click.option("--batch-size", "-b", default=8, help="Batch size for training")
@click.option("--max-length", "-l", default=512, help="Maximum sequence length")
@click.option("--compare-only", is_flag=True, help="Compare models without individual training")
def main(conll_file, models, epochs, batch_size, max_length, compare_only):
    """Run Task 3: Complete NER model fine-tuning pipeline."""
    
    logger.info("=" * 60)
    logger.info("TASK 3: FINE-TUNE AMHARIC NER MODEL")
    logger.info("=" * 60)
    
    # Check if CONLL file exists
    conll_path = Path(conll_file)
    if not conll_path.exists():
        logger.error(f"CONLL file not found: {conll_file}")
        logger.info("Please run Task 1 first to generate the CONLL dataset.")
        return
    
    # Step 1: Load and prepare data
    logger.info("Step 1: Loading CONLL dataset...")
    trainer = EnhancedNERModelTrainer(
        model_name=models[0],  # Use first model for data loading
        max_length=max_length
    )
    
    datasets = trainer.load_conll_data(str(conll_path))
    logger.info(f"✓ Loaded dataset with {len(datasets['train'])} train, {len(datasets['validation'])} validation, {len(datasets['test'])} test samples")
    
    # Step 2: Model comparison
    if len(models) > 1 or compare_only:
        logger.info("\nStep 2: Comparing multiple models...")
        
        # Define model configurations
        model_configs = []
        for model_name in models:
            config = {
                "model_name": model_name,
                "num_labels": 9,
                "max_length": max_length,
                "training_args": {
                    "num_train_epochs": epochs,
                    "per_device_train_batch_size": batch_size,
                    "per_device_eval_batch_size": batch_size,
                    "warmup_steps": 500,
                    "weight_decay": 0.01,
                    "learning_rate": 2e-5,
                    "save_total_limit": 3,
                    "load_best_model_at_end": True,
                    "metric_for_best_model": "f1",
                    "greater_is_better": True,
                }
            }
            model_configs.append(config)
        
        # Compare models
        comparison_results = compare_models(model_configs, datasets, "models")
        
        # Generate comparison report
        generate_comparison_report(comparison_results)
        
        # Select best model
        best_model = select_best_model(comparison_results)
        logger.info(f"✓ Best model: {best_model}")
        
    else:
        # Step 2: Single model training
        logger.info("\nStep 2: Training single model...")
        
        model_name = models[0]
        trainer = EnhancedNERModelTrainer(
            model_name=model_name,
            max_length=max_length
        )
        
        # Training arguments
        training_args = {
            "num_train_epochs": epochs,
            "per_device_train_batch_size": batch_size,
            "per_device_eval_batch_size": batch_size,
            "warmup_steps": 500,
            "weight_decay": 0.01,
            "learning_rate": 2e-5,
            "save_total_limit": 3,
            "load_best_model_at_end": True,
            "metric_for_best_model": "f1",
            "greater_is_better": True,
        }
        
        # Train model
        trained_trainer = trainer.train(datasets, training_args)
        
        # Evaluate on test set
        test_results = trained_trainer.evaluate()
        
        # Save model
        model_path = trainer.save_model(trained_trainer)
        
        # Generate report
        trainer.generate_training_report(trained_trainer, test_results, model_path)
        
        logger.info(f"✓ Model training completed. Results: {test_results}")
    
    logger.info("\n" + "=" * 60)
    logger.info("TASK 3 COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)
    logger.info("Next steps:")
    logger.info("1. Review training reports in models/ directory")
    logger.info("2. Run Task 4: Model comparison and selection")
    logger.info("3. Run Task 5: Model interpretability analysis")


def generate_comparison_report(comparison_results: dict):
    """Generate comprehensive model comparison report.
    
    Args:
        comparison_results: Dictionary with model comparison results
    """
    report_file = Path("models") / f"model_comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    with open(report_file, 'w') as f:
        f.write("ETHIOMART NER MODEL COMPARISON REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("MODEL COMPARISON RESULTS:\n")
        f.write("-" * 30 + "\n\n")
        
        # Create comparison table
        f.write(f"{'Model':<30} {'F1':<8} {'Precision':<10} {'Recall':<8} {'Accuracy':<10}\n")
        f.write("-" * 70 + "\n")
        
        for model_name, results in comparison_results.items():
            test_results = results["test_results"]
            f.write(f"{model_name:<30} {test_results['f1']:<8.4f} {test_results['precision']:<10.4f} "
                   f"{test_results['recall']:<8.4f} {test_results['accuracy']:<10.4f}\n")
        
        f.write("\n" + "-" * 70 + "\n\n")
        
        # Detailed results
        for model_name, results in comparison_results.items():
            f.write(f"\n{model_name.upper()}:\n")
            f.write("-" * len(model_name) + "\n")
            f.write(f"Model Path: {results['model_path']}\n")
            f.write("Test Results:\n")
            for metric, value in results["test_results"].items():
                f.write(f"  {metric}: {value:.4f}\n")
            f.write(f"Config: {json.dumps(results['config'], indent=2)}\n")
    
    logger.info(f"✓ Comparison report saved to {report_file}")


def select_best_model(comparison_results: dict) -> str:
    """Select the best performing model based on F1 score.
    
    Args:
        comparison_results: Dictionary with model comparison results
        
    Returns:
        Name of the best model
    """
    best_model = None
    best_f1 = -1
    
    for model_name, results in comparison_results.items():
        f1_score = results["test_results"]["f1"]
        if f1_score > best_f1:
            best_f1 = f1_score
            best_model = model_name
    
    logger.info(f"Best model selected: {best_model} (F1: {best_f1:.4f})")
    return best_model


if __name__ == "__main__":
    main() 