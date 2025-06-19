#!/usr/bin/env python3
"""
Complete EthioMart NER Pipeline Runner

This master script runs the complete AI pipeline from data ingestion to vendor analytics:
Task 1: Amharic Data Ingestion & Preprocessing
Task 2: Label Data in CoNLL Format (semi-automated)
Task 3: Fine-tune Amharic NER Model
Task 4: Compare & Select Best NER Model
Task 5: Interpret Model with SHAP/LIME
Task 6: Fintech Vendor Scorecard for Micro-Lending
"""

import logging
import click
from pathlib import Path
import sys
import subprocess
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/complete_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@click.command()
@click.option("--start-task", "-s", default=1, type=int, help="Task to start from (1-6)")
@click.option("--end-task", "-e", default=6, type=int, help="Task to end at (1-6)")
@click.option("--skip-tasks", "-k", multiple=True, type=int, help="Tasks to skip")
@click.option("--channels", "-c", multiple=True, 
              default=["ethiopian_marketplace", "addis_deals", "ethio_shopping", "addis_tech_market", "ethiopia_electronics"],
              help="Telegram channels to scrape")
@click.option("--models", "-m", multiple=True, 
              default=["xlm-roberta-base", "bert-base-multilingual-cased"],
              help="Models to compare for NER")
@click.option("--epochs", default=5, help="Training epochs for NER models")
@click.option("--batch-size", default=8, help="Batch size for training")
@click.option("--dry-run", is_flag=True, help="Show what would be run without executing")
def main(start_task, end_task, skip_tasks, channels, models, epochs, batch_size, dry_run):
    """Run the complete EthioMart NER pipeline."""
    
    logger.info("=" * 80)
    logger.info("ETHIOMART NER PIPELINE - COMPLETE RUN")
    logger.info("=" * 80)
    logger.info(f"Start Task: {start_task}")
    logger.info(f"End Task: {end_task}")
    logger.info(f"Skip Tasks: {skip_tasks}")
    logger.info(f"Channels: {channels}")
    logger.info(f"Models: {models}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Batch Size: {batch_size}")
    logger.info(f"Dry Run: {dry_run}")
    logger.info("=" * 80)
    
    # Define task configurations
    tasks = {
        1: {
            "name": "Data Ingestion & Preprocessing",
            "script": "scripts/run_task1_ingestion.py",
            "args": ["--channels"] + list(channels) + ["--limit", "200", "--days", "30"],
            "description": "Scrape Telegram channels and preprocess Amharic text"
        },
        2: {
            "name": "CONLL Labeling",
            "script": "scripts/run_task2_labeling.py",
            "args": [],
            "description": "Convert data to CONLL format for NER training"
        },
        3: {
            "name": "NER Model Fine-tuning",
            "script": "scripts/run_task3_finetune.py",
            "args": ["--models"] + list(models) + ["--epochs", str(epochs), "--batch-size", str(batch_size)],
            "description": "Fine-tune transformer models for Amharic NER"
        },
        4: {
            "name": "Model Comparison",
            "script": "scripts/run_task4_comparison.py",
            "args": ["--models"] + list(models),
            "description": "Compare different NER models and select best"
        },
        5: {
            "name": "Model Interpretability",
            "script": "scripts/run_task5_interpretability.py",
            "args": [],
            "description": "Generate SHAP and LIME explanations"
        },
        6: {
            "name": "Vendor Analytics & Scorecard",
            "script": "scripts/run_task6_analytics.py",
            "args": ["--min-posts", "5", "--score-threshold", "20"],
            "description": "Generate vendor scorecards for micro-lending"
        }
    }
    
    # Run tasks
    for task_num in range(start_task, end_task + 1):
        if task_num in skip_tasks:
            logger.info(f"Skipping Task {task_num}: {tasks[task_num]['name']}")
            continue
        
        task_config = tasks[task_num]
        logger.info(f"\n{'='*60}")
        logger.info(f"TASK {task_num}: {task_config['name']}")
        logger.info(f"{'='*60}")
        logger.info(f"Description: {task_config['description']}")
        
        if dry_run:
            logger.info(f"DRY RUN - Would execute: python {task_config['script']} {' '.join(task_config['args'])}")
            continue
        
        # Check if script exists
        script_path = Path(task_config['script'])
        if not script_path.exists():
            logger.error(f"Script not found: {script_path}")
            logger.info("Please ensure all task scripts are available.")
            return
        
        # Execute task
        try:
            logger.info(f"Executing: python {task_config['script']} {' '.join(task_config['args'])}")
            
            # Run the script
            result = subprocess.run(
                [sys.executable, task_config['script']] + task_config['args'],
                capture_output=True,
                text=True,
                cwd=Path.cwd()
            )
            
            if result.returncode == 0:
                logger.info(f"✓ Task {task_num} completed successfully")
                if result.stdout:
                    logger.info("Output:")
                    logger.info(result.stdout)
            else:
                logger.error(f"✗ Task {task_num} failed with return code {result.returncode}")
                if result.stderr:
                    logger.error("Error output:")
                    logger.error(result.stderr)
                if result.stdout:
                    logger.info("Standard output:")
                    logger.info(result.stdout)
                
                # Ask user if they want to continue
                if not click.confirm(f"Task {task_num} failed. Continue with next task?"):
                    logger.info("Pipeline stopped by user.")
                    return
                    
        except Exception as e:
            logger.error(f"✗ Task {task_num} failed with exception: {e}")
            if not click.confirm(f"Task {task_num} failed. Continue with next task?"):
                logger.info("Pipeline stopped by user.")
                return
    
    logger.info(f"\n{'='*80}")
    logger.info("PIPELINE COMPLETED!")
    logger.info(f"{'='*80}")
    
    if not dry_run:
        generate_pipeline_summary()


def generate_pipeline_summary():
    """Generate a summary of the pipeline execution."""
    logger.info("Generating pipeline summary...")
    
    summary_file = Path("outputs") / f"pipeline_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(summary_file, 'w') as f:
        f.write("ETHIOMART NER PIPELINE EXECUTION SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Execution Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("GENERATED FILES:\n")
        f.write("-" * 20 + "\n")
        
        # Check for generated files
        directories = {
            "Raw Data": "data/raw",
            "Processed Data": "data/processed", 
            "Labelled Data": "data/labelled",
            "Models": "models",
            "Outputs": "outputs",
            "Logs": "logs"
        }
        
        for name, path in directories.items():
            dir_path = Path(path)
            if dir_path.exists():
                files = list(dir_path.glob("*"))
                f.write(f"\n{name} ({path}):\n")
                for file in files[:10]:  # Show first 10 files
                    f.write(f"  - {file.name}\n")
                if len(files) > 10:
                    f.write(f"  ... and {len(files) - 10} more files\n")
        
        f.write("\nNEXT STEPS:\n")
        f.write("-" * 15 + "\n")
        f.write("1. Review generated models in models/ directory\n")
        f.write("2. Check vendor scorecards in outputs/ directory\n")
        f.write("3. Analyze logs in logs/ directory for any issues\n")
        f.write("4. Consider running individual tasks for fine-tuning\n")
        f.write("5. Deploy best model for production use\n")
    
    logger.info(f"✓ Pipeline summary saved to {summary_file}")


if __name__ == "__main__":
    main() 