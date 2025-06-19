# EthioMart NER Pipeline - Setup Guide

This guide will help you set up and run the complete EthioMart NER pipeline for Amharic e-commerce text analysis.

## 🚀 Quick Start

### 1. Prerequisites

- **Python 3.8+** (recommended: Python 3.9 or 3.10)
- **Git** for cloning the repository
- **Telegram API credentials** (see section below)
- **8GB+ RAM** (16GB recommended for model training)
- **GPU** (optional but recommended for faster training)

### 2. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/OliyadTeshome/ethioMartNER.git
cd ethioMartNER

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install pre-commit hooks
pre-commit install
```

### 3. Telegram API Setup

1. **Get Telegram API credentials:**
   - Go to https://my.telegram.org/
   - Log in with your phone number
   - Go to "API Development Tools"
   - Create a new application
   - Note down your `api_id` and `api_hash`

2. **Configure environment:**
   ```bash
   # Copy environment template
   cp .env.example .env
   
   # Edit .env file with your credentials
   nano .env
   ```

   Add your credentials to `.env`:
   ```
   TELEGRAM_API_ID=your_api_id_here
   TELEGRAM_API_HASH=your_api_hash_here
   TELEGRAM_PHONE_NUMBER=your_phone_number_here
   ```

### 4. Configuration

Edit `config.yaml` to customize pipeline settings:

```yaml
# Data settings
data:
  max_length: 512
  batch_size: 8
  test_size: 0.2
  val_size: 0.1

# Model settings
model:
  base_model: "xlm-roberta-base"
  learning_rate: 2e-5
  num_epochs: 5
  warmup_steps: 500

# Telegram settings
telegram:
  channels:
    - "ethiopian_marketplace"
    - "addis_deals"
    - "ethio_shopping"
  limit_per_channel: 200
  days_back: 30
```

## 🏃‍♂️ Running the Pipeline

### Option 1: Complete Pipeline (Recommended)

Run all tasks from start to finish:

```bash
# Run complete pipeline
python scripts/run_complete_pipeline.py

# Run with custom parameters
python scripts/run_complete_pipeline.py \
  --channels ethiopian_marketplace addis_deals \
  --models xlm-roberta-base bert-base-multilingual-cased \
  --epochs 5 \
  --batch-size 8
```

### Option 2: Individual Tasks

Run tasks individually for more control:

```bash
# Task 1: Data Ingestion
python scripts/run_task1_ingestion.py --channels ethiopian_marketplace addis_deals

# Task 2: CONLL Labeling
python scripts/run_task2_labeling.py

# Task 3: Model Fine-tuning
python scripts/run_task3_finetune.py --models xlm-roberta-base --epochs 5

# Task 6: Vendor Analytics
python scripts/run_task6_analytics.py --min-posts 5
```

### Option 3: Dry Run

See what would be executed without running:

```bash
python scripts/run_complete_pipeline.py --dry-run
```

## 📊 Expected Outputs

After running the pipeline, you'll find:

### Data Files
- `data/raw/telegram_comprehensive_*.json` - Raw scraped data
- `data/processed/ner_data_*_*.json` - Preprocessed data splits
- `data/labelled/ner_dataset.conll` - CONLL format dataset

### Models
- `models/xlm-roberta-base_*/` - Trained model checkpoints
- `models/model_comparison_report_*.txt` - Model comparison results

### Analytics
- `outputs/vendor_scorecard_*.csv` - Vendor scorecards
- `outputs/lending_recommendations.json` - Lending decisions
- `outputs/summary_statistics.json` - Analytics summary

### Logs
- `logs/task1_ingestion.log` - Data ingestion logs
- `logs/task3_finetune.log` - Model training logs
- `logs/complete_pipeline.log` - Complete pipeline logs

## 🔧 Troubleshooting

### Common Issues

1. **Telegram API Error:**
   ```
   Error: No module named 'telethon'
   ```
   **Solution:** Install telethon: `pip install telethon`

2. **CUDA/GPU Issues:**
   ```
   RuntimeError: CUDA out of memory
   ```
   **Solution:** Reduce batch size in config or use CPU: `--device cpu`

3. **Memory Issues:**
   ```
   MemoryError: Unable to allocate array
   ```
   **Solution:** Reduce max_length or batch_size in config

4. **Import Errors:**
   ```
   ModuleNotFoundError: No module named 'src'
   ```
   **Solution:** Ensure you're in the project root directory

### Performance Optimization

1. **For faster training:**
   - Use GPU if available
   - Increase batch size (if memory allows)
   - Reduce max_length for shorter sequences

2. **For better accuracy:**
   - Increase epochs
   - Use larger models (e.g., xlm-roberta-large)
   - Add more training data

3. **For memory efficiency:**
   - Use gradient accumulation
   - Enable mixed precision training
   - Use smaller models (e.g., distilbert)

## 🧪 Testing

Run tests to ensure everything is working:

```bash
# Run all tests
pytest tests/

# Run specific test modules
pytest tests/test_data_processing.py
pytest tests/test_ner_models.py

# Run with coverage
pytest --cov=src tests/
```

## 📈 Monitoring

### Logs
Check logs in the `logs/` directory for detailed execution information.

### Progress Tracking
The pipeline provides progress updates for each task:
- Data ingestion progress
- Model training epochs
- Evaluation metrics
- Analytics generation

### Performance Metrics
Expected performance ranges:
- **XLM-RoBERTa**: F1 ~0.85, Precision ~0.87, Recall ~0.83
- **BERT-Multilingual**: F1 ~0.82, Precision ~0.84, Recall ~0.80
- **DistilBERT**: F1 ~0.79, Precision ~0.81, Recall ~0.77

## 🔄 Updating

To update the pipeline:

```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade

# Reinstall pre-commit hooks
pre-commit install
```

## 📞 Support

If you encounter issues:

1. **Check the logs** in `logs/` directory
2. **Review the README** for detailed documentation
3. **Create an issue** on GitHub with:
   - Error message
   - System information
   - Steps to reproduce
   - Log files

## 🎯 Next Steps

After successful setup:

1. **Review generated data** in `data/` directory
2. **Check model performance** in `models/` directory
3. **Analyze vendor scorecards** in `outputs/` directory
4. **Customize parameters** in `config.yaml` for your use case
5. **Deploy best model** for production use

## 📚 Additional Resources

- [Telegram API Documentation](https://core.telegram.org/api)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [Amharic NLP Resources](https://github.com/amharic-nlp)
- [NER Best Practices](https://github.com/microsoft/nlp-recipes) 