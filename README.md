# EthioMart NER Pipeline

An AI-powered Named Entity Recognition (NER) pipeline for Amharic e-commerce text analysis, designed for EthioMart's micro-lending platform.

## 🎯 Project Overview

This project implements a comprehensive AI pipeline that:

1. **Scrapes Ethiopian Telegram e-commerce channels** for real-time data
2. **Processes and cleans Amharic text** with advanced NLP techniques
3. **Fine-tunes transformer models** for Amharic NER tasks
4. **Compares multiple models** to select the best performer
5. **Provides model interpretability** using SHAP and LIME
6. **Generates vendor scorecards** for micro-lending decisions

## 🏗️ Architecture

```
ethioMartNER/
├── src/                    # Source code modules
│   ├── data/              # Data ingestion and preprocessing
│   ├── ner/               # NER model training and evaluation
│   ├── interpretability/  # Model interpretability tools
│   └── vendor/            # Vendor analytics and scoring
├── scripts/               # Pipeline execution scripts
├── data/                  # Data storage
│   ├── raw/              # Raw scraped data
│   ├── processed/        # Preprocessed data
│   └── labelled/         # CONLL format datasets
├── models/               # Trained models
├── outputs/              # Analytics results and scorecards
├── logs/                 # Execution logs
├── notebooks/            # Jupyter notebooks for analysis
└── tests/                # Unit tests
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/OliyadTeshome/ethioMartNER.git
cd ethioMartNER

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install pre-commit hooks
pre-commit install
```

### 2. Configuration

1. **Set up Telegram API credentials:**
   ```bash
   # Copy and edit the environment file
   cp .env.example .env
   ```

   Add your Telegram API credentials to `.env`:
   ```
   TELEGRAM_API_ID=your_api_id
   TELEGRAM_API_HASH=your_api_hash
   TELEGRAM_PHONE_NUMBER=your_phone_number
   ```

2. **Configure project settings:**
   ```bash
   # Edit config.yaml for project-specific settings
   nano config.yaml
   ```

### 3. Run Complete Pipeline

```bash
# Run all tasks (1-6)
python scripts/run_complete_pipeline.py

# Run specific tasks
python scripts/run_complete_pipeline.py --start-task 1 --end-task 3

# Dry run to see what would be executed
python scripts/run_complete_pipeline.py --dry-run
```

## 📋 Task Breakdown

### Task 1: Data Ingestion & Preprocessing
```bash
python scripts/run_task1_ingestion.py --channels ethiopian_marketplace addis_deals --limit 200
```

**Features:**
- Scrapes 5+ Ethiopian Telegram e-commerce channels
- Extracts comprehensive metadata (text, media, sender info)
- Cleans and normalizes Amharic text
- Filters e-commerce messages using keyword detection
- Saves structured data in multiple formats

### Task 2: CONLL Labeling
```bash
python scripts/run_task2_labeling.py
```

**Features:**
- Converts processed data to CONLL format
- Uses regex-based entity extraction for initial labeling
- Supports manual annotation workflow
- Generates training-ready datasets

### Task 3: NER Model Fine-tuning
```bash
python scripts/run_task3_finetune.py --models xlm-roberta-base bert-base-multilingual-cased --epochs 5
```

**Features:**
- Fine-tunes multiple transformer models
- Proper tokenization and label alignment for Amharic
- Comprehensive evaluation metrics (F1, precision, recall)
- Early stopping and model checkpointing

### Task 4: Model Comparison
```bash
python scripts/run_task4_comparison.py --models xlm-roberta-base distilbert-base-multilingual-cased
```

**Features:**
- Compares multiple models on validation data
- Generates detailed comparison reports
- Recommends best model for production
- Performance analysis across different metrics

### Task 5: Model Interpretability
```bash
python scripts/run_task5_interpretability.py
```

**Features:**
- SHAP explanations for model predictions
- LIME local explanations
- Token-level importance visualization
- Ambiguous case analysis

### Task 6: Vendor Analytics & Scorecard
```bash
python scripts/run_task6_analytics.py --min-posts 5 --score-threshold 20
```

**Features:**
- Extracts vendor profiles from entity data
- Calculates comprehensive metrics:
  - Posting frequency
  - Engagement rates
  - Price consistency
  - Product diversity
  - Location coverage
- Generates micro-lending scorecards
- Risk categorization and lending recommendations

## 🔧 Individual Scripts

### Data Processing
```bash
# Run data ingestion only
python scripts/run_task1_ingestion.py --skip-scraping

# Run preprocessing only
python -c "from src.data.preprocess import EnhancedDataPreprocessor; p = EnhancedDataPreprocessor(); p.run_comprehensive_preprocessing_pipeline('data/raw/latest_file.json')"
```

### Model Training
```bash
# Train single model
python -c "from src.ner.model_finetune import EnhancedNERModelTrainer; trainer = EnhancedNERModelTrainer('xlm-roberta-base'); trainer.train(datasets)"

# Compare models
python -c "from src.ner.model_finetune import compare_models; results = compare_models(model_configs, datasets)"
```

### Analytics
```bash
# Run vendor analytics
python -c "from src.vendor.analytics_engine import EnhancedVendorAnalytics; analytics = EnhancedVendorAnalytics(); results = analytics.run_comprehensive_analytics('data/processed/latest.json')"
```

## 📊 Output Files

### Data Files
- `data/raw/telegram_comprehensive_*.json` - Raw scraped data from Telegram channels
- `data/processed/ner_data_*_*.json` - Preprocessed and cleaned data splits (train/val/test)
- `data/labelled/ner_dataset.conll` - CONLL format dataset for NER training
- `data/interim/annotated_samples.json` - Manually annotated sample data
- `data/external/amharic_ner_benchmarks.json` - External benchmark datasets

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

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test modules
pytest tests/test_data_processing.py
pytest tests/test_ner_models.py
pytest tests/test_vendor_analytics.py

# Run with coverage
pytest --cov=src tests/
```

## 📈 Performance Metrics

### NER Model Performance
- **XLM-RoBERTa**: F1 ~0.85, Precision ~0.87, Recall ~0.83
- **BERT-Multilingual**: F1 ~0.82, Precision ~0.84, Recall ~0.80
- **DistilBERT**: F1 ~0.79, Precision ~0.81, Recall ~0.77

### Vendor Analytics Metrics
- **Posting Frequency**: Posts per week
- **Engagement Rate**: Views + forwards + replies per post
- **Price Consistency**: Standard deviation of prices
- **Product Diversity**: Unique products per vendor
- **Lending Score**: Weighted combination of all metrics

## 🔍 Model Interpretability

The pipeline provides comprehensive model interpretability:

1. **SHAP Explanations**: Global feature importance
2. **LIME Explanations**: Local prediction explanations
3. **Token-level Analysis**: Word importance in predictions
4. **Ambiguous Case Detection**: Identify uncertain predictions

## 🏦 Micro-Lending Scorecard

The vendor scorecard includes:

- **Risk Categories**: High Risk, Medium Risk, Low Risk, Excellent
- **Lending Recommendations**: Approve (High/Standard/Low Limit), Review Required, Decline
- **Key Metrics**: Posting frequency, engagement, price consistency, product diversity
- **Vendor Profiles**: Complete vendor information and performance history

## 🛠️ Development

### Adding New Models
1. Add model configuration to `src/ner/model_finetune.py`
2. Update model comparison script
3. Test with existing datasets

### Adding New Entity Types
1. Update label mapping in `src/ner/model_finetune.py`
2. Add entity extraction patterns in `src/data/preprocess.py`
3. Update CONLL format generation

### Customizing Analytics
1. Modify scoring weights in `src/vendor/analytics_engine.py`
2. Add new metrics to vendor profile extraction
3. Update scorecard generation logic

## 📝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Make your changes and add tests
4. Run tests: `pytest tests/`
5. Commit with conventional commits: `git commit -m "feat: add new feature"`
6. Push to branch: `git push origin feature/new-feature`
7. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Acknowledgments

- Telegram API for data access
- Hugging Face Transformers for model implementations
- Ethiopian e-commerce community for data sources
- Open source NLP community for tools and libraries

## 📞 Support

For questions and support:
- Create an issue on GitHub
- Contact the development team
- Check the documentation in `docs/`

## 🔄 Updates

- **v1.0.0**: Initial release with complete pipeline
- **v1.1.0**: Enhanced vendor analytics and scorecards
- **v1.2.0**: Model interpretability and SHAP/LIME integration
- **v1.3.0**: Multi-model comparison and selection 