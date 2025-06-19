# EthioMart NER Pipeline

An AI-powered Named Entity Recognition (NER) pipeline in Amharic for Ethiopian e-commerce data aggregation.

## 🎯 Project Overview

EthioMart NER is a comprehensive pipeline designed to extract structured information from Ethiopian e-commerce data sources, particularly Telegram channels. The system identifies and categorizes entities such as prices, phone numbers, locations, and products in Amharic text.

## 🏗️ Project Structure

```
ethioMartNER/
├── .gitignore                 # Git ignore patterns
├── .env                       # Environment variables (create from env.template)
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies
├── pyproject.toml            # Project configuration
├── config.yaml               # Pipeline configuration
├── telegram_config.json      # Telegram API configuration
├── data/                     # Data storage
│   ├── raw/                  # Raw scraped data
│   └── processed/            # Processed data for training
├── notebooks/                # Jupyter notebooks for each task
│   ├── 1_data_ingestion.ipynb
│   ├── 2_conll_labeling.ipynb
│   ├── 3_finetune_ner_model.ipynb
│   ├── 4_model_comparison.ipynb
│   ├── 5_model_interpretability.ipynb
│   └── 6_vendor_scorecard.ipynb
├── src/                      # Source code
│   ├── __init__.py
│   ├── data/                 # Data processing modules
│   │   ├── telegram_scraper.py
│   │   └── preprocess.py
│   ├── ner/                  # NER model modules
│   │   ├── dataset_loader.py
│   │   ├── model_finetune.py
│   │   └── evaluate.py
│   ├── interpretability/     # Model interpretability
│   │   └── shap_lime_explain.py
│   └── vendor/               # Vendor analytics
│       └── analytics_engine.py
├── scripts/                  # Command-line scripts
│   ├── run_ingestion.py
│   ├── run_finetune.py
│   └── generate_scorecard.py
├── tests/                    # Unit tests
│   ├── test_scraper.py
│   ├── test_preprocess.py
│   ├── test_ner.py
│   └── test_scorecard.py
└── logs/                     # Log files
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd ethioMartNER

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp env.template .env

# Edit .env with your API credentials
# - TELEGRAM_API_ID
# - TELEGRAM_API_HASH
# - TELEGRAM_BOT_TOKEN
# - TELEGRAM_PHONE_NUMBER
```

### 3. Data Ingestion

```bash
# Run data ingestion from Telegram channels
python scripts/run_ingestion.py --channels ethiopian_marketplace addis_deals --limit 100
```

### 4. Model Training

```bash
# Run NER model fine-tuning
python scripts/run_finetune.py --epochs 5
```

### 5. Vendor Analytics

```bash
# Generate vendor scorecards
python scripts/generate_scorecard.py --data-file data/processed/ecommerce_data.json
```

## 📋 Tasks Overview

### Task 1: Data Ingestion
- Scrape Ethiopian e-commerce data from Telegram channels
- Filter relevant messages using Amharic keywords
- Store raw data in structured format

### Task 2: CONLL Labeling
- Convert raw text to CONLL format
- Implement entity extraction using regex patterns
- Prepare training data for NER models

### Task 3: NER Model Fine-tuning
- Fine-tune multilingual BERT models for Amharic NER
- Implement custom training pipeline
- Save and version trained models

### Task 4: Model Comparison
- Compare different NER models and architectures
- Evaluate performance using standard metrics
- Generate comprehensive evaluation reports

### Task 5: Model Interpretability
- Implement SHAP and LIME explanations
- Analyze model decision-making process
- Generate interpretability reports

### Task 6: Vendor Analytics
- Analyze vendor performance metrics
- Generate vendor scorecards
- Create business intelligence dashboards

## 🔧 Configuration

### Model Configuration (`config.yaml`)
- Base model selection (BERT, RoBERTa, etc.)
- Training hyperparameters
- Data split ratios
- NER label definitions

### Telegram Configuration (`telegram_config.json`)
- API credentials
- Target channels
- Scraping parameters

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test module
pytest tests/test_scraper.py

# Run with coverage
pytest --cov=src
```

## 📊 Notebooks

Each task has a corresponding Jupyter notebook in the `notebooks/` directory:

1. **1_data_ingestion.ipynb**: Data collection and exploration
2. **2_conll_labeling.ipynb**: Data preprocessing and labeling
3. **3_finetune_ner_model.ipynb**: Model training and validation
4. **4_model_comparison.ipynb**: Model evaluation and comparison
5. **5_model_interpretability.ipynb**: SHAP and LIME analysis
6. **6_vendor_scorecard.ipynb**: Vendor analytics and reporting

## 🤖 NER Labels

The system recognizes the following entity types:
- **PRICE**: Product prices in Ethiopian Birr
- **PHONE**: Contact phone numbers
- **LOCATION**: Geographic locations and addresses
- **PRODUCT**: Product names and descriptions

## 📈 Performance Metrics

- **F1-Score**: Overall model performance
- **Precision**: Accuracy of positive predictions
- **Recall**: Coverage of actual entities
- **Entity-level metrics**: Per-entity type performance

## 🔒 Security

- API credentials stored in `.env` file (not committed to Git)
- Rate limiting for API calls
- Secure session management for Telegram API

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Ethiopian e-commerce community
- Hugging Face Transformers library
- Telegram API for data access
- Amharic language processing community

## 📞 Support

For questions and support, please open an issue on GitHub or contact the development team.

---

**Note**: This is a research and development project. Please ensure compliance with Telegram's Terms of Service and respect user privacy when scraping data. 