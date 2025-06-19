"""
Data processing module for EthioMart NER pipeline.

This module handles data ingestion from Telegram channels and preprocessing
of Amharic text data for NER training.
"""

from .telegram_scraper import TelegramScraper
from .preprocess import DataPreprocessor

__all__ = ["TelegramScraper", "DataPreprocessor"] 