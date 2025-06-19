"""
Tests for data preprocessing module.
"""

import pytest
import pandas as pd
from src.data.preprocess import DataPreprocessor


def test_clean_text():
    """Test text cleaning functionality."""
    preprocessor = DataPreprocessor()
    
    # Test URL removal
    text_with_url = "Check this link https://example.com and buy now"
    cleaned = preprocessor.clean_text(text_with_url)
    assert "https://example.com" not in cleaned
    
    # Test phone number removal
    text_with_phone = "Call +251912345678 for details"
    cleaned = preprocessor.clean_text(text_with_phone)
    assert "+251912345678" not in cleaned


def test_extract_entities():
    """Test entity extraction."""
    preprocessor = DataPreprocessor()
    
    text = "አዲስ ስልክ 5000 ብር ሽያጭ ላይ ነው"
    entities = preprocessor.extract_entities(text)
    
    assert "PRICE" in entities
    assert "PRODUCT" in entities 