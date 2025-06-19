"""
Tests for Telegram scraper module.
"""

import pytest
from src.data.telegram_scraper import TelegramScraper


def test_telegram_scraper_init():
    """Test TelegramScraper initialization."""
    # This is a basic test - in real implementation you'd mock the API credentials
    with pytest.raises(ValueError):
        TelegramScraper(api_id=None, api_hash=None, phone_number=None)


def test_filter_ecommerce_messages():
    """Test e-commerce message filtering."""
    scraper = TelegramScraper(api_id="test", api_hash="test", phone_number="test")
    
    # Mock data
    import pandas as pd
    test_data = pd.DataFrame({
        "text": [
            "ሽያጭ አዲስ ስልክ 5000 ብር",
            "Hello world",
            "ዋጋ ቅናሽ አለ"
        ]
    })
    
    filtered = scraper.filter_ecommerce_messages(test_data)
    assert len(filtered) == 2  # Should filter out "Hello world" 