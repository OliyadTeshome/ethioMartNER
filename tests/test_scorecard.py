"""
Tests for vendor analytics module.
"""

import pytest
import pandas as pd
from src.vendor.analytics_engine import VendorAnalytics


def test_vendor_analytics_init():
    """Test VendorAnalytics initialization."""
    analytics = VendorAnalytics()
    assert analytics.vendor_data == {}


def test_analyze_vendor_performance():
    """Test vendor performance analysis."""
    analytics = VendorAnalytics()
    
    # Mock data
    test_data = pd.DataFrame({
        "vendor_id": ["v1", "v2"],
        "price": [1000, 2000],
        "product": ["phone", "laptop"]
    })
    
    result = analytics.analyze_vendor_performance(test_data)
    assert "total_vendors" in result
    assert "avg_price" in result 