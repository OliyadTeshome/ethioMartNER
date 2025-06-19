"""
EthioMart NER - AI-powered Named Entity Recognition pipeline in Amharic.

This package provides a comprehensive NER pipeline for Ethiopian e-commerce data,
including data ingestion, model fine-tuning, evaluation, and vendor analytics.
"""

__version__ = "0.1.0"
__author__ = "EthioMart Team"
__email__ = "team@ethiomart.com"

from . import data
from . import ner
from . import interpretability
from . import vendor

__all__ = ["data", "ner", "interpretability", "vendor"] 