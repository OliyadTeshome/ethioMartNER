"""
Tests for NER module.
"""

import pytest
from src.ner.dataset_loader import NERDatasetLoader


def test_dataset_loader_init():
    """Test NERDatasetLoader initialization."""
    loader = NERDatasetLoader()
    assert loader.label2id["O"] == 0
    assert loader.label2id["B-PRICE"] == 1


def test_label_mapping():
    """Test label mapping functionality."""
    loader = NERDatasetLoader()
    assert loader.id2label[0] == "O"
    assert loader.id2label[1] == "B-PRICE" 