"""
Tests for data preparation utilities.
"""

import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from prepare_data import clean_text, create_sample_data
from utils import load_jsonl, save_jsonl, count_parameters, format_time


def test_clean_text():
    """Test text cleaning function."""
    # Test whitespace normalization
    text = "This  is   a    test"
    cleaned = clean_text(text)
    assert cleaned == "This is a test"
    
    # Test strip
    text = "  test  "
    cleaned = clean_text(text)
    assert cleaned == "test"


def test_create_sample_data():
    """Test sample data creation."""
    data = create_sample_data()
    
    # Check structure
    assert isinstance(data, list)
    assert len(data) > 0
    
    # Check each item has required fields
    for item in data:
        assert "input" in item
        assert "output" in item
        assert "source" in item
        assert isinstance(item["input"], str)
        assert isinstance(item["output"], str)


def test_format_time():
    """Test time formatting."""
    # Test seconds
    assert format_time(45) == "45s"
    
    # Test minutes
    assert format_time(125) == "2m 5s"
    
    # Test hours
    assert format_time(3725) == "1h 2m 5s"


def test_jsonl_operations(tmp_path):
    """Test JSONL save and load operations."""
    # Create test data
    test_data = [
        {"input": "test1", "output": "answer1"},
        {"input": "test2", "output": "answer2"},
    ]
    
    # Save to temporary file
    test_file = tmp_path / "test.jsonl"
    save_jsonl(test_data, str(test_file))
    
    # Load and verify
    loaded_data = load_jsonl(str(test_file))
    assert len(loaded_data) == len(test_data)
    assert loaded_data[0]["input"] == test_data[0]["input"]
    assert loaded_data[1]["output"] == test_data[1]["output"]


class MockModel:
    """Mock model for testing parameter counting."""
    def __init__(self):
        self.params = [
            MockParam(100, requires_grad=True),
            MockParam(200, requires_grad=True),
            MockParam(50, requires_grad=False),
        ]
    
    def parameters(self):
        return self.params


class MockParam:
    """Mock parameter."""
    def __init__(self, size, requires_grad=True):
        self.size = size
        self.requires_grad = requires_grad
    
    def numel(self):
        return self.size


def test_count_parameters():
    """Test parameter counting."""
    model = MockModel()
    params = count_parameters(model)
    
    assert params["total"] == 350  # 100 + 200 + 50
    assert params["trainable"] == 300  # 100 + 200
    assert params["trainable_percent"] == pytest.approx(85.71, rel=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
