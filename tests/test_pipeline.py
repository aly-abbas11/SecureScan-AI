"""Smoke tests for preprocessing pipeline."""
import torch
import pytest

def test_model_output_shape():
    """Test model outputs correct shape."""
    from src.models.securescan_model import SecureScanModel
    model = SecureScanModel()
    dummy_ids = torch.randint(0, 1000, (2, 512))
    dummy_mask = torch.ones(2, 512, dtype=torch.long)
    with torch.no_grad():
        output = model(dummy_ids, dummy_mask)
    assert output.shape == (2, 2)

def test_baseline_mlp_shape():
    """Test baseline MLP outputs correct shape."""
    from src.models.baseline_mlp import BaselineMLP
    model = BaselineMLP()
    dummy_input = torch.randn(4, 768)
    output = model(dummy_input)
    assert output.shape == (4, 2)
