"""Smoke tests for the model architectures.

The models are built from a local, tiny encoder config (see ``conftest.py``) so
these tests never download pretrained weights and stay fast enough for CI.
"""
import torch
import pytest


def test_model_output_shape(tiny_model):
    """Test model outputs correct shape."""
    dummy_ids = torch.randint(0, 64, (2, 32))
    dummy_mask = torch.ones(2, 32, dtype=torch.long)
    with torch.no_grad():
        output = tiny_model(dummy_ids, dummy_mask)
    assert output.shape == (2, 2)


def test_baseline_mlp_shape():
    """Test baseline MLP outputs correct shape."""
    from src.models.baseline_mlp import BaselineMLP
    model = BaselineMLP()
    dummy_input = torch.randn(4, 768)
    output = model(dummy_input)
    assert output.shape == (4, 2)


def test_freeze_layers_freezes_only_the_leading_layers(make_tiny_model):
    """``freeze_layers`` freezes exactly the first N encoder layers."""
    model = make_tiny_model(freeze_layers=1)

    frozen = model.codebert.encoder.layer[0]
    trainable = model.codebert.encoder.layer[1]
    assert all(not p.requires_grad for p in frozen.parameters())
    assert all(p.requires_grad for p in trainable.parameters())


def test_freeze_layers_above_layer_count_is_rejected(make_tiny_model):
    """A freeze count larger than the encoder must fail loudly, not silently."""
    with pytest.raises(ValueError, match="freeze_layers"):
        make_tiny_model(freeze_layers=3)
