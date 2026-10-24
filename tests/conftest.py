"""Shared test fixtures and import setup.

Two things are handled here.

**Import path.** The repository root is added to ``sys.path`` so that both
``from src.models... import ...`` and ``import app`` work no matter how pytest
was invoked (``pytest tests/`` does not guarantee the root directory is
importable).

**Offline models.** ``SecureScanModel()`` with its default
``codebert_model="microsoft/codebert-base"`` downloads roughly 500MB. Tests must
not do that, so the model fixtures point at a tiny local BERT config instead and
build with ``from_scratch=True``. Same code path, no network, milliseconds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from transformers import BertConfig  # noqa: E402  (needs sys.path set up first)


# Small enough to build and run instantly, but structurally a real BERT: every
# attribute SecureScanModel reaches for (``encoder.layer``, ``config.hidden_size``)
# exists, so the architecture under test is the production one.
TINY_VOCAB_SIZE = 64
TINY_HIDDEN_SIZE = 32
TINY_LAYERS = 2


@pytest.fixture(scope="session")
def tiny_encoder_dir(tmp_path_factory) -> Path:
    """Directory holding a minimal encoder config.

    ``AutoConfig.from_pretrained`` reads this from disk, so no network access is
    needed; ``from_scratch=True`` then skips the weight download entirely.
    """
    directory = tmp_path_factory.mktemp("tiny-codebert")
    config = BertConfig(
        vocab_size=TINY_VOCAB_SIZE,
        hidden_size=TINY_HIDDEN_SIZE,
        num_hidden_layers=TINY_LAYERS,
        num_attention_heads=4,
        intermediate_size=37,
        max_position_embeddings=64,
    )
    config.save_pretrained(directory)
    return directory


@pytest.fixture
def make_tiny_model(tiny_encoder_dir):
    """Factory returning fresh tiny models, for load-into-a-new-instance tests."""
    from src.models.securescan_model import SecureScanModel

    def _make(**overrides):
        kwargs = {
            "codebert_model": str(tiny_encoder_dir),
            "freeze_layers": 0,
            "lstm_hidden": 8,
            "lstm_layers": 1,
            "dropout": 0.0,
            "from_scratch": True,
        }
        kwargs.update(overrides)
        return SecureScanModel(**kwargs)

    return _make


@pytest.fixture
def tiny_model(make_tiny_model):
    """An untrained SecureScanModel backed by the tiny encoder."""
    return make_tiny_model()