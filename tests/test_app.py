"""Tests for the Gradio demo in ``app.py``.

Importing ``app`` must stay free of side effects: the previous version built the
tokenizer and the model at import time, which downloaded ~500MB and crashed when
no checkpoint was present.

Prediction is exercised through a stub tokenizer and stub models, so these tests
need no network access and no trained checkpoint.
"""

from __future__ import annotations

import torch
import pytest

import app
from app import Predictor, parse_args


class StubTokenizer:
    """Stand-in for ``AutoTokenizer``, returning a deterministic encoding.

    ``length`` models the real tokenizer's truncation behaviour: it is capped at
    ``max_length`` whenever a cap is supplied.
    """

    def __init__(self, length: int = 8) -> None:
        self.length = length
        self.calls: list[str] = []

    def __call__(self, code, return_tensors=None, truncation=None, max_length=None):
        self.calls.append(code)
        length = self.length if max_length is None else min(self.length, max_length)
        return {
            "input_ids": torch.zeros(1, length, dtype=torch.long),
            "attention_mask": torch.ones(1, length, dtype=torch.long),
        }


class ConstantLogits(torch.nn.Module):
    """Model stub returning the same logits for every input."""

    def __init__(self, safe: float = 0.0, vulnerable: float = 0.0) -> None:
        super().__init__()
        self.values = (safe, vulnerable)

    def forward(self, input_ids, attention_mask=None):
        base = torch.tensor(self.values, dtype=torch.float32)
        return base.repeat(input_ids.shape[0], 1)


# softmax([2, -2]) puts P(vulnerable) at ~0.018, well inside "Safe".
SAFE_LOGITS = {"safe": 2.0, "vulnerable": -2.0}
VULNERABLE_LOGITS = {"safe": -2.0, "vulnerable": 2.0}


def make_predictor(logits=None, **kwargs) -> Predictor:
    """Build a predictor around a stub model: no checkpoint, no network."""
    logits = SAFE_LOGITS if logits is None else logits
    return Predictor(ConstantLogits(**logits), StubTokenizer(), **kwargs)


def test_importing_app_builds_nothing():
    """The module must not construct a model or tokenizer at import time."""
    assert not hasattr(app, "model")
    assert not hasattr(app, "tokenizer")


def test_rejects_empty_input():
    predictor = make_predictor()
    result = predictor.predict("   \n\t ")
    assert result["error"] == "Please provide some source code."
    assert result["verdict"] is None
    assert predictor.tokenizer.calls == []


def test_rejects_over_long_input_without_tokenizing():
    predictor = make_predictor()
    result = predictor.predict("x" * (app.MAX_CHARS + 1))
    assert "limit is" in result["error"]
    assert predictor.tokenizer.calls == []


def test_clean_snippet_is_reported_safe():
    result = make_predictor().predict("int main() { return 0; }")
    assert result["error"] is None
    assert result["verdict"] == "Safe"
    assert result["p_vulnerable"] < 0.5
    assert result["truncated"] is False
    assert result["confidence"].endswith("%")
    assert "warning" not in result


def test_flagged_snippet_is_reported_vulnerable():
    result = make_predictor(VULNERABLE_LOGITS).predict("strcpy(buffer, input);")
    assert result["verdict"] == "Vulnerable"
    assert result["p_vulnerable"] > 0.5


def test_threshold_override_does_not_mutate_the_predictor():
    """A per-call threshold replaces shared-state mutation, which is unsafe when
    Gradio serves concurrent requests."""
    predictor = make_predictor(SAFE_LOGITS, threshold=0.5)

    flagged = predictor.predict("int x;", threshold=0.01)
    assert flagged["verdict"] == "Vulnerable"
    assert flagged["threshold"] == 0.01

    assert predictor.threshold == 0.5
    assert predictor.predict("int x;")["verdict"] == "Safe"


def test_predict_with_forwards_the_ui_threshold():
    """The Gradio handler must not write onto the shared predictor."""
    predictor = make_predictor(SAFE_LOGITS, threshold=0.5)
    assert app._predict_with(predictor, "int x;", 0.01)["verdict"] == "Vulnerable"
    assert predictor.threshold == 0.5


def test_untrained_predictor_labels_its_output():
    result = make_predictor(trained=False).predict("int x;")
    assert result["error"] is None
    assert "UNTRAINED" in result["warning"]


def test_truncation_is_measured_on_tokens():
    """A long unbroken line must trip the flag: whitespace counting missed it."""
    at_cap = Predictor(ConstantLogits(), StubTokenizer(length=app.MAX_LENGTH))
    assert at_cap.predict("x" * 5000)["truncated"] is True

    below_cap = Predictor(ConstantLogits(), StubTokenizer(length=app.MAX_LENGTH - 1))
    assert below_cap.predict("int x;")["truncated"] is False


def test_model_failure_is_returned_as_an_error():
    class Broken(torch.nn.Module):
        def forward(self, input_ids, attention_mask=None):
            raise RuntimeError("boom")

    result = Predictor(Broken(), StubTokenizer()).predict("int x;")
    assert "Prediction failed" in result["error"]
    assert result["verdict"] is None


def test_unexpected_output_width_is_returned_as_an_error():
    """Indexing a single logit must not escape as an IndexError trace."""

    class OneClass(torch.nn.Module):
        def forward(self, input_ids, attention_mask=None):
            return torch.zeros(input_ids.shape[0], 1)

    result = Predictor(OneClass(), StubTokenizer()).predict("int x;")
    assert "Prediction failed" in result["error"]
    assert result["verdict"] is None


def test_smoke_test_mode_skips_weights_and_files(monkeypatch, tiny_model):
    """``--smoke-test`` must not download weights or read a checkpoint."""
    captured = {}

    def fake_build(model_cls, **kwargs):
        captured.update(kwargs)
        return tiny_model

    def unexpected(*args, **kwargs):
        raise AssertionError("load_checkpoint must not run in smoke-test mode")

    monkeypatch.setattr(app, "build_model", fake_build)
    monkeypatch.setattr(app, "load_checkpoint", unexpected)
    monkeypatch.setattr(
        app.AutoTokenizer, "from_pretrained", lambda *a, **k: StubTokenizer()
    )

    predictor = app.build_predictor(None, smoke_test=True)

    # The encoder is built from its config, so no ~500MB weight download happens.
    assert captured == {"from_scratch": True}
    assert predictor.trained is False


def test_checkpoint_mode_loads_the_requested_path(monkeypatch, tiny_model):
    loaded = []

    monkeypatch.setattr(app, "build_model", lambda *a, **k: tiny_model)
    monkeypatch.setattr(
        app, "load_checkpoint", lambda model, path, **kwargs: loaded.append(str(path))
    )
    monkeypatch.setattr(
        app.AutoTokenizer, "from_pretrained", lambda *a, **k: StubTokenizer()
    )

    checkpoint = "my/own/checkpoint.pt"
    predictor = app.build_predictor(checkpoint)

    assert loaded == [checkpoint]
    assert predictor.trained is True


def test_parse_args_defaults():
    args = parse_args([])
    assert args.checkpoint is None
    assert args.smoke_test is False
    assert args.threshold == app.DEFAULT_THRESHOLD
    assert (args.host, args.port, args.share) == ("127.0.0.1", 7860, False)


def test_parse_args_overrides():
    args = parse_args(
        [
            "--checkpoint",
            "cp.pt",
            "--smoke-test",
            "--threshold",
            "0.7",
            "--port",
            "9000",
        ]
    )
    assert args.checkpoint == "cp.pt"
    assert args.smoke_test is True
    assert args.threshold == pytest.approx(0.7)
    assert args.port == 9000