"""Tests for :mod:`src.models.loader`.

The checkpoint validator exists because a ``BaselineMLP`` checkpoint used to be
loaded into a ``SecureScanModel`` with ``strict=False``. The two architectures
share no parameter names, so every weight was discarded and the model still
returned a confident probability. ``test_baseline_checkpoint_is_rejected`` is
the regression test for that.
"""

from __future__ import annotations

import torch
import pytest
from transformers import AutoConfig

from src.models import loader
from src.models.baseline_mlp import BaselineMLP
from src.models.loader import (
    CheckpointError,
    build_model,
    diagnose_state_dict,
    extract_state_dict,
    load_checkpoint,
    resolve_checkpoint,
    save_checkpoint,
)
from src.models.securescan_model import SecureScanModel


# --------------------------------------------------------------------------- #
# resolve_checkpoint
# --------------------------------------------------------------------------- #

def test_resolve_checkpoint_accepts_absolute_path(tmp_path):
    path = tmp_path / "model.pt"
    path.write_bytes(b"")
    assert resolve_checkpoint(path) == path


def test_resolve_checkpoint_searches_the_legacy_directory(tmp_path, monkeypatch):
    """A bare filename still resolves against the historical ``src/models``."""
    legacy = tmp_path / "src" / "models"
    legacy.mkdir(parents=True)
    target = legacy / "best_model.pt"
    target.write_bytes(b"")

    monkeypatch.setattr(loader, "PROJECT_ROOT", tmp_path / "absent")
    monkeypatch.setattr(loader, "DEFAULT_CHECKPOINT_DIR", tmp_path / "checkpoints")
    monkeypatch.setattr(loader, "LEGACY_CHECKPOINT_DIR", legacy)

    assert resolve_checkpoint("best_model.pt") == target.resolve()


def test_resolve_checkpoint_ignores_the_working_directory(tmp_path, monkeypatch):
    """Lookup is anchored to the project, not to wherever the script started."""
    project = tmp_path / "project"
    (project / "checkpoints").mkdir(parents=True)
    target = project / "checkpoints" / "best_model.pt"
    target.write_bytes(b"")

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "best_model.pt").write_bytes(b"a copy in the cwd")

    monkeypatch.setattr(loader, "PROJECT_ROOT", project)
    monkeypatch.setattr(loader, "DEFAULT_CHECKPOINT_DIR", project / "checkpoints")
    monkeypatch.setattr(loader, "LEGACY_CHECKPOINT_DIR", project / "src" / "models")
    monkeypatch.chdir(elsewhere)

    assert resolve_checkpoint("best_model.pt") == target.resolve()


def test_resolve_checkpoint_reports_every_location_searched(tmp_path, monkeypatch):
    monkeypatch.setattr(loader, "PROJECT_ROOT", tmp_path / "a")
    monkeypatch.setattr(loader, "DEFAULT_CHECKPOINT_DIR", tmp_path / "b")
    monkeypatch.setattr(loader, "LEGACY_CHECKPOINT_DIR", tmp_path / "c")

    with pytest.raises(FileNotFoundError) as excinfo:
        resolve_checkpoint("nope.pt")

    message = str(excinfo.value)
    assert "not found" in message
    for location in ("a", "b", "c"):
        assert str(tmp_path / location) in message


# --------------------------------------------------------------------------- #
# extract_state_dict
# --------------------------------------------------------------------------- #

def test_extract_state_dict_passes_through_a_bare_state_dict(tiny_model):
    state = tiny_model.state_dict()
    assert extract_state_dict(state) is state


@pytest.mark.parametrize("key", ["model_state_dict", "state_dict"])
def test_extract_state_dict_unwraps_provenance_wrappers(tiny_model, key):
    """The wrapper written by ``src.utils.helpers.save_checkpoint``."""
    state = tiny_model.state_dict()
    assert extract_state_dict({key: state, "epoch": 3}) is state


def test_extract_state_dict_accepts_a_whole_module(tiny_model):
    extracted = extract_state_dict({"model": tiny_model})
    assert set(extracted) == set(tiny_model.state_dict())


def test_extract_state_dict_rejects_unknown_formats():
    with pytest.raises(CheckpointError, match="Unrecognised checkpoint format"):
        extract_state_dict({"epoch": 3, "val_f1": 0.9})


# --------------------------------------------------------------------------- #
# diagnose_state_dict
# --------------------------------------------------------------------------- #

def test_diagnose_reports_missing_and_unexpected_keys(tiny_model):
    state = tiny_model.state_dict()
    dropped = next(iter(state))
    partial = {k: v for k, v in state.items() if k != dropped}
    partial["classifier.9.weight"] = torch.zeros(2)

    usable, missing, unexpected, shape_mismatch = diagnose_state_dict(tiny_model, partial)

    assert missing == [dropped]
    assert unexpected == ["classifier.9.weight"]
    assert shape_mismatch == []
    assert len(usable) == len(state) - 1


def test_diagnose_rejects_shape_mismatches(tiny_model):
    state = tiny_model.state_dict()
    name = next(n for n, t in state.items() if t.dim() >= 2)
    state[name] = torch.zeros((1,) + tuple(state[name].shape[1:]))

    usable, missing, unexpected, shape_mismatch = diagnose_state_dict(tiny_model, state)

    assert missing == [] and unexpected == []
    assert len(shape_mismatch) == 1 and name in shape_mismatch[0]
    assert name not in usable


def test_diagnose_strips_data_parallel_prefix(tiny_model):
    """A checkpoint saved through ``DataParallel`` maps onto the bare model."""
    own = tiny_model.state_dict()
    wrapped = {f"module.{name}": tensor for name, tensor in own.items()}

    usable, missing, unexpected, shape_mismatch = diagnose_state_dict(tiny_model, wrapped)

    assert (missing, unexpected, shape_mismatch) == ([], [], [])
    assert len(usable) == len(own)


# --------------------------------------------------------------------------- #
# save_checkpoint / load_checkpoint
# --------------------------------------------------------------------------- #

def test_save_checkpoint_round_trips_every_parameter(make_tiny_model, tmp_path):
    source = make_tiny_model()
    original = {k: v.clone() for k, v in source.state_dict().items()}
    destination = tmp_path / "nested" / "best_model.pt"

    written = save_checkpoint(source, destination, epoch=4, metrics={"val_f1": 0.9})

    assert written == destination.resolve()
    assert written.is_file()

    payload = torch.load(written, map_location="cpu", weights_only=True)
    assert payload["epoch"] == 4
    assert payload["metrics"] == {"val_f1": 0.9}

    target = make_tiny_model()
    load_checkpoint(target, written)
    for name, tensor in target.state_dict().items():
        assert torch.equal(tensor, original[name]), name


def test_load_checkpoint_restores_from_a_helpers_wrapper(make_tiny_model, tmp_path):
    """``src.utils.helpers.save_checkpoint`` writes ``model_state_dict``."""
    source = make_tiny_model()
    torch.save(
        {
            "epoch": 1,
            "model_state_dict": source.state_dict(),
            "optimizer_state_dict": {},
            "val_f1": 0.5,
        },
        tmp_path / "wrapped.pt",
    )

    target = make_tiny_model()
    load_checkpoint(target, tmp_path / "wrapped.pt")
    expected = source.state_dict()
    for name, tensor in target.state_dict().items():
        assert torch.equal(tensor, expected[name]), name


def test_baseline_checkpoint_is_rejected(make_tiny_model, tmp_path):
    """The bug this module exists to prevent.

    ``BaselineMLP`` and ``SecureScanModel`` share no parameter names, so the old
    ``strict=False`` load discarded everything and left the classifier head
    random, while the app kept reporting confident predictions.
    """
    baseline_path = save_checkpoint(BaselineMLP(), tmp_path / "baseline.pt")
    target = make_tiny_model()
    before = {k: v.clone() for k, v in target.state_dict().items()}

    with pytest.raises(CheckpointError) as excinfo:
        load_checkpoint(target, baseline_path)

    message = str(excinfo.value)
    assert "does not fit" in message
    assert "coverage" in message
    # Refusing must mean refusing: nothing may be applied on the way out.
    for name, tensor in target.state_dict().items():
        assert torch.equal(tensor, before[name]), name


def test_partial_load_must_be_requested_explicitly(tiny_model, tmp_path):
    """``strict=False`` alone used to still demand 100% coverage."""
    state = dict(tiny_model.state_dict())
    del state[next(iter(state))]
    torch.save({"model_state_dict": state}, tmp_path / "partial.pt")

    with pytest.warns(RuntimeWarning, match="Partial load"):
        load_checkpoint(tiny_model, tmp_path / "partial.pt", strict=False)


def test_min_coverage_below_the_threshold_is_rejected(tiny_model, tmp_path):
    state = dict(tiny_model.state_dict())
    del state[next(iter(state))]
    torch.save({"model_state_dict": state}, tmp_path / "partial.pt")

    with pytest.raises(CheckpointError, match="covers only"):
        load_checkpoint(
            tiny_model, tmp_path / "partial.pt", strict=False, min_coverage=1.0
        )


def test_min_coverage_outside_unit_range_is_rejected(tiny_model):
    """Argument validation happens before any file access."""
    with pytest.raises(ValueError, match="min_coverage"):
        load_checkpoint(tiny_model, "does-not-exist.pt", min_coverage=1.5)


# --------------------------------------------------------------------------- #
# build_model / from_config
# --------------------------------------------------------------------------- #

def test_build_model_ignores_unknown_config_keys():
    model = build_model(
        BaselineMLP, input_dim=16, hidden_dims=[4], dropout=0.0, not_a_kwarg=1
    )
    assert isinstance(model, BaselineMLP)


def test_build_model_applies_aliases(tiny_encoder_dir):
    # The default encoder is 768 wide, so this only holds if the aliased key
    # actually reached the constructor.
    expected_width = AutoConfig.from_pretrained(str(tiny_encoder_dir)).hidden_size

    model = build_model(
        SecureScanModel,
        aliases={"codebert": "codebert_model"},
        codebert=str(tiny_encoder_dir),
        freeze_layers=0,  # the tiny encoder has fewer layers than the default 6
        from_scratch=True,
        unknown_key="ignored",
    )

    assert isinstance(model, SecureScanModel)
    assert model.codebert.config.hidden_size == expected_width


def test_from_config_accepts_the_config_file_spelling(tiny_encoder_dir):
    """``phases/phase4-refinement/config.yaml`` spells the encoder ``codebert``."""
    expected_width = AutoConfig.from_pretrained(str(tiny_encoder_dir)).hidden_size

    model = SecureScanModel.from_config(
        {
            "name": "SecureScanModel",  # unknown key, must be ignored
            "codebert": str(tiny_encoder_dir),
            "freeze_layers": 1,
            "lstm_hidden": 8,
            "lstm_layers": 1,
            "dropout": 0.0,
            "from_scratch": True,
        }
    )

    assert model.codebert.config.hidden_size == expected_width
    assert all(
        not p.requires_grad for p in model.codebert.encoder.layer[0].parameters()
    )