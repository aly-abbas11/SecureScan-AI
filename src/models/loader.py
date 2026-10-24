"""
Checkpoint loading utilities for SecureScan AI.

This module exists because the original demo loaded a ``BaselineMLP``
checkpoint into a ``SecureScanModel`` with ``strict=False``.  None of the
parameter names overlap between those two architectures, so *every* weight was
silently discarded and the classifier head ran on random initialisation while
still reporting a confident probability.

Loading is therefore treated as a validated operation: a checkpoint that does
not cover the model's parameters is an error, not a warning.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Tuple, Type

import torch
import torch.nn as nn

# Repository root, resolved from this file rather than the current working
# directory so that imports and relative asset paths behave identically no
# matter where a script is invoked from.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Canonical on-disk locations.  ``*.pt`` is gitignored, so these files are
# produced by training or fetched from a GitHub release rather than committed.
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
LEGACY_CHECKPOINT_DIR = PROJECT_ROOT / "src" / "models"


class CheckpointError(RuntimeError):
    """Raised when a checkpoint cannot be loaded into a model safely."""


def resolve_checkpoint(path: str | os.PathLike[str]) -> Path:
    """Resolve a checkpoint path, tolerating the legacy ``src/models`` layout.

    Args:
        path: Absolute path, or a path relative to the project root.

    Returns:
        The resolved absolute path.

    Raises:
        FileNotFoundError: If no candidate location holds the checkpoint.
    """
    candidate = Path(path)
    if candidate.is_absolute() and candidate.is_file():
        return candidate

    searched = []
    for base in (PROJECT_ROOT, DEFAULT_CHECKPOINT_DIR, LEGACY_CHECKPOINT_DIR):
        resolved = (base / candidate).resolve()
        if resolved.is_file():
            return resolved
        searched.append(resolved)

    raise FileNotFoundError(
        f"Checkpoint {str(path)!r} not found. Searched:\n"
        + "\n".join(f"  - {p}" for p in searched)
        + "\n\nModel weights are not committed to git (*.pt is gitignored). "
        "Train a model first (see README 'Run Training'), or download a "
        "released checkpoint into checkpoints/."
    )


def extract_state_dict(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    """Pull a ``state_dict`` out of a raw or wrapped checkpoint.

    Handles the three shapes produced across this project's history: a bare
    ``state_dict``, the ``{"model_state_dict": ...}`` wrapper written by
    :func:`src.utils.helpers.save_checkpoint`, and ``{"state_dict": ...}``.

    Args:
        checkpoint: Object loaded from a ``.pt`` file.

    Returns:
        A mapping of parameter name to tensor.

    Raises:
        CheckpointError: If no recognisable state dict is present.
    """
    if isinstance(checkpoint, nn.Module):
        return checkpoint.state_dict()

    if isinstance(checkpoint, Mapping):
        for key in ("model_state_dict", "state_dict", "model"):
            inner = checkpoint.get(key)
            if isinstance(inner, nn.Module):
                return inner.state_dict()
            if isinstance(inner, Mapping):
                return inner
        # A bare state dict: every value is a tensor.
        if checkpoint and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            return checkpoint

    raise CheckpointError(
        "Unrecognised checkpoint format. Expected a state_dict, or a mapping "
        "containing one of 'model_state_dict', 'state_dict', or 'model'."
    )


def _normalise(module: nn.Module) -> nn.Module:
    """Strip a ``DataParallel``/``DistributedDataParallel`` ``module.`` prefix."""
    return module.module if hasattr(module, "module") else module


def diagnose_state_dict(
    model: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], list[str], list[str], list[str]]:
    """Compare a checkpoint against a model's parameters.

    Args:
        model: The model the checkpoint is intended for.
        state_dict: Candidate parameters, possibly carrying a ``module.`` prefix.

    Returns:
        Tuple of ``(usable, missing, unexpected, shape_mismatch)`` where
        ``usable`` holds only the entries that match by name *and* shape, and
        the three lists hold human-readable names of everything rejected.
    """
    target = _normalise(model)
    own = target.state_dict()

    # Normalise the incoming keys once, so a DataParallel checkpoint still maps.
    incoming = {
        (k[len("module.") :] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }

    usable: Dict[str, torch.Tensor] = {}
    missing: list[str] = []
    shape_mismatch: list[str] = []

    for name, tensor in own.items():
        if name not in incoming:
            missing.append(name)
            continue
        candidate = incoming[name]
        if not isinstance(candidate, torch.Tensor):
            shape_mismatch.append(f"{name} (not a tensor)")
            continue
        if tuple(candidate.shape) != tuple(tensor.shape):
            shape_mismatch.append(
                f"{name} (checkpoint {tuple(candidate.shape)} "
                f"!= model {tuple(tensor.shape)})"
            )
            continue
        usable[name] = candidate

    unexpected = [name for name in incoming if name not in own]

    return usable, missing, unexpected, shape_mismatch


def load_checkpoint(
    model: nn.Module,
    path: str | os.PathLike[str],
    *,
    strict: bool = True,
    min_coverage: float | None = None,
    map_location: str | torch.device = "cpu",
) -> nn.Module:
    """Load a checkpoint into ``model``, validating that it actually applies.

    Args:
        model: Model to load weights into, modified in place.
        path: Checkpoint location; see :func:`resolve_checkpoint`.
        strict: If True, require full coverage and no shape mismatches.
        min_coverage: Minimum fraction of model parameters the checkpoint must
            supply. Defaults to 1.0 when ``strict`` is True and 0.0 otherwise,
            so a deliberately partial load does not have to restate the
            threshold. An explicit value is always honoured.
        map_location: Device mapping passed to :func:`torch.load`.

    Returns:
        The same ``model`` instance, for convenient chaining.

    Raises:
        ValueError: If ``min_coverage`` is outside ``[0, 1]``.
        FileNotFoundError: If the checkpoint cannot be located.
        CheckpointError: If the checkpoint is missing parameters, contains
            mismatched shapes, or falls below ``min_coverage``. The message
            lists the offending keys so the mismatch is diagnosable.

    Warns:
        RuntimeWarning: When a partial load was explicitly permitted and some
            parameters are therefore left at their initial values.
    """
    if min_coverage is None:
        min_coverage = 1.0 if strict else 0.0
    if not 0.0 <= min_coverage <= 1.0:
        raise ValueError(
            f"min_coverage={min_coverage} must be within [0, 1]. Use 1.0 to "
            "require a complete load, or strict=False to permit a partial one."
        )

    resolved = resolve_checkpoint(path)

    # weights_only=True prevents arbitrary code execution from a pickle. This
    # is also the default from PyTorch 2.6 onward, where omitting it breaks.
    checkpoint = torch.load(resolved, map_location=map_location, weights_only=True)
    state_dict = extract_state_dict(checkpoint)

    usable, missing, unexpected, shape_mismatch = diagnose_state_dict(model, state_dict)

    total = len(_normalise(model).state_dict())
    coverage = len(usable) / total if total else 0.0

    if strict and (missing or shape_mismatch):
        details = []
        if missing:
            details.append(
                f"{len(missing)} parameter(s) absent from the checkpoint, e.g. "
                + ", ".join(missing[:5])
            )
        if shape_mismatch:
            details.append(
                f"{len(shape_mismatch)} shape mismatch(es), e.g. "
                + ", ".join(shape_mismatch[:3])
            )
        raise CheckpointError(
            f"Checkpoint {resolved.name!r} does not fit the model "
            f"({coverage:.1%} coverage).\n  "
            + "\n  ".join(details)
            + "\n\nThis usually means the checkpoint was saved from a different "
            "architecture (for example a BaselineMLP checkpoint loaded into "
            "SecureScanModel). Refusing to load, because a partial load leaves "
            "randomly-initialised layers that still produce confident output."
        )

    if coverage < min_coverage:
        raise CheckpointError(
            f"Checkpoint {resolved.name!r} covers only {coverage:.1%} of model "
            f"parameters, below the required {min_coverage:.1%}."
        )

    if missing or shape_mismatch:
        warnings.warn(
            f"Partial load of {resolved.name!r}: {len(missing)} parameter(s) "
            f"and {len(shape_mismatch)} shape mismatch(es) were left at their "
            "initial values, so output from this model is not trustworthy.",
            RuntimeWarning,
            stacklevel=2,
        )

    _normalise(model).load_state_dict(usable, strict=False)
    return model


def save_checkpoint(
    model: nn.Module,
    path: str | os.PathLike[str],
    *,
    epoch: int | None = None,
    metrics: Mapping[str, float] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Save a checkpoint in the wrapper format this project reads back.

    Args:
        model: Model whose weights are written.
        path: Destination path; parent directories are created.
        epoch: Optional epoch number for provenance.
        metrics: Optional validation metrics for provenance.
        optimizer: Optional optimizer state, restored by the training loop.
        extra: Optional additional provenance (config, git SHA, dataset hash).

    Returns:
        The path written to.
    """
    destination = Path(path)
    if not destination.is_absolute():
        destination = (PROJECT_ROOT / destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    payload: MutableMapping[str, Any] = {
        "model_state_dict": _normalise(model).state_dict(),
    }
    if epoch is not None:
        payload["epoch"] = epoch
    if metrics is not None:
        payload["metrics"] = dict(metrics)
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if extra is not None:
        payload.update(extra)

    torch.save(payload, destination)
    return destination


def build_model(
    model_cls: Type[nn.Module],
    *,
    aliases: Mapping[str, str] | None = None,
    **kwargs: Any,
) -> nn.Module:
    """Instantiate ``model_cls``, forwarding only the kwargs it accepts.

    Lets callers pass a full config dict without each model class having to
    declare every field.

    Args:
        model_cls: Model class to instantiate.
        aliases: Optional ``{config_key: constructor_arg}`` renaming applied
            before the constructor signature is consulted. Without it a config
            key that only *nearly* matches is dropped silently and the default
            is used instead — ``config.yaml`` in this repository spells the
            encoder ``codebert`` while the argument is ``codebert_model``.
        **kwargs: Candidate constructor arguments; unknown keys are ignored.

    Returns:
        An instance of ``model_cls``.
    """
    import inspect

    renamed = dict(kwargs)
    for source, target in (aliases or {}).items():
        if source in renamed and target not in renamed:
            renamed[target] = renamed.pop(source)

    accepted = set(inspect.signature(model_cls.__init__).parameters) - {"self"}
    return model_cls(**{k: v for k, v in renamed.items() if k in accepted})
