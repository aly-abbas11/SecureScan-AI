"""
Main SecureScan AI model: CodeBERT + BiLSTM + MLP.

Primary vulnerability detection architecture.
"""

from __future__ import annotations

from typing import Any, Dict, cast

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class SecureScanModel(nn.Module):
    """CodeBERT + Bidirectional LSTM + MLP for vulnerability detection.

    Architecture:
        1. CodeBERT: Pretrained transformer for code understanding
        2. BiLSTM: Captures sequential patterns in code
        3. MLP: Final classification head

    Args:
        codebert_model: Pretrained model name, or a path to a local directory.
        freeze_layers: Number of leading CodeBERT layers to freeze.
        lstm_hidden: LSTM hidden size per direction.
        lstm_layers: Number of LSTM layers.
        dropout: Dropout probability.
        num_classes: Number of output classes.
        from_scratch: Build a randomly-initialised encoder from the model's
            published config instead of downloading its weights. Intended for
            tests and CI, where pulling ~500MB per run is not acceptable. The
            resulting model is architecturally identical but untrained.

    Raises:
        ValueError: If ``freeze_layers`` exceeds the encoder's layer count.
    """

    def __init__(
        self,
        codebert_model: str = "microsoft/codebert-base",
        freeze_layers: int = 6,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 2,
        from_scratch: bool = False,
    ) -> None:
        super().__init__()

        # Encoder. ``from_scratch`` reuses the published config but skips the
        # weight download, which keeps the parameter shapes identical.
        if from_scratch:
            config = AutoConfig.from_pretrained(codebert_model)
            self.codebert = AutoModel.from_config(config)
        else:
            self.codebert = AutoModel.from_pretrained(codebert_model)

        layer_count = len(self.codebert.encoder.layer)
        if not 0 <= freeze_layers <= layer_count:
            raise ValueError(
                f"freeze_layers={freeze_layers} is out of range for an encoder "
                f"with {layer_count} layers."
            )
        for i in range(freeze_layers):
            for param in self.codebert.encoder.layer[i].parameters():
                param.requires_grad = False

        # Read the encoder width from its config rather than hard-coding 768,
        # so the model still builds against a different checkpoint.
        encoder_dim = self.codebert.config.hidden_size

        self.lstm = nn.LSTM(
            input_size=encoder_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            input_ids: Tokenized input IDs, shape ``(batch, seq_len)``.
            attention_mask: Attention mask tensor, same shape.

        Returns:
            Logits of shape ``(batch, num_classes)``.
        """
        codebert_out = self.codebert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        lstm_out, _ = self.lstm(codebert_out.last_hidden_state)
        # The first position carries the [CLS] representation, which the
        # frozen-layer scheme in this project was trained against. Note that
        # for a bidirectional LSTM this is *not* a pooled summary of the
        # sequence; changing it would invalidate existing checkpoints.
        return self.classifier(lstm_out[:, 0, :])

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SecureScanModel":
        """Build a model from a config mapping, ignoring unknown keys.

        Args:
            config: Mapping of constructor arguments, e.g. the ``model`` section
                of ``phases/phase4-refinement/config.yaml``.

        Returns:
            An initialised model.

        Note:
            ``codebert`` is accepted as an alias for ``codebert_model``, because
            that is how this project's config files spell it. Without the alias
            the key would be dropped and the encoder would silently fall back to
            the default.
        """
        # Imported here rather than at module scope so this module stays
        # importable on its own, and so no import cycle can form later.
        from src.models.loader import build_model

        return cast(
            "SecureScanModel",
            build_model(cls, aliases={"codebert": "codebert_model"}, **config),
        )
