"""
Main SecureScan AI model: CodeBERT + BiLSTM + MLP.
Primary vulnerability detection architecture.
"""

import torch
import torch.nn as nn
from transformers import AutoModel


class SecureScanModel(nn.Module):
    """CodeBERT + Bidirectional LSTM + MLP for vulnerability detection.
    
    Architecture:
        1. CodeBERT: Pretrained transformer for code understanding
        2. BiLSTM: Captures sequential patterns in code
        3. MLP: Final classification head
    
    Args:
        codebert_model: Pretrained model name.
        freeze_layers: Number of CodeBERT layers to freeze.
        lstm_hidden: LSTM hidden size per direction.
        lstm_layers: Number of LSTM layers.
        dropout: Dropout probability.
        num_classes: Number of output classes.
    """

    def __init__(
        self,
        codebert_model: str = "microsoft/codebert-base",
        freeze_layers: int = 6,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 2
    ) -> None:
        super(SecureScanModel, self).__init__()

        # CodeBERT
        self.codebert = AutoModel.from_pretrained(codebert_model)
        for i in range(freeze_layers):
            for param in self.codebert.encoder.layer[i].parameters():
                param.requires_grad = False

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        # MLP Classifier
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            input_ids: Tokenized input IDs.
            attention_mask: Attention mask tensor.
        
        Returns:
            Logits tensor of shape (batch_size, num_classes).
        """
        codebert_out = self.codebert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        lstm_out, _ = self.lstm(codebert_out.last_hidden_state)
        return self.classifier(lstm_out[:, 0, :])
