
import torch
import torch.nn as nn

from transformers import AutoModel


class SecureSenseModel(nn.Module):

    def __init__(
        self,
        model_name="microsoft/codebert-base",
        hidden_size=256,
        dropout=0.3,
        num_classes=2
    ):

        super().__init__()

        self.codebert = AutoModel.from_pretrained(model_name)

        self.bilstm = nn.LSTM(
            input_size=768,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(
            hidden_size * 2,
            num_classes
        )

    def forward(self, input_ids, attention_mask):

        outputs = self.codebert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        sequence_output = outputs.last_hidden_state

        lstm_output, _ = self.bilstm(sequence_output)

        final_output = lstm_output[:, -1, :]

        final_output = self.dropout(final_output)

        logits = self.classifier(final_output)

        return logits
