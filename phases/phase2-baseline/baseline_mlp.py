"""
Baseline MLP model for vulnerability detection.
Uses CodeBERT embeddings as input features.
"""

import torch
import torch.nn as nn
from typing import List


class BaselineMLP(nn.Module):
    """Feedforward MLP baseline model for vulnerability detection.
    
    Takes CodeBERT CLS token embeddings (768-dim) as input
    and classifies code as vulnerable or safe.
    
    Args:
        input_dim: Input feature dimension. Default 768 (CodeBERT).
        hidden_dims: List of hidden layer sizes.
        dropout: Dropout probability.
        num_classes: Number of output classes.
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dims: List[int] = [512, 256, 128],
        dropout: float = 0.3,
        num_classes: int = 2
    ) -> None:
        super(BaselineMLP, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

        # Xavier initialization
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize weights using Xavier uniform initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim).
        
        Returns:
            Logits tensor of shape (batch_size, num_classes).
        """
        return self.network(x)
