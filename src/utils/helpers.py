"""
Utility functions for SecureScan AI project.
Includes seed setting, logging, and helper functions.
"""

import os
import random
import numpy as np
import torch
import logging


def seed_everything(seed: int = 42) -> None:
    """Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value. Default is 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"✅ Seeds set to {seed}")


def setup_logging(log_file: str = "experiments/training.log") -> logging.Logger:
    """Set up logging to both console and file.
    
    Args:
        log_file: Path to log file.
    
    Returns:
        Configured logger instance.
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def count_parameters(model: torch.nn.Module) -> dict:
    """Count total and trainable parameters in a model.
    
    Args:
        model: PyTorch model.
    
    Returns:
        Dictionary with total and trainable parameter counts.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total': total,
        'trainable': trainable,
        'frozen': total - trainable
    }


def save_checkpoint(model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    epoch: int,
                    val_f1: float,
                    path: str) -> None:
    """Save model checkpoint.
    
    Args:
        model: PyTorch model to save.
        optimizer: Optimizer state to save.
        epoch: Current epoch number.
        val_f1: Validation F1 score.
        path: Path to save checkpoint.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_f1': val_f1,
    }, path)
    print(f"✅ Checkpoint saved to {path}")


def load_checkpoint(model: torch.nn.Module,
                    path: str,
                    optimizer: torch.optim.Optimizer = None):
    """Load model checkpoint.
    
    Args:
        model: PyTorch model to load weights into.
        path: Path to checkpoint file.
        optimizer: Optional optimizer to restore state.
    
    Returns:
        Tuple of (model, optimizer, epoch, val_f1)
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"✅ Checkpoint loaded from {path}")
    return model, optimizer, checkpoint['epoch'], checkpoint['val_f1']
