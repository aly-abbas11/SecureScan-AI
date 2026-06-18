"""
Training script for SecureScan AI models.
Handles training loop, validation, early stopping, and checkpointing.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report
from typing import Tuple
import pandas as pd
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import seed_everything, setup_logging, save_checkpoint


# Set seed for reproducibility
seed_everything(42)
logger = setup_logging()


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scheduler=None,
    device: str = 'cuda'
) -> Tuple[float, float]:
    """Run one training epoch.
    
    Args:
        model: PyTorch model to train.
        loader: Training data loader.
        optimizer: Optimizer instance.
        criterion: Loss function.
        scheduler: Learning rate scheduler.
        device: Device to train on.
    
    Returns:
        Tuple of (average_loss, f1_score).
    """
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch_idx, batch in enumerate(loader):
        if len(batch) == 3:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
        else:
            features, labels = batch
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(features)

        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        if (batch_idx + 1) % 100 == 0:
            logger.info(f"  Batch {batch_idx+1}/{len(loader)} "
                       f"— Loss: {loss.item():.4f}")

    f1 = f1_score(all_labels, all_preds, average='macro')
    return total_loss / len(loader), f1


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str = 'cuda'
) -> Tuple[float, float]:
    """Evaluate model on a data loader.
    
    Args:
        model: PyTorch model to evaluate.
        loader: Data loader for evaluation.
        criterion: Loss function.
        device: Device to evaluate on.
    
    Returns:
        Tuple of (average_loss, f1_score).
    """
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                input_ids, attention_mask, labels = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)
                logits = model(input_ids, attention_mask)
            else:
                features, labels = batch
                features = features.to(device)
                labels = labels.to(device)
                logits = model(features)

            loss = criterion(logits, labels)
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    f1 = f1_score(all_labels, all_preds, average='macro')
    return total_loss / len(loader), f1


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scheduler=None,
    epochs: int = 5,
    patience: int = 3,
    checkpoint_path: str = 'checkpoints/best_model.pt',
    device: str = 'cuda'
) -> pd.DataFrame:
    """Full training loop with early stopping and checkpointing.
    
    Args:
        model: PyTorch model to train.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        optimizer: Optimizer instance.
        criterion: Loss function.
        scheduler: Learning rate scheduler.
        epochs: Maximum number of epochs.
        patience: Early stopping patience.
        checkpoint_path: Path to save best model.
        device: Device to train on.
    
    Returns:
        DataFrame with training history.
    """
    best_val_f1 = 0
    patience_count = 0
    history = []

    logger.info("Starting training...")
    logger.info("=" * 50)

    for epoch in range(epochs):
        logger.info(f"\nEpoch {epoch+1}/{epochs}")
        logger.info("-" * 30)

        train_loss, train_f1 = train_epoch(
            model, train_loader, optimizer, criterion, scheduler, device)
        val_loss, val_f1 = evaluate(
            model, val_loader, criterion, device)

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_f1': train_f1,
            'val_loss': val_loss,
            'val_f1': val_f1
        })

        logger.info(f"  Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f}")
        logger.info(f"  Val Loss:   {val_loss:.4f} | Val F1:   {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            save_checkpoint(model, optimizer, epoch, val_f1, checkpoint_path)
            logger.info(f"  ✓ Best model saved! Val F1: {val_f1:.4f}")
            patience_count = 0
        else:
            patience_count += 1
            logger.info(f"  No improvement. Patience: {patience_count}/{patience}")

        if patience_count >= patience:
            logger.info("\nEarly stopping triggered!")
            break

    logger.info(f"\nTraining complete! Best Val F1: {best_val_f1:.4f}")
    return pd.DataFrame(history)


def final_evaluation(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = 'cuda'
) -> None:
    """Run final evaluation on test set — call only once at end.
    
    Args:
        model: Trained PyTorch model.
        test_loader: Test data loader.
        device: Device to evaluate on.
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:
                input_ids, attention_mask, labels = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)
                logits = model(input_ids, attention_mask)
            else:
                features, labels = batch
                features = features.to(device)
                labels = labels.to(device)
                logits = model(features)

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    logger.info("\nFINAL TEST RESULTS:")
    logger.info("=" * 50)
    logger.info(classification_report(
        all_labels, all_preds,
        target_names=['Safe', 'Vulnerable']))
