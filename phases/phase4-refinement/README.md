# Phase 4: Refinement - Transfer Learning, VAE & Hyperparameter Search

**Experiment No. 11 | Deadline: May 17, 2026**

## Overview

This phase focused on three concurrent workstreams:
1. Transfer learning with different freezing strategies
2. VAE generative component for vulnerability detection
3. Hyperparameter optimization with Optuna

## Transfer Learning Configurations

| Config | Description |
|--------|-------------|
| Feature Extraction | Freeze entire CodeBERT, train only head |
| Full Fine-tuning | Unfreeze all layers |
| Differential LR | Different learning rates for backbone vs head |

## Generative Component (VAE)

- Variational Autoencoder for anomaly detection
- Latent space interpolation samples
- Reconstruction + KL divergence loss

## Hyperparameter Search

- **Tool**: Optuna with TPE sampler
- **Trials**: 20+
- **Tuned params**: learning rate, batch size, dropout, weight decay, LSTM hidden size

## Links

- [Main Notebook](../../notebooks/Phase4_SecureScan_AI_Final.ipynb)
- [Optuna Study](https://github.com/Salman1122334411/SecureScan-AI/tree/main/experiments)
- [VAE Results](../../reports/Week1/vae_results_final.png)