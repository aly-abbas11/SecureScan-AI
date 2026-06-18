# Phase 5: Rigorous Evaluation - Ablations & Error Analysis

**Experiment No. 11 | Deadline: May 17, 2026**

## Overview

This phase focused on rigorous evaluation including:
- Final test-set evaluation (single ceremony run)
- Ablation studies (5+ ablations across 3 categories)
- Error analysis with 30-50 wrong predictions
- Robustness analysis under perturbations
- Efficiency reporting

## Final Test Results

| Model | F1 Score | Precision | Recall | Accuracy |
|-------|----------|-----------|--------|----------|
| Baseline MLP | 0.7346 | 0.7123 | 0.7581 | 0.8912 |
| CodeBERT + BiLSTM + MLP | **0.9252** | 0.9184 | 0.9323 | 0.9678 |

## Ablations Performed

1. **Remove component**: Without transfer learning (random init)
2. **Replace component**: LSTM vs GRU comparison
3. **Vary capacity**: Hidden size halved/doubled
4. **Data ablation**: 10%, 25%, 50%, 100% training data
5. **Training recipe**: Without learning rate scheduling

## Links

- [Final Model](../../src/models/securescan_model.py)
- [Results](../../docs/RESULTS.md)
- [Confusion Matrix](../../reports/Week3/confusion_matrix.png)