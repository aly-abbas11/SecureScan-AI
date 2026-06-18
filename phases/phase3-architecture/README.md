# Phase 3: Core Deep Architecture - CodeBERT + BiLSTM

**Experiment No. 10 | Deadline: May 10, 2026**

## Overview

This phase implemented the main deep learning architecture:
- CodeBERT (microsoft/codebert-base) transformer backbone
- Bidirectional LSTM for sequential modeling
- MLP classifier head
- Training infrastructure upgrades

## Architecture

```
Input → CodeBERT → BiLSTM → MLP → Output
```

### Model Details

| Component | Configuration |
|-----------|---------------|
| CodeBERT | microsoft/codebert-base, freeze first 6 layers |
| BiLSTM | 256 hidden units, 2 layers, bidirectional |
| MLP | 512→256→128→2 with ReLU, BatchNorm, Dropout |

## Regularization Ablation

Three configurations tested:
1. **Unregularized** - No dropout, no weight decay
2. **Regularized** - Dropout 0.3, weight decay 1e-4
3. **Normalized** - BatchNorm added

## Links

- [Main Model](../../src/models/securescan_model.py)
- [Architecture Diagram](../../docs/ARCHITECTURE.md)
- [Week 3 Implementation](../../src/models/week3/codebert_bilstm.py)