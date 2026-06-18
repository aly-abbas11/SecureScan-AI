# Architecture

## SecureScan AI System Overview

SecureScan AI is a deep learning system for detecting security vulnerabilities in source code. The architecture combines transformer-based code understanding with sequential modeling.

### Model Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SecureScan AI Model                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input: Source Code (C/C++, Python)                              │
│         ↓                                                        │
│  Tokenizer: CodeBERT tokenizer (BPE)                             │
│         ↓                                                        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              CodeBERT (microsoft/codebert-base)            │   │
│  │  • 12-layer Transformer                                  │   │
│  │  • 768 hidden dimension                                   │   │
│  │  • First 6 layers frozen during training                    │   │
│  └─────────────────────────────────────────────────────────┘   │
│         ↓                                                        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Bidirectional LSTM                           │   │
│  │  • Hidden size: 256                                      │   │
│  │  • 2 layers                                             │   │
│  │  • Bidirectional (512 output)                             │   │
│  │  • Dropout: 0.3                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│         ↓                                                        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              MLP Classifier                             │   │
│  │  • Linear: 512 → 256 → 128 → 2                         │   │
│  │  • ReLU activation                                       │   │
│  │  • BatchNorm & Dropout between layers                    │   │
│  └─────────────────────────────────────────────────────────┘   │
│         ↓                                                        │
│  Output: Probability distribution over [Safe, Vulnerable]          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Repository Structure

```
SecureScan-AI/
├── data/
│   └── notebooks/
│       └── *.ipynb           # Model checkpoints
├── docs/
│   ├── ARCHITECTURE.md       # This file
│   ├── DATA.md               # Dataset documentation
│   ├── DEPLOYMENT.md         # Deployment guide
│   └── RESULTS.md            # Evaluation results
├── experiments/
│   ├── config.yaml           # Training configuration
│   └── training_history.csv    # Metrics log
├── notebooks/
│   ├── Phase4_SecureScan_AI_Final.ipynb  # Main notebook with outputs
│   └── *.ipynb             # Training notebooks
├── reports/
│   ├── Week1-3/            # Analysis reports
│   └── classification_report.txt
├── src/
│   ├── data/
│   │   └── loader.py         # Dataset loading utilities
│   ├── models/
│   │   ├── securescan_model.py    # Main model
│   │   ├── baseline_mlp.py        # Baseline MLP
│   │   └── week3/codebert_bilstm.py # Week 3 implementation
│   ├── preprocessing/
│   │   └── pipeline.py       # Data preprocessing
│   ├── training/
│   │   └── train.py          # Training script
│   └── utils/
│       └── helpers.py        # Utility functions
└── tests/
    └── test_pipeline.py        # Unit tests
```

### Key Components

- **`src/models/securescan_model.py`**: Main model combining CodeBERT + BiLSTM + MLP
- **`src/models/baseline_mlp.py`**: Feedforward baseline using CodeBERT embeddings
- **`src/training/train.py`**: Training loop with early stopping and checkpointing
- **`src/preprocessing/pipeline.py`**: Data preprocessing and tokenization
- **`src/data/loader.py`**: Dataset loading and batching utilities

### Training Configuration

```yaml
# experiments/config.yaml
epochs: 50
batch_size: 16
learning_rate: 2e-5
patience: 3
optimizer: AdamW
scheduler: cosine
```

## Technical Details

### Tokenization

Source code is tokenized using CodeBERT's pretrained tokenizer:
- Byte Pair Encoding (BPE) vocabulary
- Maximum sequence length: 512 tokens
- Special tokens preserved for code structure

### Training Strategy

- **Loss Function**: Cross-entropy with class weights for imbalance
- **Optimizer**: AdamW with weight decay
- **Scheduler**: Cosine annealing with warm restarts
- **Early Stopping**: Patience of 3 epochs on validation F1

### Evaluation Metrics

- Primary: Macro F1-score
- Secondary: Precision, Recall, Accuracy
- Per-class: Safe vs Vulnerable detection rates