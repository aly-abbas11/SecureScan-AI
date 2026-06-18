# Phase 2: Foundations - Data, EDA & Baseline Model

**Experiment No. 9 | Deadline: May 4, 2026**

## Overview

This phase focused on:
- Repository setup with proper structure
- Dataset acquisition and versioning
- Exploratory Data Analysis (EDA)
- Baseline MLP model implementation
- Training infrastructure

## Repository Structure (Phase 2)

```
project-root/
├── data/
│   └── notebooks/
├── notebooks/
│   └── securescan-eda.ipynb
├── src/
│   ├── data/loader.py
│   ├── preprocessing/pipeline.py
│   ├── models/baseline_mlp.py
│   ├── training/train.py
│   └── utils/helpers.py
├── experiments/
│   └── config.yaml
└── tests/test_pipeline.py
```

## Key Results

| Metric | Value |
|--------|-------|
| Training samples | 420,240 |
| Validation samples | 89,760 |
| Test samples | 90,000 |
| Class balance strategy | 3:1 (safe:vulnerable) |

## Links

- [EDA Notebook](../../notebooks/securescan-eda.ipynb)
- [Baseline Model](../../src/models/baseline_mlp.py)
- [Training Script](../../src/training/train.py)