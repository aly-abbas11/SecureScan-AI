# Data Card — SecureScan AI

## Overview

SecureScan AI is trained on a combined dataset of 749,794 code samples for vulnerability detection. This document describes the datasets used, their sources, and the preprocessing pipeline.

## Datasets

### Dataset 1: BigVul

| Property | Value |
|----------|-----|
| Source | [MSR Data on Kaggle](https://www.kaggle.com/datasets/kaggler10240/msr-data) |
| License | Unknown |
| Size | 1.47 GB |
| Samples | 188,636 |
| Vulnerable | 10,900 |
| Safe | 177,736 |
| Language | C/C++ |

**Known Issues:**
- Significant class imbalance (94% safe samples)
- Real-world CVE code with natural distribution

### Dataset 2: DiverseVul

| Property | Value |
|----------|-----|
| Source | [DiverseVul on Kaggle](https://www.kaggle.com/datasets/ahmedtabib/diversevul-cleaned) |
| License | Unknown |
| Size | 110 MB |
| Samples | 319,523 |
| Vulnerable | 17,879 |
| Safe | 301,644 |
| Language | C/C++ |

**Known Issues:**
- Class imbalance (94% safe samples)
- Cleaned version with removed duplicates

### Dataset 3: FormAI

| Property | Value |
|----------|-----|
| Source | [FormAI Dataset](https://github.com/FormAI-Dataset/FormAI-dataset) |
| License | MIT |
| Size | ~500 MB |
| Samples | 246,549 |
| Vulnerable | 201,274 |
| Safe | 45,275 |
| Language | C (AI-generated) |

**Known Issues:**
- Inverse imbalance (82% vulnerable samples)
- AI-generated synthetic code

## Combined Dataset Statistics

| Split | Samples |
|-------|---------|
| Total | 749,794 |
| Training | 420,240 |
| Validation | 89,760 |
| Test | 90,000 |

**Balancing Strategy:**
- Combined datasets to achieve ~3:1 safe:vulnerable ratio
- Stratified sampling to maintain distribution consistency

## Preprocessing Pipeline

1. **Code Cleaning**
   - Remove non-printable characters
   - Normalize whitespace
   - Strip comments (optional)

2. **Tokenization**
   - CodeBERT tokenizer (BPE vocabulary)
   - Sequence truncation to 512 tokens
   - Padding for batch processing

3. **Label Encoding**
   - Binary labels: 0 (Safe), 1 (Vulnerable)
   - Class weights computed for imbalance handling

## Data Splits

```
Dataset Distribution:
├── Training (56%)
│   ├── BigVul: 105,000
│   ├── DiverseVul: 172,000
│   └── FormAI: 143,240
├── Validation (12%)
│   └── Balanced split: 45,000 vulnerable, 44,760 safe
└── Test (12%)
    └── Held-out evaluation: 45,000 vulnerable, 45,000 safe
```

## Feature Engineering

- **Input Features**: Tokenized code sequences
- **Embedding Dimension**: 768 (from CodeBERT)
- **Sequence Length**: 512 tokens (padded/truncated)

## Verification

- BigVul row count: 188,636
- DiverseVul row count: 319,523
- FormAI row count: 246,549
- Combined total: 749,794

## Citation

If you use this dataset combination in your work, please cite:
- BigVul: Li et al., "Automated Software Vulnerability Analysis Using Deep Learning"
- DiverseVul: Tabib et al., "DiverseVul: A Diverse Vulnerability Dataset"
- FormAI: He et al., "FormAI: An LLM-Powered Dataset for Code Vulnerability Detection"