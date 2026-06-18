# SecureScan AI 🔐

> AI-powered source code vulnerability detection using CodeBERT + BiLSTM + MLP

## Overview

SecureScan AI is a deep learning system for detecting security vulnerabilities in source code. It uses a CodeBERT + BiLSTM + MLP architecture trained on 600,000+ samples from BigVul, DiverseVul, and FormAI datasets.

**Course:** AI335L Deep Learning Lab — Spring 2026

## Demo

[Live Demo](https://frontend-mu-orpin-87.vercel.app/) - Try the deployed application

![Demo Screenshot](https://raw.githubusercontent.com/Salman1122334411/SecureScan-AI/main/reports/Week3/confusion_matrix.png)

## Installation

```bash
# Clone the repository
git clone https://github.com/Salman1122334411/SecureScan-AI
cd SecureScan-AI

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.9+
- PyTorch 2.0.0
- Transformers 4.35.0
- scikit-learn 1.3.0
- CUDA (optional for GPU acceleration)

## Usage

### Quick Start

```bash
# Run the main notebook with all outputs
jupyter notebook notebooks/Phase4_SecureScan_AI_Final.ipynb

# Or run training directly
python src/training/train.py
```

### Inference Example

```python
from src.models.securescan_model import SecureScanModel
from transformers import AutoTokenizer
import torch

# Load model and tokenizer
model = SecureScanModel()
tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')

# Analyze code
code = "char buf[10]; strcpy(buf, input);"
inputs = tokenizer(code, return_tensors='pt', truncation=True, max_length=512)

with torch.no_grad():
    logits = model(inputs['input_ids'], inputs['attention_mask'])
    prediction = 'Vulnerable' if logits.argmax().item() == 1 else 'Safe'

print(f"Result: {prediction}")
```

### Project Structure

```
SecureScan-AI/
├── data/                    # Dataset files (gitignored)
├── docs/
│   ├── ARCHITECTURE.md       # Technical architecture
│   ├── DATA.md               # Dataset documentation
│   ├── DEPLOYMENT.md         # Deployment guide
│   └── RESULTS.md            # Evaluation results
├── experiments/             # Training configs and logs
├── notebooks/               # Jupyter notebooks with outputs
├── reports/                 # Analysis reports and visualizations
├── phases/
│   ├── phase1-roadmap/       # Project planning and roadmap
│   ├── phase2-baseline/      # Baseline MLP implementation
│   ├── phase3-architecture/  # Core deep architecture
│   ├── phase4-refinement/    # Transfer learning & HPO
│   ├── phase5-evaluation/    # Evaluation & ablations
│   └── phase6-deployment/    # Final deployment deliverables
├── src/
│   ├── data/loader.py       # Dataset utilities
│   ├── models/              # Model definitions
│   ├── preprocessing/       # Data preprocessing
│   ├── training/train.py    # Training script
│   └── utils/helpers.py     # Helper functions
└── tests/                   # Unit tests
```

## Results

| Model | F1 Score |
|-------|----------|
| Baseline MLP | 0.7346 |
| CodeBERT + BiLSTM + MLP | **0.9252** |

Full results available in [docs/RESULTS.md](docs/RESULTS.md).

## Model Architecture

```
Input → CodeBERT (frozen layers) → BiLSTM → MLP → [Safe, Vulnerable]
```

- **CodeBERT**: microsoft/codebert-base (first 6 layers frozen)
- **BiLSTM**: 256 hidden units, 2 layers
- **MLP**: 512→256→128→2 with ReLU, BatchNorm, Dropout

## Datasets

- **BigVul**: 188K real CVE samples (C/C++)
- **DiverseVul**: 319K C/C++ samples
- **FormAI**: 246K AI-generated code samples

Details in [docs/DATA.md](docs/DATA.md).

## Citation

```bibtex
@misc{securescan-ai,
  title={SecureScan AI -- Source Code Vulnerability Detection},
  author={Shah, Ali Abbas and Tanveer, Salman and Ali, Hammad},
  year={2026},
  howpublished={\url{https://github.com/Salman1122334411/SecureScan-AI}},
}
```

Or use [CITATION.cff](CITATION.cff) for citation tools.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- CodeBERT model: Microsoft Research
- BigVul dataset: MSR
- DiverseVul dataset: Ahmed Tabib
- Course instructor: Dr. [Name] (AI335L Deep Learning Lab)