# Deployment

## Local Installation

### Prerequisites

- Python 3.9+
- pip package manager
- CUDA-capable GPU (optional, CPU supported)
- Git

### Setup

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

### Quick Start

```bash
# Run the main notebook (includes preprocessing, training, evaluation)
jupyter notebook notebooks/Phase4_SecureScan_AI_Final.ipynb

# Or run training script directly
python src/training/train.py
```

## Inference

### Using Pre-trained Model

```python
from src.models.loader import DEFAULT_CHECKPOINT_DIR, build_model, load_checkpoint
from src.models.securescan_model import SecureScanModel
from transformers import AutoTokenizer

# Built from the encoder config here; every parameter is then taken from the
# checkpoint, which is validated before anything is applied.
model = build_model(SecureScanModel, from_scratch=True)
load_checkpoint(model, DEFAULT_CHECKPOINT_DIR / 'best_model.pt')
model.eval()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')

# Predict vulnerability
code = """
int vulnerable_function(char *input) {
    char buffer[10];
    strcpy(buffer, input);  // Buffer overflow!
    return 0;
}
"""

inputs = tokenizer(code, return_tensors='pt', truncation=True, max_length=512)
with torch.no_grad():
    logits = model(inputs['input_ids'], inputs['attention_mask'])
    prediction = 'Vulnerable' if logits.argmax().item() == 1 else 'Safe'
print(f"Prediction: {prediction}")
```

Checkpoints are not committed (`*.pt` is gitignored): train one, or drop a
released checkpoint into `checkpoints/`. A checkpoint that does not fit the
architecture raises instead of leaving layers randomly initialised — the snippet
that used to be here loaded an MLP checkpoint into this model, which silently
discarded every weight.

### Command Line Interface

```bash
# Evaluate on test set
python -c "from src.training.train import final_evaluation; ..."
```

## Testing

```bash
# Run unit tests (offline: a tiny local encoder stands in for CodeBERT)
pytest tests/ -v

# Run a single module
pytest tests/test_loader.py -v
```

The demo app can be exercised without a checkpoint:

```bash
python app.py --smoke-test   # runs the UI with untrained weights
```

## Project Structure for Deployment

```
SecureScan-AI/
├── src/
│   ├── models/           # Model definitions (importable)
│   ├── training/         # Training scripts
│   ├── preprocessing/    # Data processing
│   ├── data/             # Dataset utilities
│   └── utils/            # Helper functions
├── notebooks/            # Reproducible notebooks with outputs
└── requirements.txt      # Python dependencies
```

## Hosting

The model was developed locally and is intended for:
- Local deployment via Python API
- Integration into CI/CD pipelines
- Jupyter notebook exploration

For production deployment, consider:
- Model quantization for faster inference
- ONNX export for cross-platform compatibility
- REST API wrapper with FastAPI/Flask

## Docker Deployment (Optional)

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY models/ ./models/

CMD ["python", "src/training/train.py"]
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce batch size in config.yaml |
| Tokenizer not found | Run `pip install transformers` |
| Model download slow | Check internet connection, model ~500MB |