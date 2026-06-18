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
import torch
from src.models.securescan_model import SecureScanModel
from transformers import AutoTokenizer

# Load model
model = SecureScanModel()
model.load_state_dict(torch.load('src/models/baseline_mlp.pt'))
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

### Command Line Interface

```bash
# Evaluate on test set
python -c "from src.training.train import final_evaluation; ..."
```

## Testing

```bash
# Run unit tests
pytest tests/ -v

# Run specific test
python tests/test_pipeline.py
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