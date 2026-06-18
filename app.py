"""
SecureScan AI Demo Application
Run locally: python app.py
Then open http://localhost:7860
"""

import os
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModel
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.securescan_model import SecureScanModel

# Load model and tokenizer
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = SecureScanModel()
model.load_state_dict(torch.load("src/models/baseline_mlp.pt", map_location="cpu"), strict=False)
model.eval()

# Example code snippets
EXAMPLE_CODE = '''char buffer[10];
strcpy(buffer, input);
return 0;'''

def predict_vulnerability(code: str) -> dict:
    """Predict if code contains a vulnerability."""
    if not code or len(code.strip()) == 0:
        return {"error": "Please provide valid code", "vulnerable": None, "confidence": None}
    
    if len(code) > 10000:
        return {"error": "Code too long. Maximum 10,000 characters.", "vulnerable": None, "confidence": None}
    
    try:
        inputs = tokenizer(
            code,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        with torch.no_grad():
            logits = model(inputs["input_ids"], inputs["attention_mask"])
            probs = torch.softmax(logits, dim=-1)
            confidence = probs.max().item()
            prediction = "Vulnerable" if logits.argmax().item() == 1 else "Safe"
        
        return {
            "error": None,
            "vulnerable": prediction,
            "confidence": f"{confidence:.2%}"
        }
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}", "vulnerable": None, "confidence": None}

# Create Gradio interface
with gr.Blocks(title="SecureScan AI") as demo:
    gr.Markdown("""
    # SecureScan AI 🔐
    *AI-powered source code vulnerability detection*
    
    Trained on BigVul, DiverseVul, and FormAI datasets using CodeBERT + BiLSTM + MLP.
    """)
    
    with gr.Row():
        with gr.Column():
            code_input = gr.Textbox(
                label="Source Code",
                placeholder="Paste C/C++ or Python code here...",
                lines=10
            )
            submit_btn = gr.Button("Analyze Vulnerability")
            example_btn = gr.Button("Load Sample")
        
        with gr.Column():
            output = gr.JSON(label="Result")
    
    submit_btn.click(predict_vulnerability, inputs=code_input, outputs=output)
    example_btn.click(lambda x: x, inputs=gr.Number(value=EXAMPLE_CODE, visible=False), outputs=code_input)
    
    gr.Markdown("---")
    gr.Markdown("""
    ## About
    - **Model**: CodeBERT + BiLSTM + MLP (F1: 0.9252)
    - **Limitations**: May miss obfuscated vulnerabilities, best on C/C++ and Python
    - **GitHub**: https://github.com/Salman1122334411/SecureScan-AI
    """)

if __name__ == "__main__":
    demo.launch(server_port=7860)