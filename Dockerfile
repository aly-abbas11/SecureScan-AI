FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir gradio

COPY src/ ./src/
COPY app.py .
COPY src/models/baseline_mlp.pt ./models/

EXPOSE 7860

CMD ["python", "app.py"]