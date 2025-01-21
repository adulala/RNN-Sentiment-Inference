import uvicorn

from pydantic import BaseModel
import torch
from pyimagesearch.utils import load_model, load_vocab, predict
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
from monitoring import init_monitoring, prediction_counter, inference_time  # Import monitoring
from fastapi import FastAPI, HTTPException

# Configuration
MODEL_PATH = "rnn_sentiment_model.pth"
VOCAB_PATH = "vocab.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize FastAPI app
app = FastAPI()

# Initialize Prometheus monitoring
init_monitoring(app)

# Load the model and vocabulary
model = load_model(
    MODEL_PATH, DEVICE, vocab_size=69025, embed_dim=20, rnn_hidden_size=64, fc_hidden_size=64
)
vocab = load_vocab(VOCAB_PATH)

# Define the Pydantic model for API input
class TextInput(BaseModel):
    text: str

# Define prediction endpoint
@app.post("/predict")
async def predict_sentiment(input: TextInput):
    from time import time

    start_time = time()
    
    try:
        # Perform prediction
        result = predict(model, vocab, input.text, DEVICE)

        # Update monitoring metrics
        prediction_counter.labels(label=result["prediction"]).inc()
        inference_time.observe(time() - start_time)

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Define a health check route
@app.get("/health")
async def health():
    return {"message": "ok"}

# Define metrics endpoint
@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/test_exception")
async def test_exception():
    raise HTTPException(status_code=400, detail="Test Exception")
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


