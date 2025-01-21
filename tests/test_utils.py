import pytest
import torch
import json

from main import app  # This is your FastAPI app
from fastapi.testclient import TestClient

client = TestClient(app)

# Replace this with the path to your trained model
MODEL_PATH = "rnn_sentiment_model.pth"

# Test data (you can change this to any sentence you'd like to test)
TEST_TEXT = "I loved the movie! It was fantastic and exciting."

def load_model(model_path):
    # Load the trained model from the saved file
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = 69023  # Ensure this matches the vocab size used during training
    embed_dim = 20
    rnn_hidden_size = 64
    fc_hidden_size = 64

    # Define the model architecture (same as during training)
    model = RNN(vocab_size, embed_dim, rnn_hidden_size, fc_hidden_size)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def test_load_model():
    model = load_model(MODEL_PATH)
    assert model is not None
    assert isinstance(model, torch.nn.Module)  # Check that the loaded model is a PyTorch module

def test_predict():
    model = load_model(MODEL_PATH)
    # Send test text as JSON in a POST request
    response = client.post("/predict/", json={"text": TEST_TEXT})
    assert response.status_code == 200
    response_json = response.json()

    # Check if the response contains the 'sentiment' and 'score'
    assert "sentiment" in response_json
    assert "score" in response_json

    # Check if 'score' is a float and 'sentiment' is a string (either 'positive' or 'negative')
    assert isinstance(response_json["score"], float)
    assert response_json["sentiment"] in ["positive", "negative"]

def test_infer():
    # Testing the inference function with the test text
    model = load_model(MODEL_PATH)
    response = client.post("/predict/", json={"text": TEST_TEXT})

    assert response.status_code == 200
    response_json = response.json()

    # Extract sentiment and score from the JSON response
    sentiment = response_json["sentiment"]
    score = response_json["score"]

    # Log and assert the predictions
    print(f"Predicted Sentiment: {sentiment}, Score: {score:.4f}")

    # Check if sentiment matches the expected result (you can adjust this based on input)
    expected_sentiment = "positive"  # Replace with your expectation based on the input sentence
    assert sentiment == expected_sentiment, f"Expected {expected_sentiment}, but got {sentiment}"

