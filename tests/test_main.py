import logging
from fastapi.testclient import TestClient
from main import app  # This is your FastAPI app instance
import json

client = TestClient(app)

# Set up logging configuration
logging.basicConfig(level=logging.DEBUG)

def test_health():
    logging.info("Testing /health endpoint")
    response = client.get("/health")
    logging.debug(f"Response status: {response.status_code}")
    logging.debug(f"Response JSON: {response.json()}")
    assert response.status_code == 200
    assert response.json() == {"message": "ok"}

def test_infer_positive_sentiment():
    logging.info("Testing /predict endpoint for positive sentiment")
    
    # Sample test sentence to send
    test_text = "I loved the movie! It was fantastic and exciting."
    
    response = client.post(
        "/predict/",
        json={"text": test_text}  # sending text as JSON
    )
    
    logging.debug(f"Response status: {response.status_code}")
    logging.debug(f"Response JSON: {response.json()}")

    assert response.status_code == 200
    assert isinstance(response.json(), dict)

    # Extract sentiment and score from the JSON response
    sentiment = response.json()["sentiment"]
    score = response.json()["score"]

    logging.info(f"Predicted sentiment: {sentiment}, Score: {score:.4f}")

    # Assert positive sentiment for the given test text
    expected_sentiment = "positive"
    assert sentiment == expected_sentiment, f"Expected {expected_sentiment}, but got {sentiment}"
    assert 0 <= score <= 1, f"Score should be between 0 and 1, but got {score}"

def test_infer_negative_sentiment():
    logging.info("Testing /predict endpoint for negative sentiment")
    
    # Sample test sentence to send
    test_text = "I hated the movie. It was boring and too long."
    
    response = client.post(
        "/predict/",
        json={"text": test_text}  # sending text as JSON
    )
    
    logging.debug(f"Response status: {response.status_code}")
    logging.debug(f"Response JSON: {response.json()}")

    assert response.status_code == 200
    assert isinstance(response.json(), dict)

    # Extract sentiment and score from the JSON response
    sentiment = response.json()["sentiment"]
    score = response.json()["score"]

    logging.info(f"Predicted sentiment: {sentiment}, Score: {score:.4f}")

    # Assert negative sentiment for the given test text
    expected_sentiment = "negative"
    assert sentiment == expected_sentiment, f"Expected {expected_sentiment}, but got {sentiment}"
    assert 0 <= score <= 1, f"Score should be between 0 and 1, but got {score}"

def test_empty_text():
    logging.info("Testing /predict endpoint with empty input text")
    
    test_text = ""  # Empty string
    
    response = client.post(
        "/predict/",
        json={"text": test_text}  # sending empty text as JSON
    )
    
    logging.debug(f"Response status: {response.status_code}")
    logging.debug(f"Response JSON: {response.json()}")

    assert response.status_code == 400  # You might want to return a 400 for invalid input
    assert "detail" in response.json()  # FastAPI automatically includes an error message for invalid input

def test_missing_text_field():
    logging.info("Testing /predict endpoint with missing text field")
    
    # Missing "text" field in the request
    response = client.post(
        "/predict/",
        json={}  # Missing "text"
    )
    
    logging.debug(f"Response status: {response.status_code}")
    logging.debug(f"Response JSON: {response.json()}")

    assert response.status_code == 422  # Unprocessable entity error due to missing field
    assert "detail" in response.json()


