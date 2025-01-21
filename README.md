# RNN-Sentiment-Inference


A sentiment analysis service using a **Recurrent Neural Network (RNN)**, deployed as a RESTful API with **FastAPI**, and featuring monitoring via **Prometheus**.

---

## **Features**

- **Model**: Pre-trained RNN for binary sentiment classification (positive/negative).
- **API**: FastAPI-based RESTful interface for predictions.
- **Monitoring**: Prometheus integration to monitor metrics like prediction count and latency.
- **Docker Support**: Easily deployable with Docker.

---

## **Directory Structure**

```plaintext
RNN-Sentiment-Inference/
│
├── model/                  # Directory for storing model artifacts
│   ├── rnn_sentiment_model.pth  # Trained model weights
│   ├── vocab.pth               # Vocabulary file
│
├── app/                    # Application code
│   ├── main.py              # FastAPI application entry point
│   ├── utils.py             # Utility functions (tokenization, model loading, etc.)
│   ├── monitoring.py        # Prometheus monitoring setup
│
├── requirements.txt         # Python dependencies
├── Dockerfile               # Docker configuration file
├── README.md                # Project documentation

```

## **Getting Started**

### **Prerequisites**

- Python 3.8+
- CUDA-enabled GPU (optional but recommended)
- Prometheus (for monitoring metrics)

### **Installation**

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/RNN-Sentiment-Inference.git
   cd RNN-Sentiment-Inference


### **Running the Application
Start the FastAPI server:

bash
Copy
Edit
uvicorn app.main:app --reload

### **Deployment
Docker
Build the Docker image:

bash
Copy
Edit
docker build -t rnn-sentiment-inference .
Run the container:

bash
Copy
Edit
docker run -d -p 8000:8000 rnn-sentiment-inference
Prometheus Setup
Add the following job to your Prometheus configuration file:

yaml
Copy
Edit
scrape_configs:
  - job_name: "fastapi"
    static_configs:
      - targets: ["localhost:8000"]
Start Prometheus with the updated configuration.

### **Project Notes
Model Training: The model was trained on a labeled sentiment dataset using PyTorch.
Threshold: Predictions above 0.5 are classified as positive; otherwise, negative.

