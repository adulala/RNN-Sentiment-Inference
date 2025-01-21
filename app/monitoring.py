from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram

# Middleware initialization for Prometheus
def init_monitoring(app):
    Instrumentator().instrument(app).expose(app)

# Define custom metrics
prediction_counter = Counter(
    "predictions_total", 
    "Total number of predictions made", 
    ["label"]  # Labels for metrics, e.g., prediction label
)

inference_time = Histogram(
    "inference_latency_seconds", 
    "Time taken for model inference"
)