# Defines basic Prometheus metrics.

from prometheus_client import Counter, Histogram, start_http_server
import logging

# Define metrics
TASK_ERRORS = Counter(
    'task_errors_total',
    'Total number of task errors',
    ['task_name'] # Label for task name
)

TASK_DURATION_SECONDS = Histogram(
    'task_duration_seconds',
    'Histogram of task duration in seconds',
    ['task_name'] # Label for task name
)

# TODO: Add more metrics as needed (e.g., queue size, worker count)

def initialize_metrics(port=8000):
    """Initializes and starts the Prometheus metrics server."""
    try:
        start_http_server(port)
        logging.info(f"Prometheus metrics server started on port {port}")
    except OSError as e:
        logging.error(f"Failed to start Prometheus metrics server on port {port}: {e}")
        # TODO: Handle port already in use or other errors appropriately
    # TODO: You might want to return the server object or handle it differently
