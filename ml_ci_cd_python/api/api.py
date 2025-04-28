# Basic Flask API for the ML CI/CD Pipeline with API Key Authentication.

from flask import Flask, request, jsonify, _request_ctx_stack
from functools import wraps
import os
import logging

# TODO: Import necessary components from your pipeline (e.g., DependencyManager, WorkerPool)
# from ..utils.dependency_manager import DependencyManager
# from ..worker.worker_pool import WorkerPool

# Set up basic logging for the API
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# --- Authentication Configuration ---
# In a real application, store this securely (e.g., environment variables, secrets management)
API_KEYS = {
    'your_secret_api_key': 'pipeline_admin', # Example: API Key and associated user/role
    'another_key_123': 'pipeline_user'
}

# --- Authentication Decorator ---
def require_api_key(view_function):
    @wraps(view_function)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key') # Get API key from header

        if not api_key:
            logger.warning("API access denied: No API key provided.")
            return jsonify({'message': 'API key is required'}), 401 # Unauthorized

        if api_key not in API_KEYS:
            logger.warning(f"API access denied: Invalid API key provided: {api_key[:5]}...") # Log a prefix only
            return jsonify({'message': 'Invalid API key'}), 401 # Unauthorized

        # Optional: Store the user/role associated with the API key in the request context
        _request_ctx_stack.top.current_user = API_KEYS[api_key]
        logger.info(f"API access granted for user: {_request_ctx_stack.top.current_user}")

        return view_function(*args, **kwargs)
    return decorated_function

# --- Authorization Helper (Example) ---
# You would use this within your endpoint logic
def current_user_has_role(role):
    user_role = getattr(_request_ctx_stack.top, 'current_user', None)
    # TODO: Implement role checking logic. Simple example assumes API key directly maps to role.
    # In a real system, you might look up roles based on a user ID.
    return user_role == role

# --- API Endpoints ---

@app.route('/status', methods=['GET'])
@require_api_key
def get_status():
    # This endpoint requires a valid API key
    user = getattr(_request_ctx_ctx_stack.top, 'current_user', 'unknown')
    logger.info(f"Status endpoint accessed by user: {user}")

    # TODO: Implement logic to get the current pipeline status (e.g., from DependencyManager)
    pipeline_status = {
        'overall_status': 'running', # Example status
        'completed_tasks': [],
        'failed_tasks': [],
        'runnable_tasks': []
    }

    return jsonify(pipeline_status), 200

@app.route('/run', methods=['POST'])
@require_api_key
# TODO: Add authorization check if only certain roles can trigger runs
# @require_role('pipeline_admin')
def run_pipeline():
    user = getattr(_request_ctx_stack.top, 'current_user', 'unknown')
    logger.info(f"Run pipeline endpoint accessed by user: {user}")

    # TODO: Get pipeline configuration or parameters from the request body
    # request_data = request.get_json()
    # config_name = request_data.get('config', 'default_config.yaml')

    # TODO: Implement logic to trigger a pipeline run
    # This would involve loading the config, creating tasks/dependencies, and starting the worker pool.
    # You might need to refactor main.py to make its core logic callable.

    return jsonify({'message': 'Pipeline trigger received (not fully implemented)'}), 202 # Accepted

# --- Main execution for the API (for development/testing) ---
if __name__ == '__main__':
    # In a production environment, use a production-ready WSGI server like Gunicorn or uWSGI
    logger.info("Starting Flask API server...")
    app.run(debug=True, port=5000) # debug=True should be False in production
