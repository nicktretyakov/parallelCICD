# Basic Flask API for the ML CI/CD pipeline with authentication and authorization.

import logging
from flask import Flask, jsonify, request
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash
import yaml
import os
import csv
import json
import subprocess # To run main.py as a separate process

# Setup basic logging for the API
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
auth = HTTPBasicAuth()

# --- User Authentication Configuration ---
# TODO: Replace this with a secure way to store and retrieve user credentials (e.g., database, environment variables)
# This is for demonstration purposes ONLY.
users = {
    "admin": generate_password_hash("supersecretpassword"), # TODO: Use strong, unique passwords
    "user": generate_password_hash("secretpassword")
}

# TODO: Implement roles and permissions based on your needs
roles = {
    "admin": ['read', 'write', 'trigger', 'manage_workers'],
    "user": ['read']
}

@auth.verify_password
def verify_password(username, password):
    """Verifies the provided username and password.

    TODO: Integrate with a proper user management system.
    """
    logger.info(f"Attempting to authenticate user: {username}")
    if username in users and check_password_hash(users.get(username), password):
        logger.info(f"User {username} authenticated successfully.")
        return username # Return the username on success
    logger.warning(f"Authentication failed for user: {username}")
    return None # Return None on failure

@auth.get_user_roles
def get_user_roles(username):
    """Returns the roles for the authenticated user.

    TODO: Retrieve roles from your user management system.
    """
    logger.debug(f"Getting roles for user: {username}")
    return roles.get(username, []) # Return roles or empty list if user/roles not found
# ----------------------------------------

# --- Authorization Helper (Example) ---
def requires_role(role):
    """Decorator to require a specific role for an endpoint."
    def decorator(f):
        from functools import wraps
        @wraps(f)
        @auth.login_required # Ensure user is authenticated
        def decorated_function(*args, **kwargs):
            if role not in get_user_roles(auth.current_user()):
                logger.warning(f"User {auth.current_user()} does not have required role: {role}")
                return jsonify({'message': 'Unauthorized'}), 403 # Forbidden
            return f(*args, **kwargs)
        return decorated_function
    return decorator
# ---------------------------------------

# --- Helper function to read pipeline config ---
def read_pipeline_config(config_path="ml_ci_cd_python/pipeline_config.yaml"):
    """Reads the pipeline configuration file."
    if not os.path.exists(config_path):
        logger.error(f"Pipeline config file not found at {config_path}")
        return None
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except yaml.YAMLError as e:
        logger.error(f"Error parsing pipeline config file {config_path}: {e}")
        return None

# --- Helper function to read task history ---
def read_task_history(history_file="data/task_history.csv"):
    """Reads the task history CSV file."
    if not os.path.exists(history_file):
        logger.warning(f"Task history file not found at {history_file}")
        return []
    try:
        history_data = []
        with open(history_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Attempt to parse parameters back to a dictionary if stored as JSON string
                try:
                    if 'parameters' in row and row['parameters']:
                        row['parameters'] = json.loads(row['parameters'])
                    else:
                         row['parameters'] = {}
                except json.JSONDecodeError:
                     logger.warning(f"Could not decode parameters JSON for task {row.get('task_name','unknown')}: {row.get('parameters','')}")
                     row['parameters'] = {} # Set to empty dict on decode error
                history_data.append(row)
        return history_data
    except Exception as e:
        logger.error(f"Error reading task history file {history_file}: {e}")
        return []


# --- API Endpoints ---

@app.route('/pipeline/structure', methods=['GET'])
@requires_role('read')
def get_pipeline_structure():
    """Returns the pipeline structure based on the configuration file."
    logger.info(f"User {auth.current_user()} accessing pipeline structure endpoint.")
    config = read_pipeline_config()
    if config:
        # Return just the tasks and dependencies section of the config
        return jsonify({
            'pipeline': config.get('pipeline', {}),
            'tasks': config.get('tasks', []),
            'dependencies': config.get('dependencies', [])
        })
    else:
        return jsonify({'message': 'Could not load pipeline configuration.'}), 500


@app.route('/status', methods=['GET'])
@requires_role('read') # Requires basic authentication with 'read' role
def get_pipeline_status():
    """Returns a simplified view of the pipeline status based on recent history."
    logger.info(f"User {auth.current_user()} accessing status endpoint.")

    # NOTE: This provides STATUS based on the LAST recorded entry in history.csv.
    # For true real-time status, the API would need to interact with the actively
    # running pipeline process and its DependencyManager/Task objects.

    history_data = read_task_history()
    # Create a dictionary to hold the last status for each task
    task_last_status = {}
    for entry in history_data:
        if 'task_name' in entry and 'status' in entry:
            task_last_status[entry['task_name']] = entry['status']

    # Get expected tasks from config to show tasks that might not have history yet
    config = read_pipeline_config()
    expected_tasks = set()
    if config and 'tasks' in config:
        expected_tasks = {task['name'] for task in config['tasks'] if 'name' in task}

    # Combine known tasks (from history and config) and their last known status
    all_task_names = list(set(list(task_last_status.keys()) + list(expected_tasks)))
    all_task_names.sort() # Sort for consistent output

    status_summary = []
    for task_name in all_task_names:
         status = task_last_status.get(task_name, 'not_run_yet') # Default status
         status_summary.append({'task_name': task_name, 'status': status})


    return jsonify({'status_summary': status_summary})


@app.route('/tasks/<task_name>/logs', methods=['GET'])
@requires_role('read') # Requires basic authentication with 'read' role
def get_task_logs(task_name):
    """Returns logs for a specific task.

    TODO: Implement a proper logging solution where logs are stored per task
    in a way accessible by the API (e.g., separate log files per task or a database).

    For now, this is a placeholder.
    """
    logger.info(f"User {auth.current_user()} accessing logs for task: {task_name}")
    # Placeholder implementation: In a real scenario, read logs from a file or database
    dummy_logs = f"[INFO] This is a dummy log for task {task_name}.
[INFO] Task started...
[INFO] Task completed successfully."\
        f"
TODO: Implement actual log retrieval for {task_name}."
    return jsonify({'task_name': task_name, 'logs': dummy_logs})


@app.route('/history', methods=['GET'])
@requires_role('read') # Requires basic authentication with 'read' role
def get_historical_data():
    """Returns historical task execution data."
    logger.info(f"User {auth.current_user()} accessing historical data endpoint.")
    history_data = read_task_history()
    # TODO: Implement filtering, sorting, and pagination for large datasets
    return jsonify({'history': history_data})


@app.route('/trigger', methods=['POST'])
@requires_role('trigger') # Requires 'trigger' role
def trigger_pipeline_run():
    """Triggers a new pipeline run by running main.py as a subprocess."
    logger.info(f"User {auth.current_user()} attempting to trigger pipeline run.")

    # Get optional config path from request body
    config_path = "ml_ci_cd_python/pipeline_config.yaml" # Default config path
    request_data = request.json
    if request_data and 'config_path' in request_data:
        config_path = request_data['config_path']
        logger.info(f"Using config path from request body: {config_path}")

    # Check if the config file exists before attempting to run
    if not os.path.exists(config_path):
        logger.error(f"Trigger failed: Config file not found at {config_path}")
        return jsonify({'message': f'Error: Config file not found at {config_path}'}), 400

    try:
        # Run main.py as a separate process
        # This assumes main.py is executable and accepts --config argument
        # In a real application, you might pass more arguments (e.g., specific parameters for the run)
        # Consider using a task queue (e.g., Celery, RQ) for more robust background job management
        command = [sys.executable, "ml_ci_cd_python/main.py", "--config", config_path]
        logger.info(f"Running pipeline as subprocess: {command}")
        # Use subprocess.Popen to run in the background without waiting
        process = subprocess.Popen(command,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)

        # NOTE: The API process will NOT directly track the subprocess's detailed progress.
        # Status updates would need to be written by the main.py process to a shared state
        # (e.g., a database, status file) that the API can read.

        return jsonify({'message': 'Pipeline run triggered successfully.', 'config_path': config_path}), 202 # Accepted

    except Exception as e:
        logger.error(f"Error triggering pipeline run: {e}")
        return jsonify({'message': f'Error triggering pipeline run: {e}'}), 500

# TODO: Add an endpoint to stop a running pipeline run (requires tracking process IDs or using a job queue)

# ---------------------

if __name__ == '__main__':
    # In production, use a production-ready WSGI server (e.g., Gunicorn, uWSGI)
    # app.run(debug=True) # debug=True is for development only

    # To run this API:
    # 1. Ensure dependencies are installed: pip install Flask Flask-HTTPAuth PyYAML pandas
    # 2. Set the FLASK_APP environment variable: export FLASK_APP=ml_ci_cd_python/api/app.py
    # 3. Run flask: flask run
    #    By default, this runs on http://127.0.0.1:5000/
    # 4. Access endpoints with basic authentication (e.g., curl -u admin:supersecretpassword http://127.0.0.1:5000/status)

    # Example of running with Gunicorn (more suitable for production):
    # gunicorn -w 4 'ml_ci_cd_python.api.app:app' -b 0.0.0.0:5000

    # Note: Running the Flask app directly with app.run() is not recommended for production.
    pass
