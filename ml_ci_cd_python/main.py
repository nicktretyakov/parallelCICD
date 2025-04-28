# Main entry point for the ML CI/CD pipeline

import time
import logging
import concurrent.futures
import argparse
import csv
import os
import json
import yaml # Import PyYAML
import sys # Import sys to exit with status code
import psutil # Import psutil

from utils.logger_config import setup_logging
from metrics.prometheus_metrics import initialize_metrics, TASK_DURATION_SECONDS, TASK_ERRORS
from utils.dependency_manager import DependencyManager
from worker.worker_pool import WorkerPool
from tasks.task import Task
# Import the placeholder ML predictor
from ml_model.predictor import TaskPredictor

# Define the path for the task history data file
TASK_HISTORY_FILE = 'data/task_history.csv'
# Define the path for the saved ML model
ML_MODEL_PATH = 'ml_model/task_predictor_model.joblib'

# --- Define example Task subclasses ---
# These tasks must be importable and have a _run method
# You would likely define these in the tasks/ directory
# For now, keeping them here for simplicity with the config example

class DataIngestionTask(Task):
    def _run(self):\

        """Simulates data ingestion work."""\

        self._logger.info(f"Running data ingestion task: {self.name}")
        # Simulate work duration, potentially influenced by parameters
        # Access parameters using self.parameters
        # dataset_size_gb = self.parameters.get('dataset_size_gb', 1)
        # time.sleep(2 * dataset_size_gb)
        time.sleep(2)
        # Simulate occasional failure for retry testing
        # import random
        # if random.random() < 0.5:
        #     raise Exception("Simulated ingestion failure")
        self._logger.info(f"Data ingestion task {self.name} completed.")

        # --- Basic Resource Usage Collection (Example) ---
        process = psutil.Process(os.getpid()) # Get the current process
        cpu_percent = process.cpu_percent(interval=None) # Get instantaneous CPU usage
        memory_info = process.memory_info() # Get memory usage info
        memory_usage_mb = memory_info.rss / (1024 * 1024) # Resident Set Size in MB
        self._logger.debug(f"Task {self.name} resource usage: CPU={cpu_percent:.2f}%, Memory={memory_usage_mb:.2f}MB")
        # Return resource usage along with task result
        return {'result': f"Data for {self.name} ingested", 'resources': {'cpu_percent': cpu_percent, 'memory_usage_mb': memory_usage_mb}}
        # --------------------------------------------------

class DataProcessingTask(Task):
    def _run(self):\

        """Simulates data processing work."""\

        self._logger.info(f"Running data processing task: {self.name}")
        # Simulate work duration, potentially influenced by parameters
        # processing_intensity = self.parameters.get('processing_intensity', 'medium')
        # duration_factor = 1 if processing_intensity == 'medium' else 1.5
        # time.sleep(3 * duration_factor)
        time.sleep(3)
        self._logger.info(f"Data processing task {self.name} completed.")

        # --- Basic Resource Usage Collection (Example) ---
        process = psutil.Process(os.getpid()) # Get the current process
        cpu_percent = process.cpu_percent(interval=None) # Get instantaneous CPU usage
        memory_info = process.memory_info() # Get memory usage info
        memory_usage_mb = memory_info.rss / (1024 * 1024) # Resident Set Size in MB
        self._logger.debug(f"Task {self.name} resource usage: CPU={cpu_percent:.2f}%, Memory={memory_usage_mb:.2f}MB")
        # Return resource usage along with task result
        return {'result': f"Data for {self.name} processed", 'resources': {'cpu_percent': cpu_percent, 'memory_usage_mb': memory_usage_mb}}
        # --------------------------------------------------

class ModelTrainingTask(Task):
    def _run(self):\

        """Simulates model training work."""\

        self._logger.info(f"Running model training task: {self.name}")
        # Simulate work duration, potentially influenced by parameters
        # epochs = self.parameters.get('epochs', 50)
        # time.sleep(5 * (epochs / 50))
        time.sleep(5)
        self._logger.info(f"Model training task {self.name} completed.")

        # --- Basic Resource Usage Collection (Example) ---
        process = psutil.Process(os.getpid()) # Get the current process
        cpu_percent = process.cpu_percent(interval=None) # Get instantaneous CPU usage
        memory_info = process.memory_info() # Get memory usage info
        memory_usage_mb = memory_info.rss / (1024 * 1024) # Resident Set Size in MB
        self._logger.debug(f"Task {self.name} resource usage: CPU={cpu_percent:.2f}%, Memory={memory_usage_mb:.2f}MB")
        # Return resource usage along with task result
        return {'result': f"Model {self.name} trained", 'resources': {'cpu_percent': cpu_percent, 'memory_usage_mb': memory_usage_mb}}
        # --------------------------------------------------

# --------------------------------------

# Dictionary mapping task type strings from config to actual Task classes
TASK_TYPE_MAPPING = {
    'DataIngestionTask': DataIngestionTask,
    'DataProcessingTask': DataProcessingTask,
    'ModelTrainingTask': ModelTrainingTask,
    # TODO: Add other Task subclasses here when you create them
}

def load_pipeline_config(config_path):
    """Loads pipeline configuration from a YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Pipeline config file not found at {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_tasks_from_config(config):
    """Creates Task objects from the loaded configuration."""
    tasks = {}
    if 'tasks' not in config or not isinstance(config['tasks'], list):
        logging.warning("'tasks' section missing or not a list in config.")
        return tasks

    for task_config in config['tasks']:
        try:
            task_name = task_config['name']
            task_type_str = task_config.get('type', 'Task') # Default to base Task if type is missing
            parameters = task_config.get('parameters', {})
            max_retries = task_config.get('retries', 3) # Simplified retry definition
            retry_delay = task_config.get('retry_delay', 5)

            # Get the actual Task class from the mapping
            task_class = TASK_TYPE_MAPPING.get(task_type_str)
            if task_class is None:
                logging.error(f"Unknown task type '{task_type_str}' for task '{task_name}'. Task will not be created.")
                continue # Skip this task if type is unknown

            # Create the task instance
            task = task_class(
                name=task_name,
                parameters=parameters,
                max_retries=max_retries,
                retry_delay=retry_delay
                # Dependencies are added separately
            )
            tasks[task_name] = task
        except KeyError as e:
            logging.error(f"Missing required key in task configuration: {e}. Task skipped.")
            continue
        except Exception as e:
            logging.error(f"Error creating task from config {task_config}: {e}. Task skipped.")
            continue

    return tasks

def add_dependencies_from_config(dependency_manager, tasks, config):
    """Adds dependencies to the DependencyManager from the loaded configuration."""
    if 'dependencies' not in config or not isinstance(config['dependencies'], list):
        logging.warning("'dependencies' section missing or not a list in config.")
        return

    for dep_config in config['dependencies']:
        try:
            # Dependency format: { 'from': task_name_A, 'to': task_name_B } means B depends on A
            from_task_name = dep_config['from']
            to_task_name = dep_config['to']

            from_task = tasks.get(from_task_name)
            to_task = tasks.get(to_task_name)

            if from_task and to_task:
                # Add edge from dependency (from_task) to dependent task (to_task)
                dependency_manager.add_dependency(to_task, from_task) # 'to' depends on 'from'
            else:
                if not from_task:
                    logging.warning(f"Dependency 'from' task not found: {from_task_name}. Dependency skipped.")
                if not to_task:
                    logging.warning(f"Dependency 'to' task not found: {to_task_name}. Dependency skipped.")
        except KeyError as e:
            logging.error(f"Missing required key in dependency configuration: {e}. Dependency skipped.")
            continue
        except Exception as e:
            logging.error(f"Error adding dependency from config {dep_config}: {e}. Dependency skipped.")
            continue

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting ML CI/CD Pipeline")

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run the ML CI/CD pipeline.")
    parser.add_argument(
        '--config',
        type=str,
        required=True, # Make config file required
        help='Path to the pipeline configuration YAML file'
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=4,
        help='Maximum number of worker threads in the worker pool (default: 4)'
    )
    parser.add_argument(
        '--no-metrics',
        action='store_true',
        help='Disable the Prometheus metrics server'
    )
    parser.add_argument(
        '--train-ml-model',
        action='store_true',
        help='Train the ML model on historical data before running the pipeline'
    )
    # TODO: Add argument for ML model path if not using the default

    args = parser.parse_args()
    logger.info(f"Pipeline started with args: {args}")
    # ------------------------

    # --- Load Pipeline Configuration ---
    pipeline_config = None
    try:
        pipeline_config = load_pipeline_config(args.config)
        logger.info(f"Loaded pipeline configuration from {args.config}")
    except FileNotFoundError as e:
        logger.error(f"Configuration file error: {e}")
        sys.exit(1) # Exit with a non-zero status code
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file {args.config}: {e}")
        sys.exit(1) # Exit on YAML parsing error
    except Exception as e:
         logger.error(f"An unexpected error occurred loading config: {e}")
         sys.exit(1) # Exit on other loading errors
    # -----------------------------------

    # Initialize metrics server based on arguments
    if not args.no_metrics:
        initialize_metrics()

    # --- ML Model Integration ---
    # Initialize the TaskPredictor
    predictor = TaskPredictor()
    if args.train_ml_model:
        logger.info("'--train-ml-model' flag detected. Training ML model...")
        predictor.load_data(TASK_HISTORY_FILE) # Load data from the history file
        if predictor.train_model():
            predictor.save_model(ML_MODEL_PATH)
            logger.info(f"Trained model saved to {ML_MODEL_PATH}")
        else:
            logger.warning("ML model training failed. Running pipeline without predictions.")
            # Optionally exit or run with default behavior if training is critical
    else:
        logger.info("Attempting to load pre-trained ML model...")
        predictor.load_model(ML_MODEL_PATH)
        if not predictor._is_trained:
             logger.warning("No pre-trained model found or loaded. Running pipeline without predictions.")

    # ----------------------------

    # --- Initialize Task History File ---
    # Define CSV header columns
    history_csv_header = ['task_name', 'status', 'duration_seconds', 'timestamp', 'parameters', 'cpu_usage', 'memory_usage']

    if not os.path.exists(TASK_HISTORY_FILE):
        with open(TASK_HISTORY_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(history_csv_header)
        logger.info(f"Created task history file with header: {history_csv_header}")
    else:
         logger.info(f"Task history file already exists: {TASK_HISTORY_FILE}")
         # TODO: Consider adding logic here to check if the header matches, or handle schema evolution
    # ------------------------------------

    dependency_manager = DependencyManager()
    worker_pool = WorkerPool(max_workers=args.max_workers) # Use max_workers from args

    # --- Create Tasks and Dependencies from Config ---
    tasks = {}
    try:
        tasks = create_tasks_from_config(pipeline_config)
        if not tasks:
             logger.error("No tasks defined in the configuration file or failed to create tasks. Exiting.")
             worker_pool.shutdown()
             sys.exit(1) # Exit if no tasks created

        for task in tasks.values():
             dependency_manager.add_task(task)

        add_dependencies_from_config(dependency_manager, tasks, pipeline_config)
        logger.info("Tasks and dependencies loaded from configuration.")

    except Exception as e:
        logger.error(f"Error processing tasks/dependencies from config: {e}")
        worker_pool.shutdown()
        sys.exit(1) # Exit on config processing error
    # -------------------------------------------------

    # Check for cycles after adding all tasks and dependencies
    if dependency_manager.has_cycle():
        logger.error("Detected a cycle in the dependency graph. Exiting.")
        worker_pool.shutdown()
        sys.exit(1) # Exit with a specific error code for cycles

    # --- Main execution loop ---
    submitted_tasks = set() # Keep track of tasks that have been submitted to the worker pool
    future_to_task = {} # Map future objects back to task objects
    # Dictionary to store start times for duration calculation and resource usage
    task_start_times = {}

    logger.info("Starting task execution loop.")

    # --- Dynamic Scaling Monitoring Setup ---
    last_scaling_check_time = time.time()
    scaling_check_interval = 5 # seconds
    # Define simple thresholds based on queue size relative to max workers
    scale_up_threshold_factor = 2 # Suggest scale up if queue size > max_workers * factor
    scale_down_threshold_queue_size = 0 # Suggest scale down if queue is empty
    # Define scaling step size
    scale_step = 2
    # ----------------------------------------

    # Continue as long as there are tasks that are not yet finished (completed or failed)
    while not dependency_manager.all_tasks_finished():
        runnable_tasks = dependency_manager.get_runnable_tasks()

        # --- ML Prediction based Prioritization ---
        if predictor._is_trained and runnable_tasks:
            # Predict duration for runnable tasks and sort them by predicted duration (shortest first)
            runnable_tasks_with_predictions = []
            for task in runnable_tasks:
                predicted_duration = predictor.predict_duration(task, task.parameters)
                runnable_tasks_with_predictions.append((predicted_duration, task))
            runnable_tasks_with_predictions.sort(key=lambda x: x[0])
            runnable_tasks = [task for duration, task in runnable_tasks_with_predictions]
            logger.debug(f"Prioritized runnable tasks based on ML prediction: {[task.name for task in runnable_tasks]}")
        # -----------------------------------------

        for task in runnable_tasks:
            if task.name not in submitted_tasks:
                logger.info(f"Submitting runnable task: {task.name}")

                # --- ML Prediction/Optimization Usage ---
                predicted_duration = None
                resource_recommendations = {}
                if predictor._is_trained:
                    # Pass task parameters to the predictor
                    predicted_duration = predictor.predict_duration(task, task.parameters)
                    resource_recommendations = predictor.optimize_resources(task, task.parameters)
                    logger.info(f"Task {task.name}: Predicted duration = {predicted_duration}s, Resources = {resource_recommendations}. (Actual resource allocation depends on execution environment.)")

                # TODO: Potentially use these predictions/recommendations to influence worker assignment (more advanced)

                start_time = time.time()
                task_start_times[task.name] = start_time
                # Submit the task's execute method (which calls _run internally) to the thread pool
                future = worker_pool.executor.submit(task.execute)
                future_to_task[future] = task
                submitted_tasks.add(task.name)

        # Process completed futures
        # Use a small timeout to allow the loop to check for new runnable tasks periodically
        # Get completed futures with a timeout
        current_futures = list(future_to_task.keys())
        if not current_futures: # Avoid blocking on as_completed if no tasks are currently running
             # Perform scaling check even if no tasks are running/completed
             if time.time() - last_scaling_check_time >= scaling_check_interval:
                 queue_size = worker_pool.task_queue.qsize()
                 # Note: A more advanced scaling check could consider predicted durations of tasks in the queue.
                 logger.info(f"Scaling check: Task queue size is {queue_size}. Current max workers: {worker_pool.max_workers}")

                 # --- Dynamic Scaling Logic ---
                 if queue_size > scale_up_threshold_factor * worker_pool.max_workers:
                     new_workers = worker_pool.max_workers + scale_step
                     # Optionally set an upper limit for max workers
                     # max_allowed_workers = 10
                     # new_workers = min(new_workers, max_allowed_workers)
                     worker_pool.scale_up(new_workers)
                 elif queue_size <= scale_down_threshold_queue_size and worker_pool.max_workers > worker_pool._initial_max_workers:
                      # Simple scale down: return to initial worker count if queue is empty
                      # More complex logic could reduce by 'scale_step'
                     worker_pool.scale_down(worker_pool._initial_max_workers)
                 # -----------------------------

                 last_scaling_check_time = time.time()
             time.sleep(0.1) # Small sleep if no tasks are running to prevent tight loop
             continue # Continue to the next iteration to check for new runnable tasks

        # If there are current_futures, as_completed will handle the waiting/timeout
        for future in concurrent.futures.as_completed(current_futures, timeout=1):
            task = future_to_task.pop(future) # Remove the future once completed
            end_time = time.time()
            duration = end_time - task_start_times.pop(task.name, end_time) # Calculate duration

            # --- Collect Resource Usage from Task Result ---
            # Assuming the task's _run method returns a dictionary including a 'resources' key
            actual_cpu_usage = None
            actual_memory_usage = None
            task_result_data = future.result() # Get the result returned by task.execute() (which calls _run())
            if isinstance(task_result_data, dict) and 'resources' in task_result_data:
                 actual_cpu_usage = task_result_data['resources'].get('cpu_percent')
                 actual_memory_usage = task_result_data['resources'].get('memory_usage_mb')
                 # Log resource usage for debugging/verification
                 logger.debug(f"Collected resources for task {task.name}: CPU={actual_cpu_usage:.2f}, Memory={actual_memory_usage:.2f}MB")
            else:
                 logger.warning(f"Task {task.name} did not return resource usage data in expected format.")
            # --------------------------------------------------------------------------

            try:
                # Original result is now inside task_result_data['result'] if structured
                # If tasks return different types, handle accordingly.
                final_result = task_result_data.get('result', task_result_data) if isinstance(task_result_data, dict) else task_result_data
                logger.info(f"Task {task.name} finished successfully.")
                # Mark task as complete in DependencyManager based on successful execution
                dependency_manager.mark_task_complete(task)
                # Record task duration metric on success
                TASK_DURATION_SECONDS.labels(task_name=task.name).observe(duration)

                # --- Record Task Data (Success) ---
                try:
                    with open(TASK_HISTORY_FILE, 'a', newline='') as f:
                        writer = csv.writer(f)
                        # Serialize parameters to JSON string for storage
                        parameters_json = json.dumps(task.parameters)
                        # Write task data, including parameters and collected resource usage
                        writer.writerow([task.name, 'success', duration, time.time(), parameters_json, actual_cpu_usage, actual_memory_usage])
                    logger.debug(f"Recorded success data for task {task.name} to history file.")
                except Exception as data_e:
                    logger.error(f"Error recording task success data for {task.name}: {data_e}")
                # ----------------------------------

            except Exception as e:
                # This exception is caught if future.result() failed, meaning the task permanently failed after retries.
                logger.error(f"Task {task.name} permanently failed after retries: {e}")
                # Mark task as permanently failed in DependencyManager
                dependency_manager.mark_task_failed(task)
                # Record task error metric on permanent failure
                TASK_ERRORS.labels(task_name=task.name).inc()

                # --- Record Task Data (Failure) ---
                try:
                    with open(TASK_HISTORY_FILE, 'a', newline='') as f:
                        writer = csv.writer(f)
                        # Serialize parameters to JSON string for storage
                        parameters_json = json.dumps(task.parameters)
                        # Write task data, including parameters and collected resource usage (might be None)
                        writer.writerow([task.name, 'failure', duration, time.time(), parameters_json, actual_cpu_usage, actual_memory_usage]) # Record duration up to failure
                    logger.debug(f"Recorded failure data for task {task.name} to history file.")
                except Exception as data_e:
                     logger.error(f"Error recording task failure data for {task.name}: {data_e}")
                # ----------------------------------

                # Note: Downstream tasks depending on this one will not become runnable

        # Perform scaling check after processing completed tasks
        if time.time() - last_scaling_check_time >= scaling_check_interval:
            queue_size = worker_pool.task_queue.qsize()
            # Note: A more advanced scaling check could consider predicted durations of tasks in the queue.
            logger.info(f"Scaling check: Task queue size is {queue_size}. Current max workers: {worker_pool.max_workers}")

            # --- Dynamic Scaling Logic ---
            if queue_size > scale_up_threshold_factor * worker_pool.max_workers:
                new_workers = worker_pool.max_workers + scale_step
                # Optionally set an upper limit for max workers
                # max_allowed_workers = 10
                # new_workers = min(new_workers, max_allowed_workers)
                worker_pool.scale_up(new_workers)
            elif queue_size <= scale_down_threshold_queue_size and worker_pool.max_workers > worker_pool._initial_max_workers:
                 # Simple scale down: return to initial worker count if queue is empty
                 # More complex logic could reduce by 'scale_step'
                worker_pool.scale_down(worker_pool._initial_max_workers)
            # -----------------------------

            last_scaling_check_time = time.time()

        # Add a small sleep *only if* no futures were processed in this iteration of as_completed
        # to avoid excessive checks when tasks are running. This sleep is now handled by the timeout in as_completed
        # and the check at the beginning of the as_completed loop.
        pass # Removed redundant sleep here

    logger.info("Task execution loop finished.")
    # ---------------------------------

    # Check pipeline status after the loop
    if dependency_manager.has_failed_tasks():
        logger.error("Pipeline finished with errors. Some tasks failed.")
        sys.exit(1) # Exit with a non-zero status code indicating failure
    else:
        logger.info("Pipeline finished successfully. All tasks completed.")
        sys.exit(0) # Exit with a zero status code indicating success

    # Shutdown the worker pool (will not be reached if exiting with sys.exit)
    # worker_pool.shutdown()
    # logger.info("Worker pool shut down complete.")
    # logger.info("Pipeline process finished.")

if __name__ == "__main__":
    main()
