# Defines the Task class with retry logic and Pydantic parameter validation.

import time
import logging
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, ValidationError

# Define Pydantic models for task parameters
class DataIngestionParameters(BaseModel):
    dataset_size_gb: int = Field(..., description="Size of the dataset in GB")
    source: str = Field(..., description="Source of the data (e.g., internal, external)")

class DataProcessingParameters(BaseModel):
    processing_intensity: str = Field(..., description="Intensity of the processing (e.g., low, medium, high)")

class ModelTrainingParameters(BaseModel):
    model_type: str = Field(..., description="Type of the model (e.g., linear, boosted_tree)")
    epochs: int = Field(..., description="Number of training epochs")
    learning_rate: Optional[float] = Field(0.1, description="Learning rate for training") # Example with optional field and default


class Task:
    def __init__(self, name: str, dependencies: Optional[list] = None, max_retries: int = 3, retry_delay: int = 5, parameters: Optional[Dict[str, Any]] = None):
        self.name = name
        self.dependencies = dependencies if dependencies is not None else []
        self.max_retries = max_retries
        self.retry_delay = retry_delay # seconds
        # Parameters will be validated by subclasses
        self.parameters = parameters if parameters is not None else {}
        self._logger = logging.getLogger(__name__)
        # Store validated parameters (will be set by subclass __init__)
        self.validated_parameters: Optional[BaseModel] = None

    def execute(self):
        """Executes the task with retry logic. Subclasses should override _run()."""
        # Ensure parameters are validated before running
        if self.validated_parameters is None:
             self._logger.error(f"Task {self.name}: Parameters were not validated before execution.")
             raise ValueError("Task parameters not validated.")

        for attempt in range(self.max_retries + 1):
            try:
                self._logger.info(f"Attempt {attempt + 1}/{self.max_retries + 1} for task '{self.name}'")
                # --- Call the actual task execution logic ----
                # Pass validated parameters to _run or access via self.validated_parameters
                result = self._run(self.validated_parameters)
                # -------------------------------------------
                self._logger.info(f"Task '{self.name}' executed successfully on attempt {attempt + 1}.")
                return result # Return the result from _run()
            except ValidationError as e:
                 # Catch Pydantic validation errors during execution if _run re-validates
                 self._logger.error(f"Task '{self.name}' parameter validation failed during execution: {e}")
                 raise # Re-raise validation error
            except Exception as e:
                self._logger.error(f"Task '{self.name}' failed on attempt {attempt + 1}: {e}")
                if attempt < self.max_retries:
                    self._logger.info(f"Retrying task '{self.name}' in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    self._logger.error(f"Task '{self.name}' failed permanently after {self.max_retries + 1} attempts.")
                    raise # Re-raise the last exception if all retries fail

    def _run(self, parameters: BaseModel) -> Any:
        """Abstract method: Implement the core task logic in subclasses.

        This method should contain the actual work of the task.
        It receives the validated parameters as a Pydantic model.
        It should return a result on success and raise an Exception on failure.
        """
        raise NotImplementedError("Subclasses must implement the _run method")

    def __repr__(self):
        # Include validated parameters in the representation if available
        params_repr = self.validated_parameters if self.validated_parameters is not None else self.parameters
        return f"Task(name='{self.name}', max_retries={self.max_retries}, retry_delay={self.retry_delay}, parameters={params_repr})"

# --- Define example Task subclasses (Moved from main.py for better structure) ---

class DataIngestionTask(Task):
    def __init__(self, name: str, dependencies: Optional[list] = None, max_retries: int = 3, retry_delay: int = 5, parameters: Optional[Dict[str, Any]] = None):
        super().__init__(name, dependencies, max_retries, retry_delay, parameters)
        try:
            # Validate parameters using the Pydantic model
            self.validated_parameters = DataIngestionParameters(**self.parameters)
            self._logger.debug(f"Task {self.name}: Parameters validated successfully.")
        except ValidationError as e:
            self._logger.error(f"Task {self.name}: Parameter validation failed: {e}")
            # Store the validation error and mark task for failure or handle appropriately
            self._validation_error = e
            self.validated_parameters = None # Ensure validated_parameters is None on failure
            raise e # Re-raise the error to stop task creation

    def _run(self, parameters: DataIngestionParameters):
        """Simulates data ingestion work."""
        self._logger.info(f"Running data ingestion task: {self.name} with parameters: {parameters}")
        # Simulate work duration, potentially influenced by parameters
        # Access validated parameters using parameters.dataset_size_gb, parameters.source
        # time.sleep(2 * parameters.dataset_size_gb)
        time.sleep(2)
        # Simulate occasional failure for retry testing
        # import random
        # if random.random() < 0.05:
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
    def __init__(self, name: str, dependencies: Optional[list] = None, max_retries: int = 3, retry_delay: int = 5, parameters: Optional[Dict[str, Any]] = None):
         super().__init__(name, dependencies, max_retries, retry_delay, parameters)
         try:
             self.validated_parameters = DataProcessingParameters(**self.parameters)
             self._logger.debug(f"Task {self.name}: Parameters validated successfully.")
         except ValidationError as e:
             self._logger.error(f"Task {self.name}: Parameter validation failed: {e}")
             self._validation_error = e
             self.validated_parameters = None
             raise e

    def _run(self, parameters: DataProcessingParameters):
        """Simulates data processing work."""
        self._logger.info(f"Running data processing task: {self.name} with parameters: {parameters}")
        # Simulate work duration, potentially influenced by parameters
        # time.sleep(3 * (1 if parameters.processing_intensity == 'medium' else 1.5))
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
    def __init__(self, name: str, dependencies: Optional[list] = None, max_retries: int = 3, retry_delay: int = 5, parameters: Optional[Dict[str, Any]] = None):
         super().__init__(name, dependencies, max_retries, retry_delay, parameters)
         try:
             self.validated_parameters = ModelTrainingParameters(**self.parameters)
             self._logger.debug(f"Task {self.name}: Parameters validated successfully.")
         except ValidationError as e:
             self._logger.error(f"Task {self.name}: Parameter validation failed: {e}")
             self._validation_error = e
             self.validated_parameters = None
             raise e

    def _run(self, parameters: ModelTrainingParameters):
        """Simulates model training work."""
        self._logger.info(f"Running model training task: {self.name} with parameters: {parameters}")
        # Simulate work duration, potentially influenced by parameters
        # time.sleep(5 * (parameters.epochs / 50))
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

# -----------------------------------------------------------------------------
