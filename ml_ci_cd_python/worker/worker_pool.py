# Implements a worker pool for executing tasks.

import concurrent.futures
import queue
import logging
import threading

class WorkerPool:
    def __init__(self, max_workers=5):
        self._initial_max_workers = max_workers
        self.max_workers = max_workers
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.task_queue = queue.Queue()
        self._logger = logging.getLogger(__name__)
        self._running = True
        self._lock = threading.Lock() # Lock for scaling operations
        # Start a thread to continuously pull tasks from the queue
        self._submitter_future = self.executor.submit(self._run_tasks)

    def _run_tasks(self):
        """Continuously pulls tasks from the queue and submits them to the executor."""
        while self._running or not self.task_queue.empty():
            try:
                # Get a task with a timeout so we can check self._running periodically
                task = self.task_queue.get(timeout=1)
                self._logger.info(f"Worker pool submitting task: {task.name}")
                # Submit the task's execute method to the thread pool
                future = self.executor.submit(task.execute)
                # Note: Task completion handling is done in main loop via as_completed

            except queue.Empty:
                # No tasks in queue, continue loop to check self._running
                continue
            except Exception as e:
                self._logger.error(f"Error submitting task from queue: {e}")
                # TODO: Decide how to handle errors during submission itself

    def submit_task(self, task):
        """Adds a task to the internal queue."""
        self._logger.info(f"Adding task to worker pool queue: {task.name}")
        self.task_queue.put(task)

    def scale_up(self, new_max_workers):
        """Increases the maximum number of workers in the pool."""
        with self._lock:
            if new_max_workers > self.max_workers:
                self._logger.info(f"Scaling up worker pool from {self.max_workers} to {new_max_workers} workers.")
                self.max_workers = new_max_workers
                # Directly accessing and updating the protected _max_workers attribute
                # This is a simplification; proper dynamic scaling might require recreating the executor
                try:
                    self.executor._max_workers = new_max_workers
                except AttributeError:
                    self._logger.warning("Could not directly set _max_workers on executor. Scaling may not be fully dynamic.")
            else:
                self._logger.debug(f"Scale up requested, but {new_max_workers} is not greater than current max workers {self.max_workers}.")

    def scale_down(self, new_max_workers):
        """Logs a recommendation to scale down workers (ThreadPoolExecutor limitation)."""
        with self._lock:
            if new_max_workers < self.max_workers:
                 self._logger.info(f"Scale down requested to {new_max_workers}. ThreadPoolExecutor does not support dynamic reduction of workers. Consider recreating the pool or using a different executor type for true downscaling.")
                 # Note: Implementing actual scale down with ThreadPoolExecutor is complex
                 # and typically involves shutting down and replacing the executor,
                 # managing running tasks and the queue during the transition.
            else:
                 self._logger.debug(f"Scale down requested, but {new_max_workers} is not less than current max workers {self.max_workers}.")


    def shutdown(self):
        """Signals the worker pool to stop and waits for current tasks to finish."""
        self._logger.info("Shutting down worker pool.")
        self._running = False
        # The submitter thread will exit its loop after processing remaining queue items
        # or after the timeout if queue is empty.
        if self._submitter_future and not self._submitter_future.done():
             try:
                 self._submitter_future.result(timeout=5) # Wait for submitter to finish
             except concurrent.futures.TimeoutError:
                 self._logger.warning("Submitter thread did not finish within timeout.")

        self.executor.shutdown(wait=True) # Wait for all submitted tasks to finish
        self._logger.info("Worker pool shut down complete.")
