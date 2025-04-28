# Handles task dependencies using a library like networkx.

import networkx as nx
import logging

class DependencyManager:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.completed_tasks = set()
        self.failed_tasks = set() # Added set to track permanently failed tasks
        self._tasks = {}
        self._logger = logging.getLogger(__name__)

    def add_task(self, task):
        """Adds a task to the dependency graph."""
        if task.name in self._tasks:
            self._logger.warning(f"Task '{task.name}' already exists in the dependency manager.")
            return
        self.graph.add_node(task.name)
        self._tasks[task.name] = task
        self._logger.info(f"Added task '{task.name}' to the dependency graph.")

    def add_dependency(self, task, dependency):
        """Adds a dependency: 'task' depends on 'dependency' (dependency -> task)."""\
        if task.name not in self._tasks:
            self._logger.error(f"Task '{task.name}' not found. Add task first.")
            return
        if dependency.name not in self._tasks:
            self._logger.error(f"Dependency task '{dependency.name}' not found. Add dependency task first.")
            return
        # Add edge from dependency to task, meaning dependency must run before task
        self.graph.add_edge(dependency.name, task.name)
        self._logger.info(f"Added dependency: '{task.name}' depends on '{dependency.name}'.")

    def get_runnable_tasks(self):
        """Returns a list of tasks that are ready to be run (dependencies met and not completed or failed)."""
        runnable = []
        # Iterate through all tasks known to the manager
        for task_name, task in self._tasks.items():
            # Check if the task has already been completed or failed
            if task_name in self.completed_tasks or task_name in self.failed_tasks:
                continue

            # Check if all dependencies of this task are in the completed_tasks set
            # Note: If a dependency failed, the current task will not have its dependencies met
            dependencies_met = True
            # Get predecessors in the graph (tasks that must run before this one)
            for predecessor_name in self.graph.predecessors(task_name):
                # A dependency is met only if it is *completed*
                if predecessor_name not in self.completed_tasks:
                    dependencies_met = False
                    break # Found an incomplete or failed dependency, this task is not runnable yet

            # If all dependencies are met and the task is not completed/failed, it's runnable
            if dependencies_met:
                runnable.append(task)
                self._logger.debug(f"Task '{task_name}' is runnable.")

        return runnable

    def mark_task_complete(self, task):
        """Marks a task as completed."""
        if task.name not in self._tasks:
            self._logger.warning(f"Attempted to mark non-existent task '{task.name}' as complete.")
            return
        self.completed_tasks.add(task.name)
        # Ensure it's not in failed_tasks if it was somehow added there previously
        self.failed_tasks.discard(task.name)
        self._logger.info(f"Task '{task.name}' marked as complete.")

    def mark_task_failed(self, task):
        """Marks a task as permanently failed."""
        if task.name not in self._tasks:
            self._logger.warning(f"Attempted to mark non-existent task '{task.name}' as failed.")
            return
        self.failed_tasks.add(task.name)
        # Ensure it's not in completed_tasks
        self.completed_tasks.discard(task.name)
        self._logger.error(f"Task '{task.name}' marked as failed.")

    def all_tasks_finished(self):
        """Checks if all added tasks have finished (either completed or failed)."""
        return (len(self.completed_tasks) + len(self.failed_tasks)) == len(self._tasks)

    def has_failed_tasks(self):
        """Checks if any task has permanently failed."""
        return len(self.failed_tasks) > 0

    def has_cycle(self):
        """Checks if the dependency graph contains a cycle."""\
        try:
            nx.find_cycle(self.graph)
            self._logger.error("Dependency graph contains a cycle!")
            return True
        except nx.NetworkXNoCycle:
            return False

