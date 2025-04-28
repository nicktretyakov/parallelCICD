# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY requirements.txt .

# Install any needed dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY .

# Define the command to run your application
# CMD ["python", "main.py", "--config", "pipeline_config.yaml"]
# It's better to use ENTRYPOINT with CMD arguments for flexibility
ENTRYPOINT ["python", "main.py"]
# Default arguments for the ENTRYPOINT
CMD ["--config", "pipeline_config.yaml"]

# Expose the port for Prometheus metrics (if enabled)
# EXPOSE 8000 # Uncomment if you plan to expose metrics from the container
