from celery import Celery
from kombu import Queue
import os
import torch

# --- Configuration ---
# Default to local Redis if not set in environment
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
# Attempt GPU detection (can be overridden by environment)
try:
    DEFAULT_NUM_GPUS = torch.cuda.device_count()
    if DEFAULT_NUM_GPUS == 0:
        print("[Celery Config] Warning: No GPUs detected by torch, defaulting to 1.")
        DEFAULT_NUM_GPUS = 1
except ImportError:
    print("[Celery Config] Warning: torch not installed or CUDA not available, defaulting to 1 GPU.")
    DEFAULT_NUM_GPUS = 1

# Allow overriding NUM_GPUS via environment variable for flexibility
NUM_GPUS = int(os.environ.get("NUM_GPUS", DEFAULT_NUM_GPUS))

# --- Celery App Initialization ---
celery_app = Celery(
    "triton_tasks",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["axolotl_dev.worker"],  # Point to the module containing tasks
)

# --- Task Routing and Queues ---
# Create one queue per GPU
# Example: gpu.0, gpu.1, ...
TASK_QUEUES = tuple(
    Queue(f"gpu.{i}", routing_key=f"gpu.{i}") for i in range(NUM_GPUS)
)

# Configure Celery
celery_app.conf.update(
    task_serializer="json",        # Use JSON for task serialization
    result_serializer="json",      # Use JSON for result serialization
    accept_content=["json"],       # Accept JSON content
    timezone="UTC",                # Use UTC timezone
    enable_utc=True,
    worker_concurrency=1,          # Each worker process handles one task at a time (important for GPU work)
    worker_prefetch_multiplier=1,  # Fetch one task at a time
    task_acks_late=True,           # Acknowledge tasks after they complete/fail (important for reliability)
    task_track_started=True,       # Track when tasks start execution
    # Disable rate limits by default, can be configured per task if needed
    # task_default_rate_limit=None,
    broker_connection_retry_on_startup=True, # Retry connection on startup
    task_queues=TASK_QUEUES,       # Define the available queues
    task_default_queue='gpu.0',    # Default queue if none specified (optional)
)

print(f"[Celery Config] Initialized Celery App. Broker: {REDIS_URL}, Backend: {REDIS_URL}")
print(f"[Celery Config] Configured for {NUM_GPUS} GPUs with queues: {[q.name for q in TASK_QUEUES]}")

# Optional: Import tasks decorated elsewhere (if not using auto-discovery via 'include')
# from .worker import execute_triton_code 