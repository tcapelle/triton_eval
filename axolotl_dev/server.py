# axolotl_dev/server.py
import os
import uuid
import asyncio
import itertools
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from rich.console import Console
from rich.rule import Rule
from celery.result import AsyncResult
from typing import Optional, Dict, Any

# Import Celery app instance and task definition
from axolotl_dev.celery_config import celery_app, NUM_GPUS
from axolotl_dev.worker import execute_triton_code

# --- Rich Console Initialization ---
console = Console()
# --- End Initialization ---

# --- Globals ---
# Simple round-robin counter for GPU queue assignment
# In a multi-instance server setup, a more robust distribution mechanism (e.g., Redis counter) might be needed.
gpu_rr_counter = itertools.cycle(range(NUM_GPUS))

# --- Pydantic Models ---
class CodeExecutionRequest(BaseModel):
    code: str
    tests: str

class TaskSubmissionResponse(BaseModel):
    task_id: str
    message: str
    queue: str

class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[Dict[str, Any]] = None # Holds the output {status_code, stdout, stderr}

# --- FastAPI App Initialization ---
# No lifespan management needed here as workers are managed externally by Celery
app = FastAPI()

# --- API Endpoints ---

@app.get("/")
def read_root():
    return {"message": "Triton Celery Task Server is ready!"}

@app.post("/run_code", response_model=TaskSubmissionResponse, status_code=202)
async def run_code_endpoint(request: CodeExecutionRequest):
    """API endpoint to submit code execution task to Celery."""
    try:
        # Select next GPU queue using round-robin
        gpu_index = next(gpu_rr_counter)
        target_queue = f"gpu.{gpu_index}"
        console.print(f"[server] Assigning task to queue: {target_queue}")

        # Send task to the specific GPU queue
        task = execute_triton_code.apply_async(
            args=[request.code, request.tests],
            queue=target_queue,
            # Optional: Add task headers, priorities, etc.
        )

        console.print(f"[server] Submitted task {task.id} to queue {target_queue}")
        return TaskSubmissionResponse(
            task_id=task.id,
            message="Task submitted successfully",
            queue=target_queue
        )
    except Exception as e:
        console.print(f"[server] [bold red]Error submitting task:[/bold red] {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error submitting task: {e}")

@app.get("/results/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """API endpoint to check the status and result of a task."""
    try:
        task_result = AsyncResult(task_id, app=celery_app)

        response_data = {
            "task_id": task_id,
            "status": task_result.status,
            "result": None,
        }

        if task_result.ready():
            if task_result.successful():
                # Result is the dict returned by execute_triton_code
                result_value = task_result.get()
                response_data["result"] = result_value
                console.print(f"[server] Task {task_id} completed successfully.")
            else:
                # Task failed, result contains the exception
                # We store the traceback in the 'stderr' of our custom result dict
                try:
                    # Try to get the result which might contain our custom dict
                    result_value = task_result.get()
                    if isinstance(result_value, dict):
                         response_data["result"] = result_value
                    else: # If result isn't our dict, construct error info
                         response_data["result"] = {
                              "status_code": -1,
                              "stdout": "",
                              "stderr": str(task_result.info or task_result.result)
                         }
                except Exception as e:
                    # Fallback if getting result fails
                    console.print(f"[server] [yellow]Task {task_id} failed. Error getting exception info: {e}[/yellow]")
                    response_data["result"] = {
                        "status_code": -1,
                        "stdout": "",
                        "stderr": f"Task failed, unable to retrieve specific error: {e}"
                    }
                console.print(f"[server] Task {task_id} failed.")
        else:
             console.print(f"[server] Task {task_id} status: {task_result.status}")

        return TaskStatusResponse(**response_data)

    except Exception as e:
        console.print(f"[server] [bold red]Error checking task status for {task_id}:[/bold red] {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error checking task status: {e}")

@app.post("/control/revoke/{task_id}")
async def revoke_task(task_id: str):
    """Revokes (cancels) a running or pending task."""
    try:
        console.print(f"[server] Attempting to revoke task {task_id}...")
        # Use terminate=True to attempt to kill the running process (requires OS signals)
        # Use signal='SIGTERM' or 'SIGKILL' if needed
        celery_app.control.revoke(task_id, terminate=True, signal='SIGTERM')
        console.print(f"[server] Revocation signal sent for task {task_id}.")
        return {"message": f"Revocation signal sent for task {task_id}. Check worker logs for confirmation.", "task_id": task_id}
    except Exception as e:
        console.print(f"[server] [bold red]Error revoking task {task_id}:[/bold red] {e}")
        raise HTTPException(status_code=500, detail=f"Error revoking task: {e}")

@app.post("/control/purge_queue/{queue_name}")
async def purge_queue(queue_name: str):
    """Removes all pending tasks from a specified queue."""
    try:
        num_purged = celery_app.control.purge(queues=[queue_name])
        msg = f"Purged {num_purged} tasks from queue '{queue_name}'."
        console.print(f"[server] {msg}")
        return {"message": msg, "queue": queue_name, "purged_count": num_purged}
    except Exception as e:
        console.print(f"[server] [bold red]Error purging queue {queue_name}:[/bold red] {e}")
        raise HTTPException(status_code=500, detail=f"Error purging queue: {e}")

# Removed old /reset_workers endpoint as worker lifecycle is now managed externally.

# --- Uvicorn Runner ---
if __name__ == "__main__":
    import uvicorn
    console.print(Rule("[bold blue]Starting Uvicorn for Celery Task Server[/bold blue]"))
    console.print(f"Host: 0.0.0.0, Port: 9347")
    console.print(f"Detected {NUM_GPUS} GPUs for queue assignment.")
    # Uvicorn runs the FastAPI app; Celery workers run separately.
    uvicorn.run("axolotl_dev.server:app", host="0.0.0.0", port=9347, reload=False, log_config=None)
