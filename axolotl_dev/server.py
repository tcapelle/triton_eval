# axolotl_dev/server.py
# ... (imports and console init) ...
import os
import uuid
import multiprocessing
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rich.console import Console
from rich.rule import Rule
from dataclasses import dataclass
from typing import List, Optional

# --- Rich Console Initialization ---
console = Console()
# --- End Initialization ---

# --- Configuration ---
# Attempt GPU detection
try:
    import torch
    NUM_GPUS = torch.cuda.device_count()
    if NUM_GPUS == 0:
        console.print("[server] [yellow]Warning:[/yellow] No GPUs detected by torch, defaulting to 1.")
        NUM_GPUS = 1
except ImportError:
    console.print("[server] [yellow]Warning:[/yellow] torch not installed, defaulting to 1 GPU.")
    NUM_GPUS = 1

CONCURRENCY_PER_GPU = 1
WORKER_COUNT = NUM_GPUS * CONCURRENCY_PER_GPU
TASK_TIMEOUT_SECONDS = 60 # Timeout for each task execution in seconds (e.g., 2 minutes)
WORKER_JOIN_TIMEOUT = 10 # Seconds to wait for worker processes to join gracefully

# Queues and shared state
task_queue = multiprocessing.Queue()
result_queue = multiprocessing.Queue()
in_flight_requests = {}
workers: List[multiprocessing.Process] = []  # Populated/kept in-sync by WorkerPool for BC
# A single lock lives inside WorkerPool; expose it here for the very few legacy usages.
workers_lock: asyncio.Lock  # Forward declaration – real value assigned after WorkerPool is built.
# --- End Configuration ---

# Pydantic request/response models
class CodeExecutionRequest(BaseModel):
    code: str
    tests: str

class CodeExecutionResponse(BaseModel):
    status_code: int
    stdout: str = ""
    stderr: str = ""

app = FastAPI()

# ---------------------------------------------------------------------------
# Worker pool management
# ---------------------------------------------------------------------------

@dataclass
class WorkerProcess:
    """Lightweight wrapper holding a worker `multiprocessing.Process` and its GPU assignment."""

    process: multiprocessing.Process
    gpu_id: int


class WorkerPool:
    """Manage a fixed-size pool of worker processes.

    The pool keeps one worker per `(GPU, concurrency slot)` pair alive.  If a worker dies
    unexpectedly (e.g. because the executed Triton code segfaulted and brought the process down),
    the pool will transparently spawn a replacement so the overall capacity remains unchanged.
    """

    MONITOR_INTERVAL = 3  # Seconds between liveness checks

    def __init__(self, num_gpus: int, concurrency_per_gpu: int):
        self.num_gpus = num_gpus
        self.concurrency_per_gpu = concurrency_per_gpu
        self.task_queue = task_queue
        self.result_queue = result_queue

        self._workers: List[WorkerProcess] = []

        # An asyncio lock guards the internal state; reused by the API layer for compatibility.
        self.lock: asyncio.Lock = asyncio.Lock()

        self._monitor_task: Optional[asyncio.Task] = None

    # ---------------------------------------------------------------------
    # Public helpers
    # ---------------------------------------------------------------------

    @property
    def active_count(self) -> int:
        """Number of *alive* worker processes."""
        return sum(1 for wp in self._workers if wp.process.is_alive())

    @property
    def is_running(self) -> bool:
        """Whether the pool is considered operational.

        We report *running* as long as the pool has at least one worker process object.
        A worker may have just crashed (active_count == 0) but the monitor will respawn
        it within a couple of seconds; rejecting API requests outright would create
        avoidable 503 spikes.
        """
        return bool(self._workers)

    # ---------------------------------------------------------------------
    # Lifecycle management
    # ---------------------------------------------------------------------

    async def start(self) -> bool:
        """Spawn the full worker pool.

        Returns `True` on fresh start, `False` if workers were already running or failed to start.
        """
        async with self.lock:
            if self._workers:
                console.print("[pool] [yellow]Workers already running. Cannot start again.[/yellow]")
                return False

            # Import locally to avoid fork issues on some platforms.
            from worker import worker_main  # noqa: WPS433 (allow inside function)

            gpu_ids = [gpu for gpu in range(self.num_gpus) for _ in range(self.concurrency_per_gpu)]

            console.print(
                f"[pool] Spawning [cyan]{len(gpu_ids)}[/cyan] worker(s) for [cyan]{self.num_gpus}[/cyan] GPU(s)..."
            )

            new_workers: List[WorkerProcess] = []
            try:
                for gpu_id in gpu_ids:
                    proc = multiprocessing.Process(
                        target=worker_main,
                        args=(self.task_queue, self.result_queue, gpu_id),
                        daemon=True,
                    )
                    proc.start()
                    new_workers.append(WorkerProcess(proc, gpu_id))

                # All good – commit
                self._workers = new_workers
                console.print(f"[pool] [green]Spawned {len(self._workers)} workers.[/green]")
            except Exception as exc:  # noqa: BLE001 – broad except ok at top-level recovery
                console.print(f"[pool] [bold red]Error spawning workers:[/bold red] {exc}")
                # Roll back – terminate any partially started workers
                for wp in new_workers:
                    if wp.process.is_alive():
                        wp.process.terminate()
                self._workers = []
                return False

            # Start background liveness monitor
            if self._monitor_task is None or self._monitor_task.done():
                self._monitor_task = asyncio.create_task(self._monitor_loop())

            # Expose list for backward-compat global variable
            global workers  # noqa: WPS420 (needed for BC)
            workers = [wp.process for wp in self._workers]

            return True

    async def stop(self) -> bool:
        """Gracefully stop all workers."""
        async with self.lock:
            if not self._workers:
                console.print("[pool] [yellow]No workers currently running.[/yellow]")
                return False

            console.print(f"[pool] Sending shutdown signal to [cyan]{len(self._workers)}[/cyan] worker(s)...")

            # Send poison pills
            for _ in self._workers:
                try:
                    self.task_queue.put_nowait((None, None))
                except multiprocessing.queues.Full:
                    console.print("[pool] [yellow]Task queue full while sending poison pills.[/yellow]")
                except Exception as exc:
                    console.print(f"[pool] [red]Error sending poison pill:[/red] {exc}")

            # Wait for workers to exit (best-effort)
            join_tasks = [
                asyncio.to_thread(wp.process.join, timeout=WORKER_JOIN_TIMEOUT) for wp in self._workers
            ]
            results = await asyncio.gather(*join_tasks, return_exceptions=True)

            terminated = 0
            for idx, res in enumerate(results):
                proc = self._workers[idx].process
                if isinstance(res, Exception) or proc.is_alive():
                    proc.terminate()
                    terminated += 1

            if terminated:
                console.print(f"[pool] Force terminated {terminated} unresponsive worker(s).")

            # Clear state and cancel monitor
            self._workers.clear()
            if self._monitor_task:
                self._monitor_task.cancel()
                self._monitor_task = None

            # Sync global alias for backward compatibility
            global workers  # noqa: WPS420
            workers = []

            return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _monitor_loop(self) -> None:
        """Background task keeping the pool size constant."""
        console.print("[pool] [italic]Worker monitor task started.[/italic]")

        from worker import worker_main  # Local import to avoid fork issues

        while True:
            await asyncio.sleep(self.MONITOR_INTERVAL)

            async with self.lock:
                # Split into alive / dead buckets
                alive: List[WorkerProcess] = []
                replacements: List[WorkerProcess] = []

                for wp in self._workers:
                    if wp.process.is_alive():
                        alive.append(wp)
                    else:
                        console.print(
                            f"[pool] [yellow]Detected dead worker PID {wp.process.pid} (GPU {wp.gpu_id}). Respawning...[/yellow]"
                        )
                        try:
                            proc = multiprocessing.Process(
                                target=worker_main,
                                args=(self.task_queue, self.result_queue, wp.gpu_id),
                                daemon=True,
                            )
                            proc.start()
                            replacements.append(WorkerProcess(proc, wp.gpu_id))
                        except Exception as exc:
                            console.print(
                                f"[pool] [bold red]Failed to respawn worker on GPU {wp.gpu_id}:[/bold red] {exc}"
                            )

                if replacements:
                    console.print(f"[pool] [green]Respawned {len(replacements)} worker(s).[/green]")

                # Update internal & global state
                self._workers = alive + replacements
                global workers  # noqa: WPS420 – keep alias in-sync
                workers = [wp.process for wp in self._workers]

# Instantiate the pool *once* so the whole module shares it
worker_pool = WorkerPool(NUM_GPUS, CONCURRENCY_PER_GPU)

# Expose the lock to the original variable name so the remainder of the file compiles unmodified.
workers_lock = worker_pool.lock

async def spawn_workers_internal():
    """Shim kept for backwards compatibility – delegates to WorkerPool."""
    return await worker_pool.start()

async def kill_workers_internal():
    """Shim kept for backwards compatibility – delegates to WorkerPool."""
    # Cancel all pending request futures *before* shutting down the workers so callers
    # receive a clear signal.
    num_cancelled = 0
    for task_id, fut in list(in_flight_requests.items()):
        if not fut.done():
            fut.cancel()
            num_cancelled += 1
        in_flight_requests.pop(task_id, None)
    if num_cancelled:
        console.print(f"[server] [yellow]Cancelled {num_cancelled} pending requests due to worker stop.[/yellow]")

    return await worker_pool.stop()

async def result_collector():
    """Background task to collect results."""
    console.print("[server] [italic]Result collector task started.[/italic]")
    while True:
        try:
            result = await asyncio.to_thread(result_queue.get)
            task_id = result.get("task_id")
            if task_id in in_flight_requests:
                fut = in_flight_requests.pop(task_id)
                if not fut.done(): # Avoid setting result on already cancelled future
                    fut.set_result(result)
            # else: # No need to log warning if futures might be cancelled by stop_workers
                # console.print(f"[server] [yellow]Warning:[/yellow] Received result for unknown/old/cancelled task_id: {task_id}")
        except Exception as e:
            console.print(f"[server] [bold red]Error in result collector:[/bold red] {e}")
            await asyncio.sleep(1)


@app.on_event("startup")
async def on_startup():
    """Start background tasks, but not workers."""
    console.print(Rule("[bold blue]Server Startup[/bold blue]"))
    # Don't spawn workers here
    asyncio.create_task(result_collector())
    console.print("[server] [bold green]Startup complete. Ready for requests.[/bold green]")
    console.print("[server] [bold yellow]Workers are NOT started automatically. Use POST /start_workers.[/bold yellow]")


@app.on_event("shutdown")
async def on_shutdown():
    """Ensure workers are stopped on shutdown."""
    console.print(Rule("[bold blue]Server Shutdown[/bold blue]"))
    await kill_workers_internal()
    console.print("[server] [bold green]Shutdown complete.[/bold green]")

# --- New Endpoints ---
@app.post("/start_workers")
async def start_workers_endpoint():
    """Endpoint to start the worker processes."""
    success = await spawn_workers_internal()
    if success:
        return {"message": f"{worker_pool.active_count} workers started successfully."}
    else:
        # Use status code to indicate failure if workers were already running or failed to start
        # Check global 'workers' list state *after* attempting spawn
        status = 409 if worker_pool.is_running else 500
        detail = "Workers already running." if status == 409 else "Failed to start workers."
        raise HTTPException(status_code=status, detail=detail)


@app.post("/stop_workers")
async def stop_workers_endpoint():
    """Endpoint to stop the worker processes."""
    success = await kill_workers_internal()
    if success:
        return {"message": "Workers stopped successfully."}
    else:
        # This case (returning False) only happens if kill_workers_internal finds no workers running
        raise HTTPException(status_code=409, detail="No workers were running.")
# --- End New Endpoints ---

@app.get("/")
def read_root():
    return {"message": "Triton Worker Pool Server is ready!"}


@app.post("/run_code", response_model=CodeExecutionResponse)
async def run_code_endpoint(request: CodeExecutionRequest):
    """API endpoint to execute code."""
    # Add check to ensure workers are running (read access, no lock needed here)
    if not worker_pool.is_running:
        raise HTTPException(
            status_code=503,
            detail="Workers are not running. Please start workers via POST /start_workers.",
        )

    task_id = str(uuid.uuid4())
    code_string = (
        "import torch\n"
        "import triton\n"
        "import triton.language as tl\n\n"
        f"{request.code}\n\n"
        "# ---- Tests Below ----\n"
        "DEVICE = torch.device('cuda')\n"
        f"{request.tests}\n"
    )

    loop = asyncio.get_event_loop()
    fut = loop.create_future()

    # Check if workers are available before adding to queue/in_flight
    # Acquire lock briefly only to add the future, reducing contention
    async with workers_lock:
        if not worker_pool.is_running:  # Re-check in case they were stopped before we acquired the lock
            raise HTTPException(
                status_code=503,
                detail="Workers were stopped before the request could be processed.",
            )
        in_flight_requests[task_id] = fut  # Add future only if pool is confirmed running

    console.print(f"[server] Received request, assigning Task ID: {task_id}")
    try:
        task_queue.put_nowait((task_id, code_string)) # Use put_nowait, assuming queue isn't the bottleneck
        console.print(f"[server] Task {task_id} added to queue.")
    except multiprocessing.queues.Full:
         in_flight_requests.pop(task_id, None) # Clean up future
         console.print(f"[server] Task {task_id} [bold red]rejected[/bold red]: Task queue is full.")
         raise HTTPException(status_code=503, detail="Server busy, task queue is full.")
    except Exception as e:
        in_flight_requests.pop(task_id, None) # Clean up future
        console.print(f"[server] Task {task_id} [bold red]rejected[/bold red]: Error adding to queue: {e}")
        raise HTTPException(status_code=500, detail="Internal server error queuing task.")

    try:
        console.print(f"[server] Waiting for task {task_id} (timeout: [yellow]{TASK_TIMEOUT_SECONDS}s[/yellow])...")
        result = await asyncio.wait_for(fut, timeout=TASK_TIMEOUT_SECONDS)
        console.print(f"[server] Task {task_id} [green]completed[/green].")
        return CodeExecutionResponse(
            status_code=result["status_code"],
            stdout=result["stdout"],
            stderr=result["stderr"],
        )
    except asyncio.TimeoutError:
        console.print(f"[server] Task {task_id} [bold red]timed out[/bold red] after {TASK_TIMEOUT_SECONDS} seconds.")
        console.print(f"[server] [yellow]Hint:[/yellow] This might happen if the task took too long, or if the assigned worker process terminated unexpectedly.")

        # The dedicated monitor inside WorkerPool will take care of replacing dead workers.
        # We just log the timeout here – no manual intervention needed anymore.

        # Note: Future might have already been removed if stop_workers cancelled it
        in_flight_requests.pop(task_id, None) # Clean up future for timed-out task
        raise HTTPException(status_code=504, detail=f"Task execution timed out after {TASK_TIMEOUT_SECONDS} seconds.")
    except asyncio.CancelledError:
         console.print(f"[server] Task {task_id} [yellow]cancelled[/yellow], likely due to worker shutdown.")
         # Future is already removed/cancelled by kill_workers_internal
         raise HTTPException(status_code=503, detail="Task cancelled, workers may have been stopped.")
    except Exception as e:
         console.print(f"[server] [bold red]Error processing result[/bold red] for task {task_id}: {e}")
         in_flight_requests.pop(task_id, None)
         raise HTTPException(status_code=500, detail="Internal server error processing task result.")

# If you want to run with uvicorn directly from this file:
if __name__ == "__main__":
    import uvicorn
    console.print(Rule("[bold blue]Starting Uvicorn[/bold blue]"))
    console.print(f"Host: 0.0.0.0, Port: 9347")
    # Run uvicorn in the main process; worker spawning happens via endpoint
    uvicorn.run("server:app", host="0.0.0.0", port=9347, reload=False, log_config=None)
