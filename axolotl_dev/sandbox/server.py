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
from contextlib import asynccontextmanager

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
TASK_TIMEOUT_SECONDS = int(os.getenv("TASK_TIMEOUT_SECONDS", 180)) # Timeout for each task execution in seconds (e.g., 2 minutes)
WORKER_JOIN_TIMEOUT = int(os.getenv("WORKER_JOIN_TIMEOUT", 20)) # Seconds to wait for worker processes to join gracefully

# Queues and shared state
task_queue = multiprocessing.Queue()
result_queue = multiprocessing.Queue()
in_flight_requests = {}
workers: List[multiprocessing.Process] = []  # Populated/kept in-sync by WorkerPool for BC
# A single lock lives inside WorkerPool; expose it here for the very few legacy usages.
workers_lock: asyncio.Lock  # Forward declaration – real value assigned after WorkerPool is built.
# --- End Configuration ---

# Pydantic request/response models
class TritonExecutionRequest(BaseModel):
    code: str
    tests: str
    benchmark: bool = False  # New flag to enable benchmarking
    benchmark_runs: int = 10  # Number of benchmark runs

class PyTorchExecutionRequest(BaseModel):
    code: str
    tests: str
    benchmark: bool = False  # Enable benchmarking
    benchmark_runs: int = 10  # Number of benchmark runs
    torch_compile: bool = False  # Enable torch.compile benchmarking
    torch_compile_mode: str = "default"  # torch.compile mode: "default", "reduce-overhead", "max-autotune"
    entrypoint: Optional[str] = None  # Function name to torch.compile (if not specified, will try to find benchmark_function)

class CodeExecutionResponse(BaseModel):
    status_code: int
    stdout: str = ""
    stderr: str = ""
    # Added fields for metrics
    gpu_mem_used_gb: Optional[float] = None
    cpu_percent: Optional[float] = None
    ram_percent: Optional[float] = None
    # Benchmarking results (only populated if enable_benchmarking=True and code runs successfully)
    benchmark_mean_time_ms: Optional[float] = None
    benchmark_std_time_ms: Optional[float] = None
    benchmark_memory_peak_mb: Optional[float] = None
    benchmark_successful_runs: Optional[int] = None
    # PyTorch specific results
    torch_compile_benchmark_mean_time_ms: Optional[float] = None
    torch_compile_benchmark_std_time_ms: Optional[float] = None
    torch_compile_speedup: Optional[float] = None  # Speedup ratio of compiled vs regular

# --- Lifespan Context Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    console.print(Rule("[bold blue]Server Startup (lifespan)[/bold blue]"))
    # Start workers automatically
    success = await worker_pool.start()
    if not success:
        console.print("[server] [bold red]CRITICAL: Failed to start initial worker pool on startup![/bold red]")
        # Consider exiting or handling this failure scenario appropriately
    else:
        console.print(f"[server] [green]Initial worker pool started with {worker_pool.active_count} worker(s).[/green]")

    collector_task = asyncio.create_task(result_collector())
    console.print("[server] [bold green]Startup complete. Ready for requests.[/bold green]")

    yield # Server runs here

    # Shutdown logic
    console.print(Rule("[bold blue]Server Shutdown (lifespan)[/bold blue]"))
    await kill_workers_internal()
    # Optionally cancel the collector task if it hasn't finished
    if collector_task and not collector_task.done():
        collector_task.cancel()
        try:
            await collector_task # Wait for cancellation to complete
        except asyncio.CancelledError:
            console.print("[server] Result collector task cancelled successfully.")
    console.print("[server] [bold green]Shutdown complete.[/bold green]")

app = FastAPI(lifespan=lifespan)

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

    MONITOR_INTERVAL = 1  # Seconds between liveness checks (Reduced from 3)

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
        """Whether the pool is considered operational and has active workers.

        We consider the pool *running* only if there is at least one underlying
        worker object **and** at least one of those processes is currently
        alive.  This prevents the API from accepting requests when all workers
        have died (which would otherwise lead to timeouts).
        """
        return any(wp.process.is_alive() for wp in self._workers)

    # ---------------------------------------------------------------------
    # Lifecycle management
    # ---------------------------------------------------------------------

    async def start(self) -> bool:
        """Spawn the full worker pool.

        Returns `True` on fresh start, `False` if workers were already running or failed to start.
        """
        async with self.lock:
            # Fast path: If there are *alive* workers, consider the pool already running.
            if any(wp.process.is_alive() for wp in self._workers):
                console.print("[pool] [yellow]Workers already running. Cannot start again.[/yellow]")
                return False

            # If there are worker objects but none are alive (e.g. all crashed), clear them so we can respawn.
            if self._workers and not any(wp.process.is_alive() for wp in self._workers):
                console.print("[pool] [yellow]Found existing worker objects but none are alive. Cleaning up before restart...[/yellow]")
                self._workers.clear()

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

            console.print(f"[pool] Stopping [cyan]{len(self._workers)}[/cyan] worker(s)...")

            # 1. Cancel monitor loop *first* to prevent interference
            if self._monitor_task and not self._monitor_task.done():
                self._monitor_task.cancel()
                try:
                    await self._monitor_task # Wait for cancellation
                except asyncio.CancelledError:
                    console.print("[pool] Monitor task cancelled successfully.")
                except Exception as e:
                    console.print(f"[pool] [red]Error awaiting cancelled monitor task:[/red] {e}")
            self._monitor_task = None

            # 2. Send poison pills
            console.print(f"[pool] Sending poison pills...")
            for _ in self._workers:
                try:
                    self.task_queue.put_nowait(None)  # Updated poison pill format
                except multiprocessing.queues.Full:
                    console.print("[pool] [yellow]Task queue full while sending poison pills.[/yellow]")
                except Exception as exc:
                    console.print(f"[pool] [red]Error sending poison pill:[/red] {exc}")

            # 3. Wait for workers to exit (best-effort)
            console.print(f"[pool] Waiting for workers to join (timeout: {WORKER_JOIN_TIMEOUT}s)...")
            join_tasks = [
                asyncio.to_thread(wp.process.join, timeout=WORKER_JOIN_TIMEOUT) for wp in self._workers
            ]
            results = await asyncio.gather(*join_tasks, return_exceptions=True)

            terminated = 0
            alive_after_join = []
            for idx, res in enumerate(results):
                proc = self._workers[idx].process
                if proc.is_alive(): # Check if still alive after join attempt
                    alive_after_join.append(proc.pid)
                    console.print(f"[pool] [yellow]Worker PID {proc.pid} did not exit within timeout. Terminating...[/yellow]") # Added logging
                    proc.terminate()
                    terminated += 1
                elif isinstance(res, Exception):
                     # Log exceptions during join, but process might have exited cleanly anyway
                     console.print(f"[pool] [yellow]Error joining worker PID {proc.pid}: {res}[/yellow]")

            if terminated > 0:
                console.print(f"[pool] Force terminated {terminated} unresponsive worker(s): PIDs {alive_after_join}")
            else:
                console.print(f"[pool] All workers joined gracefully.")

            # 4. Clear internal state
            self._workers.clear()
            console.print("[pool] Worker list cleared.")

            # Sync global alias for backward compatibility
            global workers  # noqa: WPS420
            workers = []

            console.print("[pool] [green]Stop sequence complete.[/green]")
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

@app.get("/")
def read_root():
    return {"message": "Triton Worker Pool Server is ready!"}


@app.post("/run_triton", response_model=CodeExecutionResponse)
async def run_triton_endpoint(request: TritonExecutionRequest):
    """API endpoint to execute Triton code."""
    # Workers should now always be running unless startup failed or they crashed.
    # Keep the check to handle edge cases like crashes or failed resets.
    if not worker_pool.is_running:
        # Attempt to (re)start the worker pool automatically instead of immediately failing.
        console.print("[server] [yellow]Worker pool not running. Attempting automatic restart...[/yellow]")
        started = await worker_pool.start()
        if not started:
            console.print("[server] [bold red]Automatic worker pool restart failed.[/bold red]")
            raise HTTPException(
                status_code=503,
                detail="Workers are not currently operational and automatic restart failed.",
            )

    task_id = str(uuid.uuid4())
    code_string = (
        "from typing import *\n"
        "import torch\n"
        "import triton\n"
        "import triton.language as tl\n\n"
        f"{request.code}\n\n"
        "# ---- Tests Below ----\n"
        "DEVICE = torch.device('cuda')\n"
        f"{request.tests}\n"
    )
    
    # Create task data with benchmarking info
    task_data = {
        "task_id": task_id,
        "task_type": "triton",
        "code": code_string,
        "benchmark": request.benchmark,
        "benchmark_runs": request.benchmark_runs
    }

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
        # Offload the potentially blocking put operation to a thread so the event loop remains responsive.
        await asyncio.to_thread(task_queue.put, task_data)
        # qsize() may not be implemented on some platforms; fall back gracefully.
        try:
            q_sz = task_queue.qsize()
        except (NotImplementedError, AttributeError):
            q_sz = "unknown"
        console.print(f"[server] Task {task_id} added to queue (queue size: {q_sz}).")
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
            # Extract metrics from result dict
            gpu_mem_used_gb=result.get("gpu_mem_used_gb"),
            cpu_percent=result.get("cpu_percent"),
            ram_percent=result.get("ram_percent"),
            # Extract benchmark metrics
            benchmark_mean_time_ms=result.get("benchmark_mean_time_ms"),
            benchmark_std_time_ms=result.get("benchmark_std_time_ms"),
            benchmark_memory_peak_mb=result.get("benchmark_memory_peak_mb"),
            benchmark_successful_runs=result.get("benchmark_successful_runs"),
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
         # Future is already removed/cancelled by kill_workers_internal
         # Log cancellation without raising HTTPException immediately, as it might be part of a controlled stop/reset
         console.print(f"[server] Task {task_id} [yellow]cancelled[/yellow], likely due to worker shutdown or reset.")
         # Ensure the future is removed from in_flight_requests
         in_flight_requests.pop(task_id, None)
         # Raise the HTTPException to inform the client
         raise HTTPException(status_code=503, detail="Task cancelled, workers may have been stopped or reset.")
    except Exception as e:
         console.print(f"[server] [bold red]Error processing result[/bold red] for task {task_id}: {e}")
         in_flight_requests.pop(task_id, None)
         raise HTTPException(status_code=500, detail="Internal server error processing task result.")

@app.post("/run_pytorch", response_model=CodeExecutionResponse)
async def run_pytorch_endpoint(request: PyTorchExecutionRequest):
    """API endpoint to execute PyTorch code with optional torch.compile benchmarking."""
    # Workers should now always be running unless startup failed or they crashed.
    # Keep the check to handle edge cases like crashes or failed resets.
    if not worker_pool.is_running:
        # Attempt to (re)start the worker pool automatically instead of immediately failing.
        console.print("[server] [yellow]Worker pool not running. Attempting automatic restart...[/yellow]")
        started = await worker_pool.start()
        if not started:
            console.print("[server] [bold red]Automatic worker pool restart failed.[/bold red]")
            raise HTTPException(
                status_code=503,
                detail="Workers are not currently operational and automatic restart failed.",
            )

    task_id = str(uuid.uuid4())
    code_string = (
        "from typing import *\n"
        "import torch\n"
        "import torch.nn as nn\n"
        "import torch.nn.functional as F\n"
        "import time\n"
        "import gc\n\n"
        f"{request.code}\n\n"
        "# ---- Tests Below ----\n"
        "DEVICE = torch.device('cuda')\n"
        f"{request.tests}\n"
    )
    
    # Create task data with PyTorch-specific info
    task_data = {
        "task_id": task_id,
        "task_type": "pytorch",
        "code": code_string,
        "benchmark": request.benchmark,
        "benchmark_runs": request.benchmark_runs,
        "torch_compile": request.torch_compile,
        "torch_compile_mode": request.torch_compile_mode,
        "entrypoint": request.entrypoint
    }

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

    console.print(f"[server] Received PyTorch request, assigning Task ID: {task_id}")
    try:
        # Offload the potentially blocking put operation to a thread so the event loop remains responsive.
        await asyncio.to_thread(task_queue.put, task_data)
        # qsize() may not be implemented on some platforms; fall back gracefully.
        try:
            q_sz = task_queue.qsize()
        except (NotImplementedError, AttributeError):
            q_sz = "unknown"
        console.print(f"[server] Task {task_id} added to queue (queue size: {q_sz}).")
    except Exception as e:
        in_flight_requests.pop(task_id, None) # Clean up future
        console.print(f"[server] Task {task_id} [bold red]rejected[/bold red]: Error adding to queue: {e}")
        raise HTTPException(status_code=500, detail="Internal server error queuing task.")

    try:
        console.print(f"[server] Waiting for PyTorch task {task_id} (timeout: [yellow]{TASK_TIMEOUT_SECONDS}s[/yellow])...")
        result = await asyncio.wait_for(fut, timeout=TASK_TIMEOUT_SECONDS)
        console.print(f"[server] PyTorch task {task_id} [green]completed[/green].")
        return CodeExecutionResponse(
            status_code=result["status_code"],
            stdout=result["stdout"],
            stderr=result["stderr"],
            # Extract metrics from result dict
            gpu_mem_used_gb=result.get("gpu_mem_used_gb"),
            cpu_percent=result.get("cpu_percent"),
            ram_percent=result.get("ram_percent"),
            # Extract benchmark metrics
            benchmark_mean_time_ms=result.get("benchmark_mean_time_ms"),
            benchmark_std_time_ms=result.get("benchmark_std_time_ms"),
            benchmark_memory_peak_mb=result.get("benchmark_memory_peak_mb"),
            benchmark_successful_runs=result.get("benchmark_successful_runs"),
            # Extract PyTorch-specific metrics
            torch_compile_benchmark_mean_time_ms=result.get("torch_compile_benchmark_mean_time_ms"),
            torch_compile_benchmark_std_time_ms=result.get("torch_compile_benchmark_std_time_ms"),
            torch_compile_speedup=result.get("torch_compile_speedup"),
        )
    except asyncio.TimeoutError:
        console.print(f"[server] PyTorch task {task_id} [bold red]timed out[/bold red] after {TASK_TIMEOUT_SECONDS} seconds.")
        console.print(f"[server] [yellow]Hint:[/yellow] This might happen if the task took too long, or if the assigned worker process terminated unexpectedly.")

        # The dedicated monitor inside WorkerPool will take care of replacing dead workers.
        # We just log the timeout here – no manual intervention needed anymore.

        # Note: Future might have already been removed if stop_workers cancelled it
        in_flight_requests.pop(task_id, None) # Clean up future for timed-out task
        raise HTTPException(status_code=504, detail=f"Task execution timed out after {TASK_TIMEOUT_SECONDS} seconds.")
    except asyncio.CancelledError:
         # Future is already removed/cancelled by kill_workers_internal
         # Log cancellation without raising HTTPException immediately, as it might be part of a controlled stop/reset
         console.print(f"[server] PyTorch task {task_id} [yellow]cancelled[/yellow], likely due to worker shutdown or reset.")
         # Ensure the future is removed from in_flight_requests
         in_flight_requests.pop(task_id, None)
         # Raise the HTTPException to inform the client
         raise HTTPException(status_code=503, detail="Task cancelled, workers may have been stopped or reset.")
    except Exception as e:
         console.print(f"[server] [bold red]Error processing result[/bold red] for PyTorch task {task_id}: {e}")
         in_flight_requests.pop(task_id, None)
         raise HTTPException(status_code=500, detail="Internal server error processing task result.")

# --- API Endpoints ---

@app.post("/reset_workers")
async def reset_workers_endpoint():
    """Stops all current workers and starts a fresh pool."""
    console.print("[server] [bold yellow]Resetting workers...[/bold yellow]")
    await kill_workers_internal()
    success = await worker_pool.start()
    if success:
        console.print(f"[server] [green]Workers reset successfully. New worker pool started with {worker_pool.active_count} worker(s).[/green]")
        return {"message": "Workers reset successfully."}
    else:
        console.print("[server] [bold red]Failed to reset workers.[/bold red]")
        raise HTTPException(status_code=500, detail="Failed to reset workers.")


# If you want to run with uvicorn directly from this file:
if __name__ == "__main__":
    import uvicorn
    console.print(Rule("[bold blue]Starting Uvicorn[/bold blue]"))
    console.print(f"Host: 0.0.0.0, Port: 9347")
    # Run uvicorn in the main process; worker spawning happens via lifespan
    uvicorn.run("server:app", host="0.0.0.0", port=9347, reload=False, log_config=None)
