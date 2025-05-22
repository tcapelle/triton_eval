# worker.py

import os
import sys
import io
import traceback
import torch
import triton
import triton.language as tl
import triton.compiler.errors # Import Triton compiler errors
import math

import tempfile # Add tempfile
import importlib.util # Add importlib.util
import uuid # Add uuid for unique module names

# Added imports for monitoring
import pynvml
import psutil

def is_fatal_error(exception) -> bool:
    """Checks if an exception should cause the worker to terminate."""
    # Triton compilation errors are considered fatal
    if isinstance(exception, triton.compiler.errors.CompilationError):
        return True

    # Check if the traceback contains uppercase "CUDA" (e.g., CUDA OOM)
    tb_str = traceback.format_exc() # Get the full traceback string
    if "CUDA" in tb_str:
        return True

    return False

def worker_main(task_queue, result_queue, gpu_id):
    """
    Each worker process runs this function:
      - Pins itself to a specific GPU
      - Waits for code from the master process (server)
      - Executes the code via exec()
      - Returns stdout/stderr, status_code, and resource metrics
      - Exits only if a CUDA/Triton compilation error occurs, otherwise reports errors and continues.
    """
    # Pin GPU via environment (so torch sees only 1 device)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    original_stderr_for_logging = sys.__stderr__ # Capture original stderr early

    print(f"[Worker PID {os.getpid()}] Bound to GPU {gpu_id}. Initializing monitoring...", file=original_stderr_for_logging, flush=True)

    # --- Monitoring Setup ---
    nvml_initialized = False
    gpu_handle = None
    try:
        pynvml.nvmlInit()
        nvml_initialized = True
        # Even though CUDA_VISIBLE_DEVICES is set, nvml still sees all GPUs.
        # We need to get the handle for the *correct* GPU based on the original gpu_id.
        gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        print(f"[Worker PID {os.getpid()}] NVML initialized successfully for GPU {gpu_id}.", file=original_stderr_for_logging, flush=True)
    except pynvml.NVMLError as nvml_err:
        print(f"[Worker PID {os.getpid()}] [WARN] Failed to initialize NVML or get GPU handle: {nvml_err}", file=original_stderr_for_logging, flush=True)
        # Worker can continue, but GPU metrics won't be available
    # --- End Monitoring Setup ---


    print(f"[Worker PID {os.getpid()}] Importing libraries...", file=original_stderr_for_logging, flush=True)

    while True:
        task_id, code_string = task_queue.get()
        if code_string is None:
            print(f"[Worker PID {os.getpid()}] Received poison pill. Exiting.", file=original_stderr_for_logging, flush=True)
            break # Exit loop cleanly

        # Prepare to capture stdout/stderr
        old_stdout, old_stderr = sys.stdout, sys.stderr
        mystdout, mystderr = io.StringIO(), io.StringIO()
        sys.stdout, sys.stderr = mystdout, mystderr

        status_code = 0
        temp_file_path = None
        result_stdout = ""
        result_stderr = ""
        should_exit = False # Flag to control worker exit

        try:
            # Create and write to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8', dir='/tmp') as tmp:
                 temp_file_path = tmp.name
                 tmp.write(code_string)

            # Setup module loading
            module_name = f"triton_kernel_{uuid.uuid4().hex}"
            spec = importlib.util.spec_from_file_location(module_name, temp_file_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not create module spec for {temp_file_path}")
            kernel_module = importlib.util.module_from_spec(spec)
            kernel_module.__dict__.update({
                'torch': torch, 'triton': triton, 'tl': tl, 'math': math,
            })

            # Execute the code
            spec.loader.exec_module(kernel_module)
            # If execution succeeds, status_code remains 0

        except Exception as e:
            status_code = -1
            tb = traceback.format_exc()
            mystderr.write(tb) # Write traceback to captured stderr first

            if is_fatal_error(e):
                # Log as fatal error leading to exit
                print(f"[Worker PID {os.getpid()}] Task {task_id} FATAL error detected:\\n{tb}", file=original_stderr_for_logging, flush=True)
                print(f"[Worker PID {os.getpid()}] Exiting due to fatal error in task {task_id}.", file=original_stderr_for_logging, flush=True)
                should_exit = True # Mark worker for exit
            else:
                # Log as non-fatal error, worker will continue
                print(f"[Worker PID {os.getpid()}] Task {task_id} non-fatal execution error:\\n{tb}", file=original_stderr_for_logging, flush=True)
                print(f"[Worker PID {os.getpid()}] Continuing after non-fatal error in task {task_id}.", file=original_stderr_for_logging, flush=True)
                # should_exit remains False

        finally:
            # --- This block runs ALWAYS after try/except ---

            # 1. Restore stdout/stderr to capture results and prevent pollution
            sys.stdout, sys.stderr = old_stdout, old_stderr
            result_stdout = mystdout.getvalue()
            result_stderr = mystderr.getvalue() # Contains potential traceback from except blocks

            # --- Collect Metrics ---
            gpu_mem_used_gb = None
            cpu_percent = None
            ram_percent = None

            if gpu_handle:
                try:
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
                    gpu_mem_used_gb = mem_info.used / (1024**3) # Convert bytes to GiB
                except pynvml.NVMLError as nvml_err:
                     print(f"[Worker PID {os.getpid()}] [WARN] Failed to get GPU memory info: {nvml_err}", file=original_stderr_for_logging, flush=True)

            try:
                # Get CPU percent over the last interval (non-blocking)
                # First call returns 0.0 or None, subsequent calls give usage since last call.
                # Calling it here measures usage *during* the finally block, which isn't ideal,
                # but it's simple. A better approach might involve measuring before/after exec.
                # Let's just get the current system-wide usage for simplicity.
                psutil.cpu_percent(interval=None) # Initialize if first call
                cpu_percent = psutil.cpu_percent(interval=None)
                ram_percent = psutil.virtual_memory().percent
            except Exception as psutil_err:
                print(f"[Worker PID {os.getpid()}] [WARN] Failed to get CPU/RAM info: {psutil_err}", file=original_stderr_for_logging, flush=True)
            # --- End Collect Metrics ---

            # 2. Send result back to the server (including metrics)
            result = {
                "task_id": task_id,
                "status_code": status_code,
                "stdout": result_stdout,
                "stderr": result_stderr,
                "gpu_mem_used_gb": gpu_mem_used_gb,
                "cpu_percent": cpu_percent,
                "ram_percent": ram_percent,
            }
            try:
                 result_queue.put(result, timeout=5)
                 # Log success only if no error occurred before finally
                 if status_code == 0:
                      print(f"[Worker PID {os.getpid()}] Successfully executed and sent result for task {task_id}.", file=original_stderr_for_logging, flush=True)
                 else:
                      print(f"[Worker PID {os.getpid()}] Successfully sent error result for task {task_id}.", file=original_stderr_for_logging, flush=True)
            except Exception as put_e:
                 # Log failure to send result, regardless of task success/failure
                 print(f"[Worker PID {os.getpid()}] Failed to put result to queue for task {task_id}: {put_e}", file=original_stderr_for_logging, flush=True)

            # 3. Clean up the temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    print(f"[Worker PID {os.getpid()}] Task {task_id} attempting to remove temp file: {temp_file_path}", file=original_stderr_for_logging, flush=True)
                    os.remove(temp_file_path)
                    print(f"[Worker PID {os.getpid()}] Task {task_id} successfully removed temp file: {temp_file_path}", file=original_stderr_for_logging, flush=True)
                except OSError as rm_e:
                    print(f"[Worker PID {os.getpid()}] Error removing temp file {temp_file_path}: {rm_e}", file=original_stderr_for_logging, flush=True)
            elif temp_file_path:
                print(f"[Worker PID {os.getpid()}] Task {task_id} temp file {temp_file_path} did not exist, skipping removal.", file=original_stderr_for_logging, flush=True)
            else:
                print(f"[Worker PID {os.getpid()}] Task {task_id} no temp file path recorded, skipping removal.", file=original_stderr_for_logging, flush=True)

            # 4. Exit worker ONLY if a fatal error occurred (should_exit == True)
            if should_exit:
                break # Exit the while loop

        # If no fatal error occurred (should_exit is False), the loop continues to the next task implicitly

    # End of worker_main function (reached only on poison pill or fatal error)
    print(f"[Worker PID {os.getpid()}] Worker main loop finished. Process exiting.", file=original_stderr_for_logging, flush=True)

    # --- Shutdown Monitoring ---
    if nvml_initialized:
        try:
            pynvml.nvmlShutdown()
            print(f"[Worker PID {os.getpid()}] NVML shut down successfully.", file=original_stderr_for_logging, flush=True)
        except pynvml.NVMLError as nvml_err:
            print(f"[Worker PID {os.getpid()}] [WARN] Failed to shut down NVML: {nvml_err}", file=original_stderr_for_logging, flush=True)
    # --- End Shutdown Monitoring ---