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
import time

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

def _run_benchmark(kernel_module, benchmark_runs):
    """Run benchmarking on already executed and compiled kernel module."""
    times = []
    memory_peaks = []
    
    # Look for a function that might be the main entry point for benchmarking
    # We'll try to find functions that might be test/benchmark functions
    benchmark_function = None
    
    # Common patterns for benchmark functions
    possible_names = ['benchmark_function', 'main', 'test', 'run_test', 'run_benchmark']
    for name in possible_names:
        if hasattr(kernel_module, name):
            benchmark_function = getattr(kernel_module, name)
            break
    
    # If no explicit benchmark function, try to find any callable that's not a built-in
    if benchmark_function is None:
        for attr_name in dir(kernel_module):
            if not attr_name.startswith('_'):
                attr = getattr(kernel_module, attr_name)
                if callable(attr) and not isinstance(attr, type):
                    benchmark_function = attr
                    break
    
    if benchmark_function is None:
        raise Exception("No benchmarkable function found in the module")
    
    # Warmup runs (3 runs)
    for _ in range(3):
        try:
            benchmark_function()
            torch.cuda.synchronize()
        except:
            pass  # Ignore warmup failures
    
    # Actual benchmark runs
    for run_idx in range(benchmark_runs):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        start_time = time.perf_counter()
        try:
            benchmark_function()
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            run_time = (end_time - start_time) * 1000  # Convert to ms
            times.append(run_time)
            
            peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
            memory_peaks.append(peak_memory)
            
        except Exception as e:
            # Skip failed runs
            continue
    
    successful_runs = len(times)
    if successful_runs == 0:
        raise Exception("No successful benchmark runs")
    
    mean_time = sum(times) / len(times)
    std_time = (sum((t - mean_time) ** 2 for t in times) / len(times)) ** 0.5 if len(times) > 1 else 0
    memory_peak = max(memory_peaks) if memory_peaks else 0
    
    return {
        "mean_time_ms": mean_time,
        "std_time_ms": std_time,
        "memory_peak_mb": memory_peak,
        "successful_runs": successful_runs
    }

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
        task_data = task_queue.get()
        if task_data is None:
            print(f"[Worker PID {os.getpid()}] Received poison pill. Exiting.", file=original_stderr_for_logging, flush=True)
            break # Exit loop cleanly
        
        # Handle both old format (tuple) and new format (dict) for backward compatibility
        if isinstance(task_data, tuple):
            task_id, code_string = task_data
            benchmark = False
            benchmark_runs = 10
        else:
            task_id = task_data["task_id"]
            code_string = task_data["code"]
            benchmark = task_data.get("benchmark", False)
            benchmark_runs = task_data.get("benchmark_runs", 10)

        # Prepare to capture stdout/stderr
        old_stdout, old_stderr = sys.stdout, sys.stderr
        mystdout, mystderr = io.StringIO(), io.StringIO()
        sys.stdout, sys.stderr = mystdout, mystderr

        status_code = 0
        temp_file_path = None
        result_stdout = ""
        result_stderr = ""
        should_exit = False # Flag to control worker exit
        
        # Benchmark metrics (initialized)
        benchmark_mean_time_ms = None
        benchmark_std_time_ms = None
        benchmark_memory_peak_mb = None
        benchmark_successful_runs = None

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
            
            # If benchmarking is enabled and execution was successful, run benchmarks
            if benchmark and status_code == 0:
                try:
                    benchmark_results = _run_benchmark(kernel_module, benchmark_runs)
                    benchmark_mean_time_ms = benchmark_results["mean_time_ms"]
                    benchmark_std_time_ms = benchmark_results["std_time_ms"] 
                    benchmark_memory_peak_mb = benchmark_results["memory_peak_mb"]
                    benchmark_successful_runs = benchmark_results["successful_runs"]
                    print(f"[Worker PID {os.getpid()}] Task {task_id} benchmarking completed: {benchmark_successful_runs}/{benchmark_runs} runs, avg {benchmark_mean_time_ms:.2f}ms", file=original_stderr_for_logging, flush=True)
                except Exception as bench_e:
                    print(f"[Worker PID {os.getpid()}] Task {task_id} benchmarking failed: {bench_e}", file=original_stderr_for_logging, flush=True)
                    # Keep benchmark metrics as None if benchmarking fails

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
                "benchmark_mean_time_ms": benchmark_mean_time_ms,
                "benchmark_std_time_ms": benchmark_std_time_ms,
                "benchmark_memory_peak_mb": benchmark_memory_peak_mb,
                "benchmark_successful_runs": benchmark_successful_runs,
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