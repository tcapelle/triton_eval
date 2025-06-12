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
import gc

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

def _run_benchmark(temp_file_path, task_type, benchmark_runs):
    """Run benchmarking by re-executing the entire module multiple times."""
    times = []
    memory_peaks = []
    
    # Warmup runs (3 runs) - re-execute entire module
    for _ in range(3):
        try:
            # Create fresh module for warmup
            module_name = f"{task_type}_warmup_{uuid.uuid4().hex}"
            spec = importlib.util.spec_from_file_location(module_name, temp_file_path)
            if spec is None or spec.loader is None:
                continue
            warmup_module = importlib.util.module_from_spec(spec)
            
            # Add appropriate imports
            if task_type == "triton":
                warmup_module.__dict__.update({
                    'torch': torch, 'triton': triton, 'tl': tl, 'math': math,
                })
            elif task_type == "pytorch":
                import torch.nn as nn
                import torch.nn.functional as F
                warmup_module.__dict__.update({
                    'torch': torch, 'nn': nn, 'F': F, 'math': math, 'time': time, 'gc': gc,
                })
            
            # Capture output to avoid pollution
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
            
            spec.loader.exec_module(warmup_module)
            torch.cuda.synchronize()
            
            # Restore output
            sys.stdout, sys.stderr = old_stdout, old_stderr
        except:
            pass  # Ignore warmup failures
    
    # Actual benchmark runs - re-execute entire module
    for run_idx in range(benchmark_runs):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        try:
            # Create fresh module for each benchmark run
            module_name = f"{task_type}_bench_{run_idx}_{uuid.uuid4().hex}"
            spec = importlib.util.spec_from_file_location(module_name, temp_file_path)
            if spec is None or spec.loader is None:
                continue
            bench_module = importlib.util.module_from_spec(spec)
            
            # Add appropriate imports
            if task_type == "triton":
                bench_module.__dict__.update({
                    'torch': torch, 'triton': triton, 'tl': tl, 'math': math,
                })
            elif task_type == "pytorch":
                import torch.nn as nn
                import torch.nn.functional as F
                bench_module.__dict__.update({
                    'torch': torch, 'nn': nn, 'F': F, 'math': math, 'time': time, 'gc': gc,
                })
            
            # Capture output to avoid pollution
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
            
            start_time = time.perf_counter()
            spec.loader.exec_module(bench_module)
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            # Restore output
            sys.stdout, sys.stderr = old_stdout, old_stderr
            
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

def _run_pytorch_benchmark(temp_file_path, task_type, benchmark_runs, torch_compile=False, torch_compile_mode="default"):
    """Run PyTorch-specific benchmarking by re-executing the entire module multiple times."""
    
    results = {}
    
    # 1. Regular PyTorch benchmarking - re-execute entire module
    regular_times = []
    memory_peaks = []
    
    # Warmup runs (3 runs) - re-execute entire module
    for _ in range(3):
        try:
            # Create fresh module for warmup
            module_name = f"{task_type}_warmup_{uuid.uuid4().hex}"
            spec = importlib.util.spec_from_file_location(module_name, temp_file_path)
            if spec is None or spec.loader is None:
                continue
            warmup_module = importlib.util.module_from_spec(spec)
            
            # Add appropriate imports
            import torch.nn as nn
            import torch.nn.functional as F
            warmup_module.__dict__.update({
                'torch': torch, 'nn': nn, 'F': F, 'math': math, 'time': time, 'gc': gc,
            })
            
            # Capture output to avoid pollution
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
            
            spec.loader.exec_module(warmup_module)
            torch.cuda.synchronize()
            
            # Restore output
            sys.stdout, sys.stderr = old_stdout, old_stderr
        except:
            pass  # Ignore warmup failures
    
    # Benchmark regular version - re-execute entire module
    for run_idx in range(benchmark_runs):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        try:
            # Create fresh module for each benchmark run
            module_name = f"{task_type}_bench_{run_idx}_{uuid.uuid4().hex}"
            spec = importlib.util.spec_from_file_location(module_name, temp_file_path)
            if spec is None or spec.loader is None:
                continue
            bench_module = importlib.util.module_from_spec(spec)
            
            # Add appropriate imports
            import torch.nn as nn
            import torch.nn.functional as F
            bench_module.__dict__.update({
                'torch': torch, 'nn': nn, 'F': F, 'math': math, 'time': time, 'gc': gc,
            })
            
            # Capture output to avoid pollution
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
            
            start_time = time.perf_counter()
            spec.loader.exec_module(bench_module)
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            # Restore output
            sys.stdout, sys.stderr = old_stdout, old_stderr
            
            run_time = (end_time - start_time) * 1000  # Convert to ms
            regular_times.append(run_time)
            
            peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
            memory_peaks.append(peak_memory)
            
        except Exception as e:
            continue
    
    if len(regular_times) == 0:
        raise Exception("No successful regular benchmark runs")
    
    regular_mean_time = sum(regular_times) / len(regular_times)
    regular_std_time = (sum((t - regular_mean_time) ** 2 for t in regular_times) / len(regular_times)) ** 0.5 if len(regular_times) > 1 else 0
    memory_peak = max(memory_peaks) if memory_peaks else 0
    
    results.update({
        "mean_time_ms": regular_mean_time,
        "std_time_ms": regular_std_time,
        "memory_peak_mb": memory_peak,
        "successful_runs": len(regular_times)
    })
    
    # 2. torch.compile benchmarking (if requested)
    if torch_compile:
        try:
            compiled_times = []
            
            # First, create a baseline module and identify functions that can be compiled
            baseline_module_name = f"{task_type}_baseline_{uuid.uuid4().hex}"
            baseline_spec = importlib.util.spec_from_file_location(baseline_module_name, temp_file_path)
            if baseline_spec is None or baseline_spec.loader is None:
                raise Exception("Could not create baseline module spec for torch.compile")
            
            baseline_module = importlib.util.module_from_spec(baseline_spec)
            import torch.nn as nn
            import torch.nn.functional as F
            baseline_module.__dict__.update({
                'torch': torch, 'nn': nn, 'F': F, 'math': math, 'time': time, 'gc': gc,
            })
            
            # Capture output during baseline execution
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
            baseline_spec.loader.exec_module(baseline_module)
            sys.stdout, sys.stderr = old_stdout, old_stderr
            
            # Find functions that can be compiled
            compiled_functions = {}
            for attr_name in dir(baseline_module):
                if not attr_name.startswith('_'):
                    attr = getattr(baseline_module, attr_name)
                    if callable(attr) and not isinstance(attr, type) and hasattr(attr, '__code__'):
                        try:
                            compiled_functions[attr_name] = torch.compile(attr, mode=torch_compile_mode)
                        except:
                            pass  # Skip functions that can't be compiled
            
            if not compiled_functions:
                # No functions could be compiled, skip torch.compile benchmarking
                results.update({
                    "torch_compile_benchmark_mean_time_ms": None,
                    "torch_compile_benchmark_std_time_ms": None,
                    "torch_compile_speedup": None
                })
            else:
                # Warmup the compiled functions to trigger compilation
                print(f"[Worker PID {os.getpid()}] Running torch.compile warmup for {len(compiled_functions)} function(s)...", file=original_stderr_for_logging, flush=True)
                
                # Run multiple warmup passes with compiled functions
                for warmup_idx in range(5):  # More warmup runs for torch.compile
                    try:
                        warmup_module_name = f"{task_type}_warmup_compiled_{warmup_idx}_{uuid.uuid4().hex}"
                        warmup_spec = importlib.util.spec_from_file_location(warmup_module_name, temp_file_path)
                        if warmup_spec is None or warmup_spec.loader is None:
                            continue
                            
                        warmup_module = importlib.util.module_from_spec(warmup_spec)
                        warmup_module.__dict__.update({
                            'torch': torch, 'nn': nn, 'F': F, 'math': math, 'time': time, 'gc': gc,
                        })
                        
                        # Replace functions with compiled versions BEFORE executing
                        for func_name, compiled_func in compiled_functions.items():
                            warmup_module.__dict__[func_name] = compiled_func
                        
                        # Capture output during warmup
                        old_stdout, old_stderr = sys.stdout, sys.stderr
                        sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
                        
                        warmup_spec.loader.exec_module(warmup_module)
                        torch.cuda.synchronize()
                        
                        sys.stdout, sys.stderr = old_stdout, old_stderr
                    except:
                        sys.stdout, sys.stderr = old_stdout, old_stderr
                        pass  # Ignore warmup failures
                
                print(f"[Worker PID {os.getpid()}] torch.compile warmup completed, starting benchmarks...", file=original_stderr_for_logging, flush=True)
                
                # Now run actual benchmark runs with properly warmed-up compiled functions
                for run_idx in range(benchmark_runs):
                    torch.cuda.empty_cache()
                    
                    try:
                        # Create fresh module for each compiled benchmark run
                        module_name = f"{task_type}_compiled_{run_idx}_{uuid.uuid4().hex}"
                        spec = importlib.util.spec_from_file_location(module_name, temp_file_path)
                        if spec is None or spec.loader is None:
                            continue
                            
                        compiled_module = importlib.util.module_from_spec(spec)
                        compiled_module.__dict__.update({
                            'torch': torch, 'nn': nn, 'F': F, 'math': math, 'time': time, 'gc': gc,
                        })
                        
                        # Replace functions with already-warmed compiled versions BEFORE executing
                        for func_name, compiled_func in compiled_functions.items():
                            compiled_module.__dict__[func_name] = compiled_func
                        
                        # Capture output to avoid pollution
                        old_stdout, old_stderr = sys.stdout, sys.stderr
                        sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
                        
                        start_time = time.perf_counter()
                        spec.loader.exec_module(compiled_module)
                        torch.cuda.synchronize()
                        end_time = time.perf_counter()
                        
                        # Restore output
                        sys.stdout, sys.stderr = old_stdout, old_stderr
                        
                        run_time = (end_time - start_time) * 1000  # Convert to ms
                        compiled_times.append(run_time)
                        
                    except Exception as e:
                        # Restore output even on error
                        sys.stdout, sys.stderr = old_stdout, old_stderr
                        continue
            
            if len(compiled_times) > 0:
                compiled_mean_time = sum(compiled_times) / len(compiled_times)
                compiled_std_time = (sum((t - compiled_mean_time) ** 2 for t in compiled_times) / len(compiled_times)) ** 0.5 if len(compiled_times) > 1 else 0
                speedup = regular_mean_time / compiled_mean_time if compiled_mean_time > 0 else 0
                
                results.update({
                    "torch_compile_benchmark_mean_time_ms": compiled_mean_time,
                    "torch_compile_benchmark_std_time_ms": compiled_std_time,
                    "torch_compile_speedup": speedup
                })
            else:
                results.update({
                    "torch_compile_benchmark_mean_time_ms": None,
                    "torch_compile_benchmark_std_time_ms": None,
                    "torch_compile_speedup": None
                })
                
        except Exception as e:
            print(f"torch.compile benchmarking failed: {e}", file=original_stderr_for_logging, flush=True)
            results.update({
                "torch_compile_benchmark_mean_time_ms": None,
                "torch_compile_benchmark_std_time_ms": None,
                "torch_compile_speedup": None
            })
    
    return results

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
            task_type = "triton"  # Default to triton for backward compatibility
            benchmark = False
            benchmark_runs = 10
            torch_compile = False
            torch_compile_mode = "default"
        else:
            task_id = task_data["task_id"]
            task_type = task_data.get("task_type", "triton")  # Default to triton
            code_string = task_data["code"]
            benchmark = task_data.get("benchmark", False)
            benchmark_runs = task_data.get("benchmark_runs", 10)
            torch_compile = task_data.get("torch_compile", False)
            torch_compile_mode = task_data.get("torch_compile_mode", "default")

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
        
        # PyTorch-specific benchmark metrics
        torch_compile_benchmark_mean_time_ms = None
        torch_compile_benchmark_std_time_ms = None
        torch_compile_speedup = None

        try:
            # Create and write to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8', dir='/tmp') as tmp:
                 temp_file_path = tmp.name
                 tmp.write(code_string)

            # Setup module loading
            module_name = f"{task_type}_kernel_{uuid.uuid4().hex}"
            spec = importlib.util.spec_from_file_location(module_name, temp_file_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not create module spec for {temp_file_path}")
            kernel_module = importlib.util.module_from_spec(spec)
            
            # Add appropriate imports based on task type
            if task_type == "triton":
                kernel_module.__dict__.update({
                    'torch': torch, 'triton': triton, 'tl': tl, 'math': math,
                })
            elif task_type == "pytorch":
                import torch.nn as nn
                import torch.nn.functional as F
                kernel_module.__dict__.update({
                    'torch': torch, 'nn': nn, 'F': F, 'math': math, 'time': time, 'gc': gc,
                })

            # Execute the code
            spec.loader.exec_module(kernel_module)
            # If execution succeeds, status_code remains 0
            
            # Run benchmarking if enabled and execution was successful
            if benchmark and status_code == 0:
                try:
                    if task_type == "triton":
                        # Use Triton benchmarking with entire module execution
                        benchmark_results = _run_benchmark(temp_file_path, task_type, benchmark_runs)
                        benchmark_mean_time_ms = benchmark_results["mean_time_ms"]
                        benchmark_std_time_ms = benchmark_results["std_time_ms"] 
                        benchmark_memory_peak_mb = benchmark_results["memory_peak_mb"]
                        benchmark_successful_runs = benchmark_results["successful_runs"]
                        print(f"[Worker PID {os.getpid()}] Triton task {task_id} benchmarking completed: {benchmark_successful_runs}/{benchmark_runs} runs, avg {benchmark_mean_time_ms:.2f}ms", file=original_stderr_for_logging, flush=True)
                    elif task_type == "pytorch":
                        # Use PyTorch-specific benchmarking with entire module execution and optional torch.compile
                        benchmark_results = _run_pytorch_benchmark(temp_file_path, task_type, benchmark_runs, torch_compile, torch_compile_mode)
                        benchmark_mean_time_ms = benchmark_results["mean_time_ms"]
                        benchmark_std_time_ms = benchmark_results["std_time_ms"] 
                        benchmark_memory_peak_mb = benchmark_results["memory_peak_mb"]
                        benchmark_successful_runs = benchmark_results["successful_runs"]
                        
                        # Extract torch.compile results if available
                        torch_compile_benchmark_mean_time_ms = benchmark_results.get("torch_compile_benchmark_mean_time_ms")
                        torch_compile_benchmark_std_time_ms = benchmark_results.get("torch_compile_benchmark_std_time_ms")
                        torch_compile_speedup = benchmark_results.get("torch_compile_speedup")
                        
                        compile_info = ""
                        if torch_compile and torch_compile_speedup is not None:
                            compile_info = f", torch.compile speedup: {torch_compile_speedup:.2f}x"
                        print(f"[Worker PID {os.getpid()}] PyTorch task {task_id} benchmarking completed: {benchmark_successful_runs}/{benchmark_runs} runs, avg {benchmark_mean_time_ms:.2f}ms{compile_info}", file=original_stderr_for_logging, flush=True)
                        
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
                # PyTorch-specific results
                "torch_compile_benchmark_mean_time_ms": torch_compile_benchmark_mean_time_ms,
                "torch_compile_benchmark_std_time_ms": torch_compile_benchmark_std_time_ms,
                "torch_compile_speedup": torch_compile_speedup,
            }
            try:
                 result_queue.put(result, timeout=5)
                 # Log success only if no error occurred before finally
                 if status_code == 0:
                      print(f"[Worker PID {os.getpid()}] Successfully executed and sent result for {task_type} task {task_id}.", file=original_stderr_for_logging, flush=True)
                 else:
                      print(f"[Worker PID {os.getpid()}] Successfully sent error result for {task_type} task {task_id}.", file=original_stderr_for_logging, flush=True)
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