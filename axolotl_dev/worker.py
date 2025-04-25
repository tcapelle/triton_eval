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
      - Returns stdout/stderr and status_code
      - Exits only if a CUDA/Triton compilation error occurs, otherwise reports errors and continues.
    """
    # Pin GPU via environment (so torch sees only 1 device)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    original_stderr_for_logging = sys.__stderr__ # Capture original stderr early

    print(f"[Worker PID {os.getpid()}] Bound to GPU {gpu_id}. Importing libraries...", file=original_stderr_for_logging, flush=True)

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

            # 2. Send result back to the server
            result = {
                "task_id": task_id,
                "status_code": status_code,
                "stdout": result_stdout,
                "stderr": result_stderr
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
                    os.remove(temp_file_path)
                except OSError as rm_e:
                    print(f"[Worker PID {os.getpid()}] Error removing temp file {temp_file_path}: {rm_e}", file=original_stderr_for_logging, flush=True)

            # 4. Exit worker ONLY if a fatal error occurred (should_exit == True)
            if should_exit:
                break # Exit the while loop

        # If no fatal error occurred (should_exit is False), the loop continues to the next task implicitly

    # End of worker_main function (reached only on poison pill or fatal error)
    print(f"[Worker PID {os.getpid()}] Worker main loop finished. Process exiting.", file=original_stderr_for_logging, flush=True)