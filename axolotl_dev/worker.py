# worker.py

import os
import sys
import io
import traceback
import torch
import triton
import triton.language as tl
import math

import tempfile # Add tempfile
import importlib.util # Add importlib.util
import uuid # Add uuid for unique module names

def worker_main(task_queue, result_queue, gpu_id):
    """
    Each worker process runs this function:
      - Pins itself to a specific GPU
      - Waits for code from the master process (server)
      - Executes the code via exec()
      - Returns stdout/stderr and status_code
      - Exits immediately if an error occurs during task execution.
    """
    # Pin GPU via environment (so torch sees only 1 device)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # or if you prefer using torch directly: torch.cuda.set_device(0)
    # but that requires re-import after you set env; 
    # here we rely on env plus the single visible GPU

    original_stderr_for_logging = sys.__stderr__ # Capture original stderr early

    print(f"[Worker PID {os.getpid()}] Bound to GPU {gpu_id}. Importing libraries...", file=original_stderr_for_logging, flush=True)

    # We already imported torch/triton at module level; they're in memory now.
    # If your code needs a "warm up," do it here.

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
        temp_file_path = None # Initialize path variable

        try:
            # Create a temporary file in /tmp to store the code
            # Suffix ".py" helps tools recognize it as Python code
            # delete=False needed on Windows, and generally safer with separate cleanup
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8', dir='/tmp') as tmp: # Specify dir='/tmp'
                 temp_file_path = tmp.name
                 tmp.write(code_string)

            # Create a unique module name
            module_name = f"triton_kernel_{uuid.uuid4().hex}"

            # Create a module spec from the temporary file path
            spec = importlib.util.spec_from_file_location(module_name, temp_file_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not create module spec for {temp_file_path}")

            # Create a new module based on the spec
            kernel_module = importlib.util.module_from_spec(spec)

            # Add necessary modules to the new module's globals *before* execution
            # This ensures imports within the executed code can find them if needed,
            # although the code string itself should have imports.
            # Redundant imports are usually harmless.
            kernel_module.__dict__.update({
                'torch': torch,
                'triton': triton,
                'tl': tl,
                'math': math,
                # Add others if consistently needed by test code itself
            })

            # Execute the code within the new module's namespace
            spec.loader.exec_module(kernel_module)

        except Exception as e:
            status_code = -1
            tb = traceback.format_exc()
            # Log the exception traceback to the worker's *original* stderr
            print(f"[Worker PID {os.getpid()}] Task {task_id} execution failed:\n{tb}", file=original_stderr_for_logging, flush=True)
            # Write traceback to the captured stderr as well for the result payload
            mystderr.write(tb)
            # --- CRITICAL CHANGE: Exit worker after error ---
            print(f"[Worker PID {os.getpid()}] Exiting due to error in task {task_id}.", file=original_stderr_for_logging, flush=True)
            # Restore stdout/stderr *before* breaking to ensure logs are captured
            sys.stdout, sys.stderr = old_stdout, old_stderr
            # Send failure result *before* exiting
            result = { "task_id": task_id, "status_code": status_code, "stdout": mystdout.getvalue(), "stderr": mystderr.getvalue() }
            try:
                 result_queue.put(result, timeout=5) # Put result with timeout
                 print(f"[Worker PID {os.getpid()}] Successfully sent error result for task {task_id}.", file=original_stderr_for_logging, flush=True)
            except Exception as put_e:
                 print(f"[Worker PID {os.getpid()}] Failed to put error result to queue: {put_e}", file=original_stderr_for_logging, flush=True)
            # Clean up temp file before exiting
            if temp_file_path and os.path.exists(temp_file_path):
                try: os.remove(temp_file_path)
                except OSError as rm_e: print(f"[Worker PID {os.getpid()}] Error removing temp file {temp_file_path} on error exit: {rm_e}", file=original_stderr_for_logging, flush=True)
            break # Exit the while loop, terminating the worker process
        finally:
            # This finally block now only runs fully on successful execution
            # Restore stdout/stderr (also done before break in except block)
            sys.stdout, sys.stderr = old_stdout, old_stderr
            # Ensure temporary file is deleted on success path ONLY
            # (it's handled separately in the except block for errors)
            if status_code == 0 and temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except OSError as e:
                    print(f"[Worker PID {os.getpid()}] Error removing temp file {temp_file_path} on success: {e}", file=original_stderr_for_logging, flush=True)

        # --- This part is only reached on success ---
        out_str = mystdout.getvalue()
        err_str = mystderr.getvalue()
        result = { "task_id": task_id, "status_code": status_code, "stdout": out_str, "stderr": err_str }
        try:
            result_queue.put(result, timeout=5) # Send success result
        except Exception as put_e:
             print(f"[Worker PID {os.getpid()}] Failed to put success result to queue for task {task_id}: {put_e}", file=original_stderr_for_logging, flush=True)
             # If putting result fails, should we exit? Maybe not on success.

    # End of worker_main function
    print(f"[Worker PID {os.getpid()}] Worker main loop finished. Process exiting.", file=original_stderr_for_logging, flush=True)