# axolotl_dev/worker.py

import os
import sys
import io
import traceback
import torch
import triton
import triton.language as tl
import triton.compiler.errors
import math
import tempfile
import importlib.util
import uuid
import logging

from axolotl_dev.celery_config import celery_app

# Configure logging for the worker
# Note: Celery workers have their own logging setup, this might be supplemental
log = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.__stderr__) # Log to original stderr
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
log.addHandler(handler)
log.setLevel(logging.INFO)

def _is_fatal_error(exception) -> bool:
    """Checks if an exception should be considered fatal for the worker environment."""
    if isinstance(exception, triton.compiler.errors.CompilationError):
        log.warning("Fatal error type: Triton CompilationError")
        return True
    tb_str = traceback.format_exc()
    if "CUDA" in tb_str:
        log.warning("Fatal error type: CUDA error detected in traceback")
        return True
    # Add any other specific error types that indicate a poisoned worker environment
    return False

@celery_app.task(bind=True, throws=(
    # List exceptions here that should NOT trigger a retry and are handled
    # Note: If _is_fatal_error is True, Celery won't retry anyway by default if task_acks_late=True
    Exception,
))
def execute_triton_code(self, code: str, tests: str):
    """Celery task to execute Triton code with tests."""
    task_id = self.request.id
    log.info(f"Task {task_id}: Received execution request.")

    # Check which GPU this worker is assigned to (set during worker launch)
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    log.info(f"Task {task_id}: Worker assigned to GPU(s): {cuda_visible_devices if cuda_visible_devices else 'Not Set (Default)'}")

    # Combine code and tests
    code_string = (
        "import torch\n"
        "import triton\n"
        "import triton.language as tl\n"
        "import math\n\n" # Added math
        f"{code}\n\n"
        "# ---- Tests Below ----\n"
        "DEVICE = torch.device('cuda')\n" # Assumes code needs 'cuda' device
        f"{tests}\n"
    )

    # Prepare to capture stdout/stderr
    old_stdout, old_stderr = sys.stdout, sys.stderr
    mystdout, mystderr = io.StringIO(), io.StringIO()
    sys.stdout, sys.stderr = mystdout, mystderr

    status_code = 0
    temp_file_path = None
    result = {}

    try:
        log.debug(f"Task {task_id}: Creating temporary file.")
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8', dir='/tmp') as tmp:
             temp_file_path = tmp.name
             tmp.write(code_string)
        log.debug(f"Task {task_id}: Wrote code to {temp_file_path}")

        module_name = f"triton_kernel_{uuid.uuid4().hex}"
        spec = importlib.util.spec_from_file_location(module_name, temp_file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not create module spec for {temp_file_path}")

        kernel_module = importlib.util.module_from_spec(spec)
        kernel_module.__dict__.update({
            'torch': torch, 'triton': triton, 'tl': tl, 'math': math,
        })

        log.info(f"Task {task_id}: Executing code...")
        spec.loader.exec_module(kernel_module)
        log.info(f"Task {task_id}: Execution successful.")
        status_code = 0 # Explicitly set success

    except Exception as e:
        status_code = -1
        tb = traceback.format_exc()
        mystderr.write(tb) # Capture traceback

        if _is_fatal_error(e):
            log.error(f"Task {task_id}: FATAL error during execution. Worker may need restart.\n{tb}")
            # Potentially raise a specific exception or let Celery handle based on acks_late
            # For now, just log and return the error result.
            # If using acks_late, the worker might not acknowledge and Celery might retry
            # depending on configuration. If the error persists, the task will eventually fail.
        else:
            log.warning(f"Task {task_id}: Non-fatal error during execution.\n{tb}")
        # We always return a result, even on error

    finally:
        # Restore stdout/stderr
        sys.stdout, sys.stderr = old_stdout, old_stderr

        # Get captured output
        out_str = mystdout.getvalue()
        err_str = mystderr.getvalue()

        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                log.debug(f"Task {task_id}: Removed temp file {temp_file_path}")
            except OSError as rm_e:
                log.error(f"Task {task_id}: Error removing temp file {temp_file_path}: {rm_e}")

        # Prepare result dictionary
        result = {
            "status_code": status_code,
            "stdout": out_str,
            "stderr": err_str,
        }
        log.info(f"Task {task_id}: Sending result: status_code={status_code}")

    return result # Return value is stored in the Celery backend