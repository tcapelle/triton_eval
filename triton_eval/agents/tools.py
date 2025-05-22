import os
import re
import shutil
from pathlib import Path
import subprocess
import uuid
from typing import Union
import weave
import io
import contextlib
import traceback
import importlib.util
import tempfile
import sys
import textwrap

TEMP_FILES_DIR = Path("./temp_files")
TEMP_FILES_DIR.mkdir(exist_ok=True)

@weave.op
def extract_code(text: str) -> str:
    """
    1) Extract the 'scope' from <triton> tags (closed or unclosed at start)
    2) In that scope, find ALL ```...``` blocks (with optional language)
    3) Return the content of the last block, dedented and stripped
    4) If no blocks and we had a triton-scope, return that scope stripped
    5) Else return empty string
    """
    # 1) pick your search scope and possible fallback
    closed = re.findall(r"<triton>(.*?)</triton>", text, re.DOTALL)
    if closed:
        scope = closed[-1]
        fallback = scope
    elif text.startswith("<triton>") and "</triton>" not in text:
        scope = text[len("<triton>"):]
        fallback = scope
    else:
        scope = text
        fallback = None

    # 2) unified regex for fenced blocks (captures an optional language)
    pattern = re.compile(
        r"```(?:([^\n`]+)[ \t]*\n)?(.*?)(?:\n)?```",
        re.DOTALL
    )
    matches = list(pattern.finditer(scope))

    # 3) if we found any fenced blocks, take the last one
    if matches:
        body = matches[-1].group(2)
        return textwrap.dedent(body).strip()

    # 4) no blocks → if this was a triton-scope, return it
    if fallback is not None:
        return fallback.strip()

    # 5) nothing matched
    return ""

@weave.op
def remove_tests(code: str) -> str:
    "Only works on standard test files"
    code_without_tests = code.split("#"*146)[0]
    return code_without_tests

@weave.op
def extract_tests(code: str) -> str:
    "Only works on standard test files"
    tests = code.split("#"*146)[-1]
    return tests

@weave.op
def run_python_file(file_path: str, env: dict[str, str] = None, timeout: int = 60) -> dict[str, Union[int, str]]:
    """
    Executes a Python script at the given file path and captures its output.

    Args:
        file_path: The path to the Python file to execute.
        env: Optional dictionary of environment variables to add/override.

    Returns:
        A dictionary containing:
        - 'status_code': The exit code of the script (0 for success).
        - 'output': The standard output (stdout) if successful,
                    or standard error (stderr) if an error occurred.
    """
    # Create a copy of the current environment and update it with provided env vars
    current_env = os.environ.copy()
    if env:
        current_env.update(env)
    result = subprocess.run(
        ["python", file_path],
        capture_output=True,
        text=True,
        env=current_env,
        timeout=timeout
    )
    if result.returncode != 0:
        return {"status_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    return {"status_code": 0, "stdout": result.stdout, "stderr": result.stderr}

@weave.op
def save_to_file(file_path: str, content: str):
    """
    Writes the given content to a file at the specified path.
    If the file exists, it will be overwritten.

    Args:
        file_path: The path of the file to write to.
        content: The string content to write to the file.
    """
    with open(file_path, "w") as f:
        f.write(content)

def read_file(file_path: str) -> str:
    """
    Reads the entire content of a file at the specified path.

    Args:
        file_path: The path of the file to read.

    Returns:
        The content of the file as a string.
    """
    try:
        with open(file_path, "r") as f:
            return f.read()
    except FileNotFoundError as e:
        return f"FileNotFoundError: {file_path}"

@weave.op
def save_to_temp_file(content: str) -> str:
    """
    Saves the given content to a temporary Python file with a unique name.

    Args:
        content: The string content to write to the temporary file.

    Returns:
        The path to the newly created temporary file.
    """
    file_path = TEMP_FILES_DIR / f"{uuid.uuid4()}.py"
    save_to_file(file_path, content)
    return str(file_path) # Ensure the path is returned as a string

@weave.op
def clear_temp_files():
    """
    Deletes all files currently present in the designated temporary files directory.
    """
    if TEMP_FILES_DIR.exists():
        shutil.rmtree(TEMP_FILES_DIR)
        TEMP_FILES_DIR.mkdir(exist_ok=True)


@weave.op
def run_python_code(code: str, env: dict[str, str] = None) -> dict[str, Union[int, str]]:
    """Executes a snippet of Python code.

    Args:
        code: The Python code string to execute.
        env: Optional dictionary of environment variables.
    """
    file_path = save_to_temp_file(code)
    # The run_python_file function now returns the dictionary directly
    return run_python_file(file_path, env)

@weave.op
def run_python_in_process(code: str):
    """
    Executes the code in the current process using a separate module namespace
    and captures its output.

    Args:
        code: The Python code string to execute.

    Returns:
        A dictionary containing:
        - 'status_code': 0 for success, 1 for error.
        - 'stdout': The captured standard output.
        - 'stderr': The captured standard error.
        - 'error': The error message if an exception occurred (optional).
        - 'traceback': The traceback string if an exception occurred (optional).
    """
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    status_code = 0
    error_message = None
    tb_string = None

    temp_file_path = None # Initialize to None
    try:
        # Use the existing function to save the code to a temporary file
        temp_file_path = save_to_temp_file(code)

        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            # Create a module specification pointing to our temp file
            # Use a unique module name based on UUID to avoid conflicts
            module_name = f"temp_module_{uuid.uuid4().hex}"
            spec = importlib.util.spec_from_file_location(module_name, temp_file_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not create module spec for {temp_file_path}")

            # Create a new module based on that spec
            temp_module = importlib.util.module_from_spec(spec)
            # Add necessary builtins or other modules if needed
            # temp_module.__dict__.update({'__builtins__': __builtins__})

            # Execute the code in the module's namespace
            spec.loader.exec_module(temp_module)

    except Exception as e:
        status_code = 1
        error_message = str(e)
        tb_string = traceback.format_exc()
    finally:
        # Ensure the temporary file is deleted if it was created
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except OSError as e:
                pass

    # Capture output after execution
    stdout_val = stdout_capture.getvalue()
    stderr_val = stderr_capture.getvalue()

    result = {
        "status_code": status_code,
        "stdout": stdout_val,
        "stderr": stderr_val,
    }
    if error_message:
        result["error"] = error_message
        result["traceback"] = tb_string

    return result

@weave.op
def think(thought: str) -> str:
    """Use the tool to think about something. It will not obtain new information or change the database, but just append the thought to the log. Use it when complex reasoning or some cache memory is needed.

    Args:
        thought: A thought to think about.
    """
    return thought