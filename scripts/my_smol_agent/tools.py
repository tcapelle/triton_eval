import os
import re
import shutil
from pathlib import Path
import subprocess
import uuid
from typing import Union
import weave

TEMP_FILES_DIR = Path("./temp_files")
TEMP_FILES_DIR.mkdir(exist_ok=True)

@weave.op
def extract_code(code: str) -> str:
    "Extract the last code block surrounded by ```python, use re"
    pattern = r"```python(.*?)```"
    match = re.search(pattern, code, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
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

    return {"status_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}

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
def think(thought: str) -> str:
    """Use the tool to think about something. It will not obtain new information or change the database, but just append the thought to the log. Use it when complex reasoning or some cache memory is needed.

    Args:
        thought: A thought to think about.
    """
    return thought

DEFAULT_TOOLS = [
    run_python_code,
    save_to_file,
    read_file,
    think,
]