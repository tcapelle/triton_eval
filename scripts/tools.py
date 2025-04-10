import os
from pathlib import Path
import subprocess
import uuid
from typing import Dict, Union
TEMP_FILES_DIR = Path("./temp_files")
TEMP_FILES_DIR.mkdir(exist_ok=True)

def run_python_file(file_path: str) -> Dict[str, Union[int, str]]:
    """
    Executes a Python script at the given file path and captures its output.

    Args:
        file_path: The path to the Python file to execute.

    Returns:
        A dictionary containing:
        - 'status_code': The exit code of the script (0 for success).
        - 'output': The standard output (stdout) if successful,
                    or standard error (stderr) if an error occurred.
    """
    result = subprocess.run(
        ["python", file_path],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"Error running {file_path}:")
        return {"status_code": result.returncode, "output": result.stderr}
    return {"status_code": 0, "output": result.stdout}

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
    with open(file_path, "r") as f:
        return f.read()
    
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

def clear_temp_files():
    """
    Deletes all files currently present in the designated temporary files directory.
    """
    for file_path in TEMP_FILES_DIR.glob("*"):
        os.remove(file_path)


def run_python_code(code: str) -> Dict[str, Union[int, str]]:
    """
    Saves the provided Python code to a temporary file, executes it,
    and returns the result including status code and output.

    Args:
        code: The Python code string to execute.

    Returns:
        A dictionary containing:
        - 'status_code': The exit code of the script (0 for success).
        - 'output': The standard output (stdout) if successful,
                    or standard error (stderr) if an error occurred.
    """
    file_path = save_to_temp_file(code)
    # The run_python_file function now returns the dictionary directly
    return run_python_file(file_path)