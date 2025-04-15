"Some utils"
import torch
import os
import uuid
import subprocess
from pathlib import Path
from pydantic import BaseModel
from concurrent.futures import ProcessPoolExecutor, as_completed

TEMP_FILES_DIR = Path("./temp_files")
TEMP_FILES_DIR.mkdir(exist_ok=True)
GPUS = list(range(torch.cuda.device_count()))


def to_device(x, device: torch.device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    elif isinstance(x, list):
        return [to_device(v, device) for v in x]
    return x


def set_gpu_arch(gpu: str="h100"):
    """
    Set env variable for torch cuda arch list to build kernels for specified architectures
    """

    gpu_arch_mapping = {
        "l40s": ["Ada"],
        "h100": ["Hopper"],
        "a100": ["Ampere"],
        "l4": ["Ada"],
        "t4": ["Turing"],
        "a10g": ["Ampere"],
    }
    arch_list = gpu_arch_mapping[gpu.lower()]
    valid_archs = ["Maxwell", "Pascal", "Volta", "Turing", "Ampere", "Hopper", "Ada"]
    for arch in arch_list:
        if arch not in valid_archs:
            raise ValueError(
                f"Invalid architecture: {arch}. Must be one of {valid_archs}"
            )

    os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(arch_list)


def read_file(file_path: str) -> str:
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist")
        return ""

    try:
        with open(file_path, "r") as file:
            return file.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return ""
    
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
    return file_path

def save_to_temp_file(content: str) -> str:
    """
    Saves the given content to a temporary Python file with a unique name.

    Args:
        content: The string content to write to the temporary file.

    Returns:
        The path to the newly created temporary file.
    """
    TEMP_FILES_DIR.mkdir(exist_ok=True)
    file_path = TEMP_FILES_DIR / f"{uuid.uuid4()}.py"
    save_to_file(file_path, content)
    return str(file_path) # Ensure the path is returned as a string

def get_tests(script_content: str) -> str:
    """Get test functions from script content."""
    if "def test_" not in script_content:
        return ""
    
    # Extract the part that contains test function
    test_part = script_content.split("def test_", 1)[1]
    # Reconstruct the function definition
    return "def test_" + test_part


class RunResult(BaseModel):
    success: bool
    results: subprocess.CompletedProcess | None
    file_name: str

def ifnone(x, default):
    return x if x is not None else default

def run_script_on_gpu(script_content: str, test_content: str | None=None, file_name: str | None=None, gpu_id: int|None=None) -> RunResult:
    """
    Runs a given Python script on a specified GPU.

    Args:
        script_content: The content of the script to run.
        test_content: The content of the test to run.
        file_name: The name of the file to save the script to.
        gpu_id: The ID of the GPU to run the script on.

    Returns:
        success: Whether the script ran successfully.
        results: The results of the script.
        file_name: The name of the file that was run.
    """
    test_content = ifnone(test_content, "")
    content = script_content + "\n" + "#" * 146 + "\n" + test_content

    try:
        if file_name is None:
            file_path = save_to_temp_file(content)
        else:
            file_path = save_to_file(file_name, content)

        # Set GPU device for execution
        env = os.environ.copy()
        if gpu_id is not None:
            if "CUDA_VISIBLE_DEVICES" in env:
                del env["CUDA_VISIBLE_DEVICES"]
            if gpu_id is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        # Run the temporary Python file
        results = subprocess.run(
            ["python", file_path], 
            capture_output=True, 
            text=True,
            env=env
        )

        success = results.returncode == 0  # Determine if execution was successful

        return RunResult(success=success, results=results, file_name=file_name)  # Return execution success status

    except Exception as e:
        return RunResult(success=False, results=None, file_name=file_name)

def run_code_parallel(pred, test, files, gpus=GPUS):
    """
    Runs code in parallel across multiple GPUs, ensuring each GPU runs one script at a time.
    """
    total_scripts = len(pred)
    correct_count = 0
    ok_save_files = []
    with ProcessPoolExecutor(max_workers=len(gpus)) as executor:
        future_to_file = {
            executor.submit(run_script_on_gpu, p, t, f, TEMP_FILES_DIR, gpus[i % len(gpus)]): f
            for i, (p, t, f) in enumerate(zip(pred, test, files))
        }

        for future in as_completed(future_to_file):
            file_name = future_to_file[future]
            try:
                success = future.result()[0]
                if success:
                    correct_count += 1
                    ok_save_files.append(future.result()[1])
            except Exception as e:
                print(f"Error processing {file_name}: {e}", flush=True)

    # Calculate and print the correct execution rate
    correct_rate = (correct_count / total_scripts) * 100
    print(f"\nCorrect execution rate: {correct_rate:.2f}%", flush=True)
    print(ok_save_files)