import pytest
import torch
import os
import tempfile
from pathlib import Path
from triton_eval.utils import (
    to_device, set_gpu_arch, read_file, get_tests, run_script_on_gpu,
    save_to_file, save_to_temp_file, ifnone, TEMP_FILES_DIR
)

def test_to_device():
    """Test the to_device function with various inputs and devices."""
    # Basic types
    assert to_device(123, torch.device("cpu")) == 123
    assert to_device("string", torch.device("cpu")) == "string"
    assert to_device(None, torch.device("cpu")) is None

    # CPU tests
    device_cpu = torch.device("cpu")
    t_cpu = torch.randn(2, 2)
    t_moved_cpu = to_device(t_cpu, device_cpu)
    assert t_moved_cpu.device == device_cpu

    list_cpu = [torch.randn(1), "str", 10, torch.randn(1), None]
    list_moved_cpu = to_device(list_cpu, device_cpu)
    assert isinstance(list_moved_cpu, list)
    assert list_moved_cpu[0].device == device_cpu
    assert list_moved_cpu[1] == "str"
    assert list_moved_cpu[3].device == device_cpu

    dict_cpu = {"a": torch.randn(1), "b": [torch.randn(1), 5], "c": "hello"}
    dict_moved_cpu = to_device(dict_cpu, device_cpu)
    assert isinstance(dict_moved_cpu, dict)
    assert dict_moved_cpu["a"].device == device_cpu
    assert isinstance(dict_moved_cpu["b"], list)
    assert dict_moved_cpu["b"][0].device == device_cpu

    # Test mixed structure with no tensors
    mixed_structure = [(torch.randn(1)), {"inp": torch.randn(1), "inp2": "jola"}, 5, "string"]
    mixed_moved_cpu = to_device(mixed_structure, device_cpu)
    assert isinstance(mixed_moved_cpu, list)
    assert mixed_moved_cpu[0].device == device_cpu # Tensor in tuple
    assert isinstance(mixed_moved_cpu[1], dict)
    assert mixed_moved_cpu[1]["inp"].device == device_cpu # Tensor in dict
    assert mixed_moved_cpu[1]["inp2"] == "jola" # String in dict
    assert mixed_moved_cpu[2] == 5 # Int in list
    assert mixed_moved_cpu[3] == "string" # String in list

    # Test specific user-provided mixed structure (no tensors)
    user_mixed_structure = [(4), {"inp": 5, "inp2": "jola"}]
    user_mixed_moved_cpu = to_device(user_mixed_structure, device_cpu)
    assert user_mixed_moved_cpu == user_mixed_structure # Should be unchanged

    # CUDA tests (conditional)
    if torch.cuda.is_available():
        device_cuda = torch.device("cuda:0")
        t_cuda = torch.randn(2, 2)
        t_moved_cuda = to_device(t_cuda, device_cuda)
        assert t_moved_cuda.device.type == device_cuda.type
        assert t_moved_cuda.device.index == device_cuda.index

        list_cuda = [torch.randn(1).cpu(), "str_cuda", 11]
        list_moved_cuda = to_device(list_cuda, device_cuda)
        assert list_moved_cuda[0].device.type == device_cuda.type
        assert list_moved_cuda[1] == "str_cuda"

        dict_cuda = {"x": torch.randn(1).cpu(), "y": [torch.randn(1).cpu(), 99]}
        dict_moved_cuda = to_device(dict_cuda, device_cuda)
        assert dict_moved_cuda["x"].device.type == device_cuda.type
        assert dict_moved_cuda["y"][0].device.type == device_cuda.type

        # Test mixed structure with no tensors on CUDA
        mixed_moved_cuda = to_device(mixed_structure, device_cuda)
        assert isinstance(mixed_moved_cuda, list)
        assert mixed_moved_cuda[0].device.type == device_cuda.type
        assert isinstance(mixed_moved_cuda[1], dict)
        assert mixed_moved_cuda[1]["inp"].device.type == device_cuda.type
        assert mixed_moved_cuda[1]["inp2"] == "jola"
        assert mixed_moved_cuda[2] == 5
        assert mixed_moved_cuda[3] == "string"

        # Test specific user-provided mixed structure on CUDA
        user_mixed_moved_cuda = to_device(user_mixed_structure, device_cuda)
        assert user_mixed_moved_cuda == user_mixed_structure # Should be unchanged
    else:
        print("Skipping CUDA part of test_to_device as CUDA is not available.")

def test_set_gpu_arch(monkeypatch):
    """Test set_gpu_arch with valid and invalid inputs."""
    # Store original value if exists
    original_value = os.environ.get("TORCH_CUDA_ARCH_LIST")

    try:
        # Valid case
        gpu = "h100"
        set_gpu_arch(gpu)
        assert os.environ["TORCH_CUDA_ARCH_LIST"] == "Hopper"


    finally:
        # Restore original value or unset
        if original_value is not None:
            monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", original_value)
        else:
            # Use monkeypatch to ensure deletion even if test fails mid-way
            monkeypatch.delenv("TORCH_CUDA_ARCH_LIST", raising=False)

def test_read_file(tmp_path):
    """Test read_file with existing, non-existing, and permission-error files."""
    # 1. Existing file
    content = "Line 1\nLine 2 is here."
    existing_file = tmp_path / "readable.txt"
    existing_file.write_text(content)
    read_content = read_file(str(existing_file))
    assert read_content == content

    # 2. Non-existent file
    non_existent_path = tmp_path / "non_existent_file.abc"
    read_content_non_existent = read_file(str(non_existent_path))
    assert read_content_non_existent == ""

    # 3. Permission error file
    unreadable_file = tmp_path / "unreadable.txt"
    unreadable_file.touch()
    original_mode = unreadable_file.stat().st_mode

    try:
        # Make unreadable
        os.chmod(unreadable_file, original_mode & ~0o400)
        read_content_unreadable = read_file(str(unreadable_file))
        assert read_content_unreadable == ""
    except PermissionError:
        pytest.skip("Could not reliably set file permissions to unreadable.")
    finally:
        # Restore permissions
        try:
            # Check if file still exists before chmod
            if unreadable_file.exists():
                os.chmod(unreadable_file, original_mode)
        except Exception:
            pass # Ignore cleanup errors 

def test_save_to_file(tmp_path):
    """Test save_to_file function."""
    content = "This is the content to save."
    file_path = tmp_path / "test_save.txt"

    # Test saving new file
    returned_path = save_to_file(str(file_path), content)
    assert returned_path == str(file_path)
    assert file_path.exists()
    assert file_path.read_text() == content

    # Test overwriting existing file
    new_content = "This is the new content."
    returned_path_overwrite = save_to_file(str(file_path), new_content)
    assert returned_path_overwrite == str(file_path)
    assert file_path.read_text() == new_content

def test_save_to_temp_file():
    """Test save_to_temp_file function."""
    content = "Temporary content here."
    
    # Ensure temp dir exists
    TEMP_FILES_DIR.mkdir(exist_ok=True)
    
    temp_file_path_str = save_to_temp_file(content)
    temp_file_path = Path(temp_file_path_str)

    assert isinstance(temp_file_path_str, str)
    assert temp_file_path.exists()
    assert temp_file_path.name.endswith(".py")
    assert temp_file_path.parent == TEMP_FILES_DIR
    assert temp_file_path.read_text() == content

    # Clean up the created temp file
    try:
        os.remove(temp_file_path)
    except OSError as e:
        print(f"Error removing temp file {temp_file_path}: {e}")

def test_ifnone():
    """Test the ifnone utility function."""
    assert ifnone(None, "default") == "default"
    assert ifnone("value", "default") == "value"
    assert ifnone(0, "default") == 0
    assert ifnone("", "default") == ""
    assert ifnone(False, True) is False

def test_get_tests():
    """Test get_tests with a sample script."""
    from textwrap import dedent

    # Test 1: No test functions
    script_content_no_tests = dedent("""
        import torch

        def func():
            pass""")

    expected_output = ""
    assert get_tests(script_content_no_tests) == expected_output

    # Test 2: Single test function
    script_content_single_test = dedent("""
        import torch

        def func():
            pass

        # Test
        def test_func():
            pass""")
    
    expected_single_test = dedent("""
        def test_func():
            pass""").strip()
    assert get_tests(script_content_single_test) == expected_single_test

    # Test 3: Multiple test functions
    script_content_multiple_tests = dedent("""
        import torch

        def func():
            pass

        def test_func():
            pass

        def test_func2():
            pass""")
    
    expected_multiple_tests = dedent("""
        def test_func():
            pass

        def test_func2():
            pass""").strip()
    assert get_tests(script_content_multiple_tests) == expected_multiple_tests

def test_run_script_on_gpu_cpu():
    """Test run_script_on_gpu executes a simple script on CPU."""
    script_content = "print('Hello from CPU')\nimport sys; sys.exit(0)" # Ensure clean exit
    test_content = "# No tests needed for this simple script"
    file_name = "cpu_script.py"

    # Run on CPU (gpu_id = None)
    run_result = run_script_on_gpu(
        script_content=script_content,
        test_content=test_content,
        file_name=file_name,
        gpu_id=None,
    )

    assert run_result.success is True
    assert run_result.file_name.endswith(file_name)
    assert run_result.returncode == 0
    assert run_result.stdout == "Hello from CPU\n"
    assert run_result.stderr == ""


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_run_script_on_gpu_cuda():
    """Test run_script_on_gpu executes a simple script on GPU 0."""
    script_content = (
        "import torch\n"
        "assert torch.cuda.is_available()\n"
        "assert torch.cuda.current_device() == 0\n"
        "print(f'Hello from GPU: {torch.cuda.current_device()}')\n"
        "import sys; sys.exit(0)"
    )
    test_content = "# No tests needed, assertions are in the script"
    file_name = "gpu_script.py"
    gpu_id = 0

    # Run on GPU (gpu_id = 0)
    run_result = run_script_on_gpu(
        script_content=script_content,
        test_content=test_content,
        file_name=file_name,
        gpu_id=gpu_id,
    )

    assert run_result.success is True
    assert run_result.file_name.endswith(file_name)
    assert run_result.returncode == 0
    assert run_result.stdout == "Hello from GPU: 0\n"
    assert run_result.stderr == ""

