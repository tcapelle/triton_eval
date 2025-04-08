import pytest
import torch
import os
import tempfile
from triton_eval.utils import to_device, set_gpu_arch, read_file

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
        valid_archs = ["Ampere", "Hopper"]
        set_gpu_arch(valid_archs)
        assert os.environ["TORCH_CUDA_ARCH_LIST"] == "Ampere;Hopper"

        # Invalid case
        invalid_archs = ["Pascal", "InvalidArchName"]
        with pytest.raises(ValueError, match="Invalid architecture: InvalidArchName.*"): # check regex
            set_gpu_arch(invalid_archs)

        # Ensure env var wasn't changed by the invalid call (it should still be from the valid call)
        assert os.environ["TORCH_CUDA_ARCH_LIST"] == "Ampere;Hopper"

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