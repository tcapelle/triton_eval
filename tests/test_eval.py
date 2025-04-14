import pytest
import torch
import json # Keep json for potential metadata checks if needed later
from triton_eval.eval import (
    eval_kernel_against_ref,
    get_error_name,
    register_and_format_exception,
    _parse_inputs,
)
from triton_eval.utils import set_gpu_arch
# --- Global Constants and Setup ---
CUDA_IS_AVAILABLE = torch.cuda.is_available()

# Read model sources once at the module level
MODEL_EX_ADD_PATH = "tests/test_data/cuda/model_ex_add.py"
MODEL_NEW_EX_ADD_PATH = "tests/test_data/cuda/model_new_ex_add.py"

try:
    with open(MODEL_EX_ADD_PATH, "r") as f:
        original_model_src = f.read()
except FileNotFoundError:
    original_model_src = None # Handle missing file gracefully in tests

try:
    with open(MODEL_NEW_EX_ADD_PATH, "r") as f:
        custom_model_src = f.read()
except FileNotFoundError:
    custom_model_src = None # Handle missing file gracefully in tests


@pytest.mark.skipif(not CUDA_IS_AVAILABLE, reason="CUDA not available")
@pytest.mark.skipif(original_model_src is None or custom_model_src is None, reason="Test data files not found")
def test_cuda_add_kernel_evaluation():
    """Test basic eval_kernel_against_ref: compilation and correctness."""
    set_gpu_arch()
    eval_result = eval_kernel_against_ref(
        original_model_src=original_model_src,
        custom_model_src=custom_model_src,
        original_entry_point="Model",
        custom_entry_point="ModelNew",
        device=torch.device("cuda:0"),
        backend='cuda',
        num_correct_trials=1,
        measure_performance=False # Focus on correctness
    )
    assert eval_result is not None, "Evaluation unexpectedly returned None."
    assert eval_result.compiled, f"Kernel compilation failed. Metadata: {eval_result.metadata}"
    assert eval_result.correctness, f"Kernel correctness check failed. Metadata: {eval_result.metadata}"

def test_get_error_name():
    """Test get_error_name extracts correct names."""
    try:
        raise ValueError()
    except ValueError as e:
        assert get_error_name(e) == "builtins.ValueError"
    try:
        raise ImportError()
    except ImportError as e:
        assert get_error_name(e) == "builtins.ImportError"


def test_register_and_format_exception():
    """Test exception formatting and truncation."""
    metadata = {}
    try: 1 / 0
    except ZeroDivisionError as e:
        meta_short = register_and_format_exception("err", e, {}, truncate=True, max_length=5)
        meta_untruncated = register_and_format_exception("err", e, {}, truncate=False)
        meta_long_trunc = register_and_format_exception("err", e, {}, truncate=True, max_length=50)

    assert meta_short["err"] == "di..."
    assert meta_untruncated["err"] == "division by zero"
    assert meta_long_trunc["err"] == "division by zero" # Not truncated as it's shorter than 50

def test_parse_inputs_structure():
    """Test _parse_inputs structure parsing (CPU only, no tensors)."""
    mock_device = "cpu"

    # Flat list
    args, kwargs = _parse_inputs([1, "a"], mock_device)
    assert args == [1, "a"]
    assert kwargs == {}

    # Structured list/tuple
    args_s, kwargs_s = _parse_inputs((["a"], {"k": 1}), mock_device)
    assert args_s == ["a"]
    assert kwargs_s == {"k": 1}

    # Empty cases
    args_e, kwargs_e = _parse_inputs([], mock_device)
    assert args_e == [] and kwargs_e == {}
    args_es, kwargs_es = _parse_inputs([[], {}], mock_device)
    assert args_es == [] and kwargs_es == {}

    # Malformed treated as flat
    args_m1, kwargs_m1 = _parse_inputs([["a"], {"k": "v"}, "extra"], mock_device)
    assert args_m1 == [["a"], {"k": "v"}, "extra"]
    assert kwargs_m1 == {}
    args_m2, kwargs_m2 = _parse_inputs([["a"], ["not_dict"]], mock_device)
    assert args_m2 == [["a"], ["not_dict"]]
    assert kwargs_m2 == {}

# --- Triton Test ---

MODEL_EMBED_REF_PATH = "tests/test_data/triton/embed_code.py"
MODEL_EMBED_TRITON_PATH = "tests/test_data/triton/embed_triton.py"

try:
    with open(MODEL_EMBED_REF_PATH, "r") as f:
        triton_ref_model_src = f.read()
except FileNotFoundError:
    triton_ref_model_src = None

try:
    with open(MODEL_EMBED_TRITON_PATH, "r") as f:
        triton_custom_model_src = f.read()
except FileNotFoundError:
    triton_custom_model_src = None

@pytest.mark.skipif(not CUDA_IS_AVAILABLE, reason="CUDA not available")
@pytest.mark.skipif(triton_ref_model_src is None or triton_custom_model_src is None, reason="Triton test data files not found")
def test_triton_embed_kernel_evaluation():
    """Test evaluation of a Triton kernel against its reference."""
    eval_result = eval_kernel_against_ref(
        original_model_src=triton_ref_model_src,
        custom_model_src=triton_custom_model_src,
        original_entry_point="LinearEmbedding",
        custom_entry_point="LinearEmbeddingNew",
        device=torch.device("cuda:0"),
        backend='triton', # Specify Triton backend
        num_correct_trials=1,
        measure_performance=False
    )
    assert eval_result is not None, "Evaluation unexpectedly returned None."
    # Triton compilation happens during the first run, so we check correctness first
    # If correctness passes, compilation implicitly succeeded.
    assert eval_result.correctness, f"Triton kernel correctness check failed. Metadata: {eval_result.metadata}"
    # We might not get a specific 'compiled=True' flag easily for Triton JIT,
    # but correctness implies it worked.
    # assert eval_result.compiled # This might be unreliable for Triton
