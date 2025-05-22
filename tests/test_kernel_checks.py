import pytest
from triton_eval.kernel_checks import is_valid_kernel

# Test cases: (id, source_code, entrypoint, expected_result)
# expected_result is now a dict: {'is_valid': bool, 'reason': str}
kernel_check_test_cases = [
    # --- Valid Cases (expected_result = {'is_valid': True, 'reason': ''}) ---
    (
        "valid_kernel_entrypoint",
        """
import triton
import triton.language as tl
import torch

@triton.jit
def entrypoint_kernel(in_ptr, out_ptr, size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program(0, num_programs=BLOCK_SIZE) # Example primitive
    # ... kernel logic ...
""",
        "entrypoint_kernel",
        {'is_valid': True, 'reason': ''}
    ),
    (
        "valid_wrapper_triton_jit",
        """
import triton
import triton.language as tl
import torch

@triton.jit
def sub_kernel(in_ptr, out_ptr, size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program(0, num_programs=BLOCK_SIZE)
    tl.store(out_ptr + pid, tl.load(in_ptr + pid) + 1) # example primitives

def entrypoint_wrapper(a):
    out = torch.empty_like(a) # whitelisted
    grid = lambda meta: (triton.cdiv(a.numel(), meta['BLOCK_SIZE']),)
    sub_kernel[grid](a, out, a.numel(), BLOCK_SIZE=1024)
    return out
""",
        "entrypoint_wrapper",
        {'is_valid': True, 'reason': ''}
    ),
    (
        "valid_wrapper_tl_jit",
        """
import triton.language as tl
import torch
import triton # needed for triton.cdiv if used

@tl.jit # Using tl.jit
def another_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program(0, axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = x + 1 # some operation
    tl.store(y_ptr + offsets, output, mask=mask)

def entrypoint_wrapper_tl(x):
    y = torch.empty_like(x) # whitelisted
    grid = lambda META: (triton.cdiv(x.numel(), META['BLOCK_SIZE']),)
    another_kernel[grid](x, y, x.numel(), BLOCK_SIZE=1024)
    return y
""",
        "entrypoint_wrapper_tl",
        {'is_valid': True, 'reason': ''}
    ),
    (
        "valid_wrapper_whitelisted_torch",
        """
import triton
import triton.language as tl
import torch

@triton.jit
def dummy_kernel(ptr):
    pass

def entrypoint_uses_whitelisted(a):
    b = torch.empty_like(a, dtype=torch.float32) # whitelisted
    c = torch.zeros_like(b, device=a.device)    # whitelisted
    s = a.shape                                 # whitelisted attribute access
    n = torch.numel(a)                          # whitelisted
    if torch.is_tensor(a):                      # whitelisted
        pass
    grid = (1,)
    dummy_kernel[grid](a)
    return c
""",
        "entrypoint_uses_whitelisted",
        {'is_valid': False, 'reason': 'Triton kernel has an empty body.'}
    ),
    (
        "valid_alias_unused",
        """
import triton
import triton.language as tl
import torch as my_torch # aliased

@triton.jit
def kernel_alias_ok(in_ptr):
    data = tl.load(in_ptr)
    # no my_torch usage here
    tl.store(in_ptr, data)

def entrypoint_alias(a):
    grid = (1,)
    kernel_alias_ok[grid](a)
    return a
""",
        "entrypoint_alias",
        {'is_valid': True, 'reason': ''}
    ),
    (
        "valid_from_import_unused",
        """
import triton
import triton.language as tl
from torch import add, empty # imported but add not used in kernel

@triton.jit
def kernel_from_import_ok(in_ptr):
    data = tl.load(in_ptr)
    # no 'add' usage here
    tl.store(in_ptr, data)

def entrypoint_from_import(a):
    b = empty(a.shape, dtype=a.dtype, device=a.device) # whitelisted (assuming empty is from torch)
    grid = (1,)
    kernel_from_import_ok[grid](a)
    return b
""",
        "entrypoint_from_import",
        {'is_valid': True, 'reason': ''}
    ),
    # --- Invalid Cases (expected_result = {'is_valid': False, 'reason': ...}) ---
    (
        "invalid_torch_in_kernel_wrapper_entry",
        """
import triton
import triton.language as tl
import torch

@triton.jit
def kernel_with_torch(in_ptr, out_ptr):
    data = tl.load(in_ptr)
    res = torch.add(data, 1) # Problem!
    tl.store(out_ptr, res)

def entrypoint_calls_bad_kernel(a):
    out = torch.empty_like(a)
    grid = (1,)
    kernel_with_torch[grid](a, out)
    return out
""",
        "entrypoint_calls_bad_kernel",
        {'is_valid': False, 'reason': 'Torch usage detected inside a Triton kernel.'}
    ),
    (
        "invalid_torch_in_kernel_kernel_entry",
        """
import triton
import triton.language as tl
import torch

@triton.jit
def kernel_with_torch(in_ptr, out_ptr): # This kernel is the entrypoint
    data = tl.load(in_ptr)
    res = torch.add(data, 1) # Problem!
    tl.store(out_ptr, res)
""",
        "kernel_with_torch",
        {'is_valid': False, 'reason': 'Torch usage detected inside a Triton kernel.'}
    ),
    (
        "invalid_aliased_torch_in_kernel",
        """
import triton
import triton.language as tl
import torch as my_torch

@triton.jit
def kernel_with_aliased_torch(in_ptr):
    val = tl.load(in_ptr)
    val = my_torch.exp(val) # Problem!
    tl.store(in_ptr, val)

def entrypoint_calls_bad_aliased_kernel(a):
    grid = (1,)
    kernel_with_aliased_torch[grid](a)
    return a
""",
        "entrypoint_calls_bad_aliased_kernel",
        {'is_valid': False, 'reason': 'Torch usage detected inside a Triton kernel.'}
    ),
    (
        "invalid_from_import_in_kernel",
        """
import triton
import triton.language as tl
from torch import add, sin # sin is ok if not used

@triton.jit
def kernel_with_imported_torch_func(in_ptr):
    val = tl.load(in_ptr)
    val = add(val, 1) # Problem! (add is from torch)
    tl.store(in_ptr, val)

def entrypoint_calls_bad_from_kernel(a):
    grid = (1,)
    kernel_with_imported_torch_func[grid](a)
    return a
""",
        "entrypoint_calls_bad_from_kernel",
        {'is_valid': False, 'reason': 'Torch usage detected inside a Triton kernel.'}
    ),
    (
        "invalid_no_kernels_defined",
        """
import torch

def entrypoint_no_kernel(a):
    return torch.add(a, 1) # Using torch directly
""",
        "entrypoint_no_kernel",
        {'is_valid': False, 'reason': 'Entrypoint is a wrapper function, but no Triton kernels are defined in the source.'}
    ),
    (
        "invalid_kernel_defined_not_called",
        """
import triton
import triton.language as tl
import torch

@triton.jit
def defined_but_not_used_kernel(ptr):
    pass

def entrypoint_does_not_call(a):
    # defined_but_not_used_kernel is not called
    return torch.add(a, 1) # Uses torch instead
""",
        "entrypoint_does_not_call",
        {'is_valid': False, 'reason': 'Triton kernel has an empty body.'}
    ),
    (
        "invalid_wrapper_uses_non_whitelisted_torch",
        """
import triton
import triton.language as tl
import torch

@triton.jit
def some_kernel(ptr):
    tl.store(ptr, tl.load(ptr) + 1)

def entrypoint_wrapper_bad_torch(a):
    grid = (1,)
    some_kernel[grid](a)
    # After kernel call, uses disallowed torch op
    b = torch.matmul(a, a) # Problem! matmul is not whitelisted for typical wrappers
    return b
""",
        "entrypoint_wrapper_bad_torch",
        {'is_valid': False, 'reason': 'Entrypoint is a wrapper function that calls a Triton kernel, but also performs non-whitelisted torch operations.'}
    ),
    (
        "invalid_kernel_entrypoint_with_torch_op",
        """
import triton
import triton.language as tl
import torch

@triton.jit
def entrypoint_kernel_bad(in_ptr):
    data = tl.load(in_ptr)
    result = torch.sigmoid(data) # Problem!
    tl.store(in_ptr, result)
""",
        "entrypoint_kernel_bad",
        {'is_valid': False, 'reason': 'Torch usage detected inside a Triton kernel.'}
    ),
    (
        "invalid_malformed_indentation",
        "def entrypoint_malformed(a):\\n print(a)\\n   x = 1",
        "entrypoint_malformed",
        {'is_valid': False, 'reason': 'Syntax error parsing source code.'}
    ),
    (
        "invalid_malformed_unclosed_paren",
        "def entrypoint_malformed_paren(a:\\n    return a + (",
        "entrypoint_malformed_paren",
        {'is_valid': False, 'reason': 'Syntax error parsing source code.'}
    ),
    (
        "invalid_markdown_text_not_code",
        """
# This is a markdown file

It has some text but no Python code that can be parsed meaningfully.

```python
# This looks like code, but the overall string is not valid module source
def fake_func():
    pass
```
""",
        "any_entrypoint",
        {'is_valid': False, 'reason': 'Syntax error parsing source code.'}
    ),
    (
        "invalid_wrapper_aliased_non_whitelisted_torch",
        """
import triton
import triton.language as tl
import torch
t_alias = torch # alias

@triton.jit
def my_valid_kernel(ptr):
    tl.store(ptr, tl.load(ptr) * 2)

def entrypoint_wrapper_aliased_issue(x):
    grid = (1,)
    my_valid_kernel[grid](x)
    y = t_alias.sum(x) # Problem! sum is not whitelisted
    return y
""",
        "entrypoint_wrapper_aliased_issue",
        {'is_valid': False, 'reason': 'Entrypoint is a wrapper function that calls a Triton kernel, but also performs non-whitelisted torch operations.'}
    ),
    (
        "invalid_wrapper_from_imported_non_whitelisted_torch",
        """
import triton
import triton.language as tl
import torch
from torch import sum as my_sum_func # from ... import ... as

@triton.jit
def another_valid_kernel(ptr):
    tl.store(ptr, tl.load(ptr) - 1)

def entrypoint_wrapper_from_import_issue(z):
    grid = (1,)
    another_valid_kernel[grid](z)
    res = my_sum_func(z) # Problem! sum (via my_sum_func) is not whitelisted
    return res
""",
        "entrypoint_wrapper_from_import_issue",
        {'is_valid': False, 'reason': 'Entrypoint is a wrapper function that calls a Triton kernel, but also performs non-whitelisted torch operations.'}
    ),
    (
        "invalid_random_text",
        "This is just some random text, not Python code at all.",
        "any_entrypoint",
        {'is_valid': False, 'reason': 'Syntax error parsing source code.'}
    ),
    (
        "invalid_random_text_with_torch_word",
        "Some gibberish here and there torch and more text.",
        "any_entrypoint",
        {'is_valid': False, 'reason': 'Syntax error parsing source code.'}
    )
]

@pytest.mark.parametrize("test_id, src, entrypoint, expected_result", kernel_check_test_cases)
def test_is_valid_kernel(test_id, src, entrypoint, expected_result):
    """
    Tests the is_valid_kernel function with various scenarios.
    - test_id: A descriptive name for the test case.
    - src: The Python source code string to analyze.
    - entrypoint: The name of the entrypoint function in the source code.
    - expected_result: Dict {'is_valid': bool, 'reason': str}.
    """
    actual_result = is_valid_kernel(src, entrypoint)
    assert actual_result == expected_result, f"Test case '{test_id}' failed. Expected {expected_result}, got {actual_result}"