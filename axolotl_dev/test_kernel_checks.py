import pytest
from kernel_checks import uses_torch_in_kernel

def test_uses_torch_in_kernel_valid_cases():
    # Valid: Entrypoint is a clean Triton kernel
    src_valid_kernel_entrypoint = """
import triton
import triton.language as tl
import torch

@triton.jit
def entrypoint_kernel(in_ptr, out_ptr, size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program(0, num_programs=BLOCK_SIZE) # Example primitive
    # ... kernel logic ...
"""
    assert not uses_torch_in_kernel(src_valid_kernel_entrypoint, "entrypoint_kernel")

    # Valid: Entrypoint is a wrapper calling a clean Triton kernel (@triton.jit)
    src_valid_wrapper_triton_jit = """
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
"""
    assert not uses_torch_in_kernel(src_valid_wrapper_triton_jit, "entrypoint_wrapper")

    # Valid: Entrypoint is a wrapper calling a clean Triton kernel (@tl.jit - assuming tl is triton.language)
    src_valid_wrapper_tl_jit = """
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
"""
    assert not uses_torch_in_kernel(src_valid_wrapper_tl_jit, "entrypoint_wrapper_tl")

    # Valid: Wrapper uses only whitelisted torch operations
    src_valid_wrapper_whitelisted_torch = """
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
"""
    assert not uses_torch_in_kernel(src_valid_wrapper_whitelisted_torch, "entrypoint_uses_whitelisted")

    # Valid: torch imported with alias, not used in kernel
    src_valid_alias_unused = """
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
"""
    assert not uses_torch_in_kernel(src_valid_alias_unused, "entrypoint_alias")

    # Valid: Symbols imported from torch, not used in kernel
    src_valid_from_import_unused = """
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
"""
    assert not uses_torch_in_kernel(src_valid_from_import_unused, "entrypoint_from_import")

def test_uses_torch_in_kernel_invalid_cases():
    # Invalid: torch.add directly inside kernel
    src_invalid_torch_in_kernel = """
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
"""
    assert uses_torch_in_kernel(src_invalid_torch_in_kernel, "entrypoint_calls_bad_kernel")
    assert uses_torch_in_kernel(src_invalid_torch_in_kernel, "kernel_with_torch") # Also invalid if kernel is entrypoint

    # Invalid: Aliased torch used inside kernel
    src_invalid_aliased_torch_in_kernel = """
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
"""
    assert uses_torch_in_kernel(src_invalid_aliased_torch_in_kernel, "entrypoint_calls_bad_aliased_kernel")

    # Invalid: from torch import add; add() used inside kernel
    src_invalid_from_import_in_kernel = """
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
"""
    assert uses_torch_in_kernel(src_invalid_from_import_in_kernel, "entrypoint_calls_bad_from_kernel")

    # Invalid: Entrypoint is wrapper, no Triton kernels defined
    src_invalid_no_kernels = """
import torch

def entrypoint_no_kernel(a):
    return torch.add(a, 1) # Using torch directly
"""
    assert uses_torch_in_kernel(src_invalid_no_kernels, "entrypoint_no_kernel")

    # Invalid: Entrypoint is wrapper, Triton kernels defined, but wrapper doesn't call them
    src_invalid_kernel_not_called = """
import triton
import triton.language as tl
import torch

@triton.jit
def defined_but_not_used_kernel(ptr):
    pass

def entrypoint_does_not_call(a):
    # defined_but_not_used_kernel is not called
    return torch.add(a, 1) # Uses torch instead
"""
    assert uses_torch_in_kernel(src_invalid_kernel_not_called, "entrypoint_does_not_call")

    # Invalid: Wrapper calls Triton kernel, but also uses non-whitelisted torch ops
    src_invalid_wrapper_uses_bad_torch = """
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
"""
    assert uses_torch_in_kernel(src_invalid_wrapper_uses_bad_torch, "entrypoint_wrapper_bad_torch")
    
    # Invalid: Entrypoint is a Triton kernel itself but contains torch op
    src_invalid_kernel_entrypoint_with_torch = """
import triton
import triton.language as tl
import torch

@triton.jit
def entrypoint_kernel_bad(in_ptr):
    data = tl.load(in_ptr)
    result = torch.sigmoid(data) # Problem!
    tl.store(in_ptr, result)
"""
    assert uses_torch_in_kernel(src_invalid_kernel_entrypoint_with_torch, "entrypoint_kernel_bad")

    # Invalid: Malformed Python code (Indentation error)
    src_malformed_indent = "def entrypoint_malformed(a):\\n print(a)\\n   x = 1" 
    assert uses_torch_in_kernel(src_malformed_indent, "entrypoint_malformed")

    # Invalid: Malformed Python code (Unclosed parenthesis)
    src_malformed_paren = "def entrypoint_malformed_paren(a:\\n    return a + (" 
    assert uses_torch_in_kernel(src_malformed_paren, "entrypoint_malformed_paren")

    # Invalid: Not Python code at all (Markdown text)
    src_markdown_text = """
# This is a markdown file

It has some text but no Python code that can be parsed meaningfully.

```python
# This looks like code, but the overall string is not valid module source
def fake_func():
    pass
```
"""
    assert uses_torch_in_kernel(src_markdown_text, "any_entrypoint")
    
    # Invalid: Wrapper calls triton kernel, but also uses aliased non-whitelisted torch op.
    src_invalid_wrapper_aliased_bad_torch = """
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
"""
    assert uses_torch_in_kernel(src_invalid_wrapper_aliased_bad_torch, "entrypoint_wrapper_aliased_issue")

    # Invalid: Wrapper calls triton kernel, but also uses non-whitelisted torch op (imported via from)
    src_invalid_wrapper_from_imported_bad_torch = """
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
"""
    assert uses_torch_in_kernel(src_invalid_wrapper_from_imported_bad_torch, "entrypoint_wrapper_from_import_issue")

    # Invalid: Random text that cannot be parsed as Python
    src_random_text = "This is just some random text, not Python code at all."
    assert uses_torch_in_kernel(src_random_text, "any_entrypoint")

    # Invalid: Random unformatted text that happens to contain the word torch
    src_random_text_with_torch = "Some gibberish here and there torch and more text."
    assert uses_torch_in_kernel(src_random_text_with_torch, "any_entrypoint")

    # Test case from Weave trace (potentially problematic due to torch._C usage in wrapper)
    src_weave_example_potentially_invalid = """
import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
# Simulating the torch._C imports as they might appear
# In a real scenario, these would be actual C extension objects/functions
class MockTorchC:
    class _dynamo:
        class guards:
            def assert_size_stride(self, *args, **kwargs): pass
            def _empty_strided_cuda(self, *args, **kwargs): 
                # This is the problematic call if not whitelisted and considered computation
                return torch.empty(1) # return a dummy tensor for flow
    _C = _dynamo()
    def _cuda_getCurrentRawStream(self, *args, **kwargs): return 0

# Make them available as if imported from torch._C
assert_size_stride = MockTorchC._C.guards.assert_size_stride
empty_strided_cuda = MockTorchC._C.guards._empty_strided_cuda
get_raw_stream = MockTorchC._cuda_getCurrentRawStream

@triton.jit
def triton_poi_fused_mul_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    # xnumel = 256 # This was in the image, but causes unused var if not used below
    pid = tl.program_id(0)
    xoffset = pid * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)
    xmask = xindex < xnumel # xnumel used here
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 2.0
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + x0, tmp2, xmask)

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4, 4, 4), (64, 16, 4, 1))
    with torch.cuda.DeviceGuard(0):
        torch.cuda.set_device(0) # whitelisted
        # The following line uses empty_strided_cuda, which is from torch._C
        # and not on the TORCH_WRAPPER_WHITELIST directly by that name.
        # Our current checker may not flag this as a torch op if alias tracking isn't deep enough.
        buf0 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_mul_0[grid(256)](arg0_1, buf0, 256, XBLOCK=256)
    return buf0
"""
    # This assertion depends on how strictly we define "disallowed torch op in wrapper".
    # If empty_strided_cuda (from torch._C) is considered disallowed because it's not whitelisted,
    # this should be True. Current checker might say False due to alias tracking limitations.
    # Let's assert True, expecting the check to be strict or to reveal this limitation.
    assert uses_torch_in_kernel(src_weave_example_potentially_invalid, "call"), \
        "Expected Weave example to be flagged due to non-whitelisted torch._C aliased functions in wrapper."