import re
import ast
import numpy as np

from copy import deepcopy
import weave
from weave.flow.util import async_foreach
from weave.trace.op_caller import async_call

async def map(ds, func, num_proc=10):
    "Apply a function asynchronously to a dataset"
    async def apply_func(row: dict) -> dict:
        "Wrapper to make the function async"
        return await async_call(func, row)
    
    results = []
    n_complete = 0
    async for i, input_row, out_row in async_foreach(ds, apply_func, max_concurrent_tasks=num_proc):
        input_row_copy: dict = deepcopy(dict(input_row))
        input_row_copy.update(out_row)
        results.append(input_row_copy)  # Use the copy instead of original
        n_complete += 1
        print(f"Completed {n_complete} / {len(ds)}")
    return results



def parse_tensor_str(tensor_str):
    """
    Convert a single tensor string (as printed by PyTorch) into a NumPy array.
    - Strips 'tensor(' and trailing ')'
    - Removes any ', device='...'' annotations
    - Raises if ellipses '...' remain (to prompt full printing)
    """
    s = tensor_str.strip()
    # Remove 'tensor(' prefix and the matching trailing ')'
    if s.startswith('tensor(') and s.endswith(')'):
        s = s[len('tensor('):-1]
    # Drop device annotations
    s = re.sub(r",\s*device='[^']*'", "", s)
    # Check for ellipses
    if '...' in s:
        raise ValueError(
            "Ellipses detected in tensor printout; "
            "set torch.set_printoptions(threshold=int(1e9)) to print the full tensor."
        )
    # Parse nested Python lists into a NumPy array
    return np.array(ast.literal_eval(s))

@weave.op
def compare_outputs(expected_str, actual_str, rtol=1e-05, atol=1e-08):
    """
    Compare two stdout dumps of dicts of tensors (as strings).
    - Extracts each test_case by name
    - Parses into NumPy arrays
    - Checks shape equality
    - Uses allclose for floats (with given tolerances)
    - Uses array_equal for ints
    Prints a per-test summary and returns a list of (name, status, message).
    """
    # Regex to pull 'key': tensor([...]) entries
    entry_re = re.compile(
        r"'(?P<name>[^']+)'\s*:\s*(?P<tensor>tensor\([\s\S]*?\))(?:,|$)"
    )
    expected = {m.group('name'): m.group('tensor') for m in entry_re.finditer(expected_str)}
    actual   = {m.group('name'): m.group('tensor') for m in entry_re.finditer(actual_str)}
    
    results = []
    
    # Compare expected → actual
    for name, exp_str in expected.items():
        if name not in actual:
            results.append((name, 'MISSING_IN_ACTUAL', '', None))
            continue
        act_str = actual[name]
        try:
            a = parse_tensor_str(exp_str)
            b = parse_tensor_str(act_str)
        except Exception as e:
            results.append((name, 'PARSE_ERROR', str(e), None))
            continue
        
        if a.shape != b.shape:
            results.append((name, 'SHAPE_MISMATCH', f"{a.shape} vs {b.shape}", None))
            continue
        
        # Floating point or mixed dtypes → use allclose
        if np.issubdtype(a.dtype, np.floating) or np.issubdtype(b.dtype, np.floating):
            max_err = float(np.max(np.abs(a - b)))
            close = bool(np.allclose(a, b, rtol=rtol, atol=atol))
            status = 'PASS' if close else 'FAIL'
            results.append((name, status, f"max_error={max_err:.3e}", max_err))
        else:
            # Integer or exact comparison
            eq = bool(np.array_equal(a, b))
            status = 'PASS' if eq else 'FAIL'
            msg = 'exact_match' if eq else 'values_differ'
            results.append((name, status, msg, None))
    
    # Check for unexpected entries in actual
    for name in actual:
        if name not in expected:
            results.append((name, 'UNEXPECTED_IN_ACTUAL', '', None))
    
    # Print summary
    for name, status, msg, _ in results:
        print(f"{name}: {status} ({msg})")
    
    return results