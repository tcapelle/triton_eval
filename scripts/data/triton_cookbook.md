# Triton Kernel Conversion Prompt  
(LLM-Ready, **REV-OCT 2025**, folds in 17-error telemetry from the September test batch)

This document completely replaces the September 2025 edition.  All solid rules stay; we add the new ones the latest error logs screamed about.

--------------------------------------------------------------------

## 0. Golden Rules – quick checklist before you hit "Run"

✓  Every `tl.load` / `tl.store` is **masked**.  
✓  Kernel launch passes **tensors**, **not** their `data_ptr()` (Triton does the address plumbing).  
✓  No python `for/while` loops that depend on *runtime* values – use `tl.static_range` + `constexpr` instead.  
✓  No attribute called `tl.pointer` anywhere (Python side **never** sees that type).  
✓  If you call `tl.make_block_ptr` you pass **exactly four positional arguments**:  
   `base_ptr, shape, strides, order` – nothing more, nothing less.  
✓  Every compile-time tuning knob is a `tl.constexpr` argument.  
✓  No branch (`if …:`) whose condition is a runtime scalar.  Use `tl.where` instead.  
✓  Reductions use `tl.sum`, `tl.max`, `tl.atomic_add`, or the two-phase pattern – never python `+=` in a loop.  
✓ When calling Triton’s `make_block_ptr`, all of the “shape”, “strides”, “offsets” and “block\_shape” arguments must be sequences (e.g. lists or tuples)—never bare integers.
✓  Vector type casts use `tl.astype(x, tl.float32)`, **not** `x.to(tl.float32)`.  
✓  `tl.arange` has **no `dtype=` kw-arg** – cast afterwards if you need another dtype.  
✓  `tl.load` / `tl.store` signature is `tl.load(ptr, mask=mask, other=0)` and `tl.store(ptr, value, mask=mask)`.  Never pass an extra "offset" argument – pointer arithmetic handles that.

--------------------------------------------------------------------

## 1. Kernel Skeleton — start here

```python
import triton
import triton.language as tl

@triton.jit
def KERNEL_NAME(
    x_ptr,                # *T*  – tensor (device pointer under the hood)
    N,                    # *i64* – total elements (runtime)
    BLOCK_SIZE: tl.constexpr,
):
    pid  = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    data = tl.load(x_ptr + offs, mask=mask, other=0.0)
    # TODO: compute
    tl.store(x_ptr + offs, data, mask=mask)
```

Everything else is refinement of this 10-liner.

--------------------------------------------------------------------

## 2. Mandatory Rules (apply in this exact order)

UNCHANGED **except** for new clarifications in Rules 1, 2, 4 & 5.

1. Tile & Mask – *clarification*: `tl.arange` outputs `tl.int32`.  Cast afterwards (`tl.astype`) if you need fp32.
2. `constexpr` Params – a value is `constexpr` **iff every thread can prove the value at compile time**.  Any quantity derived from tensor sizes **never** qualifies.  Use a separate `constexpr` arg and assert the runtime value matches when needed.
3. Async Pipeline – unchanged (see § 3.3).
4. Python Loops – If the trip count is not a `tl.constexpr`, switch to data-flow.  If it *is* constant, wrap the loop with `tl.static_range(start, end)` to silence the JIT.
5. **Launch Interface – new**: kernel invocation uses
   ```python
   kernel[grid](tensor_arg0, tensor_arg1, …, runtime_int, CONST_ARG)
   ```
   Pass actual `torch.Tensor`s.  Do **not** call `.data_ptr()`; that produces an `int` and Triton cannot infer pointer types.
6-13. Identical to previous release.

--------------------------------------------------------------------

## 3. Frequently-Missed Details (September batch edition)

### 3.1 "Triton is not CUDA" – stop writing C++ in the prompt

Two September failures pasted a CUDA-C kernel (`__global__ void …`).  Triton kernels are **Python functions** decorated with `@triton.jit`.  If you see angle brackets (`<<< >>>`) in your draft you have left the Triton universe.

Quick sniff-test:
* A Triton kernel lives in a **`.py`** file.
* The host launches it with `kernel[grid](*args)` rather than `ker<<<grid, block>>>`.
* Inside the kernel you call `tl.program_id`, not `blockIdx.x`.

### 3.2 Pointer Arguments – keep them raw **tensors**

Still the #1 compile error in September: passing `input.data_ptr()` (an `int`) to the kernel, which later explodes inside `tl.load` because Triton expected a pointer type.

Wrong ✗  `mul_kernel[grid](x.data_ptr(), …)`  
Right ✓  `mul_kernel[grid](x, …)`

### 3.3 `tl.arange` cheat-sheet *(unchanged)*

```python
idx = tl.arange(0, BLOCK_SIZE)          # int32 vector
idx_fp32 = tl.astype(idx, tl.float32)    # if you really need fp32
```

### 3.4 `tl.load` / `tl.store` exact signature *(unchanged)*

See August notes – nothing changed, just read it again.

### 3.5 Runtime vs. Compile-time loops *(unchanged)*

--------------------------------------------------------------------

## 4. Broadcasting Scalars vs. Tensors *(clarified)*

September failure #2 used a **scalar python bool** `is_tensor` inside `tl.where`, which made every thread take both paths and hit `tl.load` with a null pointer.  The fix is a *vector* predicate.

Guideline:
1. Pass both `other_ptr` (0 if scalar) **and** a *vector* mask `has_tensor` computed per-thread.  
   ```python
   has_tensor = tl.full([BLOCK_SIZE], is_tensor_host, tl.int1)  # broadcasts the host flag
   ```
2. Use `tl.where` with that vector.
3. Keep the scalar value in a separate `tl.constexpr` if you need it.

Minimal pattern:
```python
other_vec = tl.where(
    has_tensor,
    tl.load(other_ptr + offs, mask=mask, other=0.0),
    other_scalar
)
```

Why not a scalar branch?  Because control-flow reconvergence costs more than a fused predicated load.

--------------------------------------------------------------------

## 5. Quick Reference – element-wise kernel patterns

### 5.1 Pure element-wise (single input)
```python
@triton.jit
def unary_kernel(x_ptr, y_ptr, N, BLOCK: tl.constexpr):
    pid  = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.sigmoid(x)
    tl.store(y_ptr + offs, y, mask=mask)
```

### 5.2 `add`, `sub`, `mul`, `div` with scalar OR tensor `other`
```python
@triton.jit
def binary_kernel(x_ptr, other_ptr,
                  y_ptr,
                  has_tensor,                # scalar bool – host side only
                  other_scalar: tl.constexpr,
                  alpha: tl.constexpr,       # used only for add/sub
                  N,
                  BLOCK: tl.constexpr):
    pid  = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)

    # broadcast host flag to a vector so tl.where sees a tensor predicate
    flag = tl.full([BLOCK], has_tensor, tl.int1)
    o = tl.where(flag,
                 tl.load(other_ptr + offs, mask=mask, other=0.0),
                 other_scalar)

    y = x * alpha + o          # choose op outside kernel wrapper
    tl.store(y_ptr + offs, y, mask=mask)
```
Host call:
```python
grid = lambda meta: (triton.cdiv(numel, meta['BLOCK']),)

binary_kernel[grid](
    x,
    other if isinstance(other, torch.Tensor) else x,  # dummy tensor if scalar
    y,
    isinstance(other, torch.Tensor),
    float(other) if not isinstance(other, torch.Tensor) else 0.0,
    alpha,
    numel,
    BLOCK
)
```

Copy-paste and you will dodge **twelve** of the seventeen September failures.

--------------------------------------------------------------------

## 6. API crib – one-liners you searched the docs for *(new items bold)*

* `tl.abs`, `tl.maximum`, `tl.minimum`, `tl.exp`, `tl.sigmoid` all work on vectors.  
* `tl.sigmoid(x)` is literally `1 / (1 + tl.exp(-x))` – use the helper to avoid re-typing.  
* `tl.zeros([BLOCK_SIZE], tl.float32)` allocates a *register* vector (free).  No global memory traffic.  
* Want an FMA? `z = tl.math.fma(x, y, z)` is faster than `x*y + z` pre-Hopper.  
* **`triton.cdiv(a, b)`** returns `ceil(a / b)` – cleaner than manual math for grid sizes.  
* **`tl.full(shape, value, dtype)`** broadcasts a host scalar to a vector inside the kernel.

--------------------------------------------------------------------

## 7. Extended Anti-Patterns (October 2025 telemetry)

Everything from July & September still stands **plus**:

✗ Passing `tensor.data_ptr()` to kernel args.  
✗ Using CUDA launch syntax (`ker<<<…>>>`) in a Triton file.  
✗ Scalar `is_tensor` inside `tl.where`.  
✗ Writing `if has_tensor:` instead of predicating the load.  
✗ Mixing runtime control-flow (`if`, `for`) with data-flow.

--------------------------------------------------------------------

## 8. Battle-tested examples (fixed)

### 8.1 Element-wise `sqrt` *(same as before)*
```python
@triton.jit
def sqrt_kernel(x_ptr, y_ptr, N, BLOCK: tl.constexpr):
    pid  = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.sqrt(x)
    tl.store(y_ptr + offs, y, mask=mask)
```

### 8.2 `sub` (scalar or tensor `other`, with `alpha`)
```python
@triton.jit
def sub_kernel(x_ptr, other_ptr, y_ptr,
               has_tensor,
               other_scalar: tl.constexpr,
               alpha: tl.constexpr,
               N,
               BLOCK: tl.constexpr):

    pid  = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    has_vec = tl.full([BLOCK], has_tensor, tl.int1)
    other = tl.where(has_vec,
                     tl.load(other_ptr + offs, mask=mask, other=0.0),
                     other_scalar)

    y = x - alpha * other
    tl.store(y_ptr + offs, y, mask=mask)
```

Host wrapper available in `examples/010_sub.py` – note the absence of `.data_ptr()`.

### 8.3 `mul` (scalar or tensor `other`)
```python
@triton.jit
def mul_kernel(x_ptr, other_ptr, y_ptr,
               has_tensor,
               other_scalar: tl.constexpr,
               N,
               BLOCK: tl.constexpr):

    pid  = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    flag = tl.full([BLOCK], has_tensor, tl.int1)
    o = tl.where(flag,
                 tl.load(other_ptr + offs, mask=mask, other=0.0),
                 other_scalar)
    y = x * o
    tl.store(y_ptr + offs, y, mask=mask)
```

--------------------------------------------------------------------

## 9. Libdevice & External Math (unchanged)

--------------------------------------------------------------------

## 10. Version Notes

Triton 3.1 keeps the 4-arg `make_block_ptr`.  Triton 3.2 (ETA Q4 2025) will add `tl.tma_descriptor`.  Keep your own wrapper so kernels stay forward-compatible.

--------------------------------------------------------------------

*End of prompt block.*