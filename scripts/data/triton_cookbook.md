# Triton Kernel Conversion Prompt  
(LLM-Ready, **REV-DEC 2025**, folds in 6-error telemetry from the November test batch)

This document *completely* replaces the November 2025 edition.  Anything not shown here is obsolete.

--------------------------------------------------------------------

## 0. GOLDEN RULES – **quick checklist before you hit “Run”**

✓  Triton supports **exactly three grid dimensions** – `program_id(0|1|2)`.  If your algorithm needs a 4-D launch, *flatten* extra axes into one of the three.
✓  Every `tl.load` / `tl.store` is **masked**.  
✓  Kernel launch passes **tensors**, **not** their `data_ptr()` (Triton does the address plumbing).  
✓  No python `for/while` loops that depend on *runtime* values – use `tl.static_range` + `constexpr` instead.  
✓  No attribute called `tl.pointer` anywhere (Python side **never** sees that type).  
✓  If you call `tl.make_block_ptr` you pass **exactly four positional arguments**:  
   `base_ptr, shape, strides, order` – nothing more, nothing less.  
✓  Every compile-time tuning knob is a `tl.constexpr` argument **and you only tag it `constexpr` if the host passes a literal / hard-coded value**.  
✓  No branch (`if …:`) whose condition is a runtime scalar.  Use `tl.where` or predicated arithmetic instead.  
✓  Reductions use `tl.sum`, `tl.max`, `tl.atomic_add`, or the two-phase pattern – never python `+=` in a loop.  
✓  When calling `tl.make_block_ptr`, all of the “shape”, “strides”, “offsets” and “block_shape” arguments must be sequences (e.g. lists or tuples)—never bare integers.  
✓  Vector type casts use **`tl.astype(x, tl.float32)` – *never* `tl.cast`**.  
✓  `tl.arange` has **no `dtype=` kw-arg** – cast afterwards if you need another dtype.  
✓  `tl.load` / `tl.store` signature is `tl.load(ptr, mask=mask, other=0)` and `tl.store(ptr, value, mask=mask)`.  Never pass an extra "offset" argument – pointer arithmetic handles that.  
✓  **No python collections or slice assignment inside the kernel.**  A `list`, `dict`, or `flat[:, i] = …` lives on the host; Triton tensors are *immutable*.  Build a *new* tensor with `tl.where` or `tl.concatenate` instead.  
✓  **Every numeric literal that participates in Triton math is written as a Triton value**: `0.0 → tl.zeros_like(x)` or `tl.full([], 0.0, x.dtype)` if scalar, never bare python `0.0`.  
✓  **Every reduction call explicitly states an axis** (`tl.sum(x, axis=0)`) so you don’t get surprised by the default.

--------------------------------------------------------------------

## 1. KERNEL SKELETON — start here  *(unchanged)*
```python
import triton
import triton.language as tl

@triton.jit
def KERNEL_NAME(
    x_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid  = tl.program_id(0)              # 0 ≤ pid < grid[0]
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    data = tl.load(x_ptr + offs, mask=mask, other=0.0)
    # TODO: compute
    tl.store(x_ptr + offs, data, mask=mask)
```

--------------------------------------------------------------------

## 2. MANDATORY RULES (apply in this exact order)

1. **Tile & Mask** – `tl.arange` outputs `tl.int32`.  Cast afterwards (`tl.astype`) if you need fp32.  
2. **`constexpr` Params – expanded**: a value is `constexpr` **iff every thread can prove the value at compile time**.  Any quantity derived from tensor sizes, dropout probabilities computed at runtime, or random seeds **never** qualifies.  Passing a runtime float into a `: tl.constexpr` argument is the fastest route to a cryptic compiler error.
3. **Grid Rank = 3** – Triton exposes `program_id(0)`, `program_id(1)`, and `program_id(2)`.  Need more?  Flatten: e.g. `(N,H,W,C)` → let `grid = (N, H, W*C)` and recover `c = pid2 % C`, `w = pid2 // C` inside the kernel.  Avoid `program_id(3)` – it doesn’t exist.
4. **Python Loops** – If the trip count is not a `tl.constexpr`, switch to data-flow.  If it *is* constant, wrap the loop with `tl.static_range(start, end)` to silence the JIT.  *Never* build or mutate a python collection inside the loop – Triton tensors are immutable.  Use `tl.where` to create new tensors instead of slice assignment.
5. **Launch Interface – reminder**: kernel invocation uses
```python
kernel[grid](tensor_arg0, tensor_arg1, …, runtime_int, CONST_ARG)
```
Pass actual `torch.Tensor`s.  Do **not** call `.data_ptr()`; that produces an `int` and Triton cannot infer pointer types.
6. **Scalar Branches – new**: All scalar control-flow inside a kernel must be *compile-time*.  Replace
```python
if runtime_flag:
    …
else:
    …
```
with
```python
flag = tl.full([VEC], runtime_flag, tl.int1)
val  = tl.where(flag, pathA, pathB)
```
7. **No `tl.cast`** – The only legal caster is `tl.astype`.  Everywhere else you want a new dtype, write `tl.astype(x, NEW_DTYPE)`.
8. **No in-place tensor writes** – `x += y`, `flat[:, i] = z`, or `A[i] = …` on Triton tensors is illegal.  Build a new tensor: `x = x + y`; `flat = tl.where(cond, new_val, flat)`.
9. **Mask type must be a vector** – boolean scalars (`True/False`) cannot be `&`-ed with vector masks.  Promote scalars via `tl.full([], 1, tl.int1)` or compare a scalar to itself (`tl.zeros([], tl.int32) == 0`).
10. **Reductions explicit axis** – already covered but repeated here because half the November failures missed it.

--------------------------------------------------------------------

## 3. FREQUENTLY-MISSED DETAILS (November batch edition)

### 3.1 Three-dimensional grid, period
Trying `program_id(3)` triggered two separate ICEs this month.  Flatten extra dimensions early and comment the decoding math so reviewers can follow.

### 3.2  *tl.astype* not *tl.cast*
`tl.cast` is a Python helper used by the Triton compiler itself – it is **not part of the public DSL**.  Use `tl.astype` everywhere.  If you see `tl.cast` your kernel will raise “expected tl.tensor, got …”.

### 3.3 Register tensors are immutable
All six new failures tried slice assignment (`buf[:, i] = x`) or in-place fma (`acc += val`).  Triton tensors are SSA values; build a **new** tensor instead;
```python
buf = tl.where(selector, new_val, buf)   # not buf[:, i] = …
acc = acc + val                          # not acc += val
```

### 3.4 Broadcasting scalars
A python int/float on the right-hand side of `+` promotes to host scalar, not a Triton value.  Correct idioms:
```python
alpha = tl.full([], runtime_scalar, tl.float32)   # scalar
vec   = tl.full([BLOCK], runtime_scalar, tl.float32)  # vector
```
Then use `alpha`/`vec` in expressions.

### 3.5 `tl.num_programs(axis)` is compile-time only
It exists mainly for autotuners.  Do **not** multiply it into pointers – you already know your launch grid on the host.  Pre-compute strides outside the kernel and pass them in as arguments.

### 3.6 Pointer arithmetic with scalars
`pid = tl.program_id(0)` is a scalar **tensor**, not a python int.  When you add it to a vector of offsets, cast/broadcast: `ptr = base_ptr + pid * stride + vec_offs` where `stride` is a python int and `vec_offs` is a Triton vector.

--------------------------------------------------------------------

## 4. “HOW DO I FLATTEN MY GRID?” quick patterns

1. 4-D tensor point-wise (`N,C,H,W`):
```python
grid = (N, triton.cdiv(C*H*W, BLOCK))
...
idx  = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
w    = idx % W
h    = (idx // W) % H
c    = (idx // (H*W)) % C
```
2. GEMM fusion (`M,N` tiles per CTA): use `(triton.cdiv(M, TM), triton.cdiv(N, TN))` and keep `K` inside the kernel with a static loop.

--------------------------------------------------------------------

## 5. UPDATED QUICK REFERENCE – extra one-liners
* `tl.astype`, `tl.full`, `tl.zeros_like`, `triton.cdiv`, `triton.next_power_of_two`.
* `tl.math.fma(x, y, z)` – still valid, but remember to create `z` without `+=`.
* `tl.atomic_add(ptr, val)` when many CTAs write the same address – median-pool accumulation is **atomic territory**.

--------------------------------------------------------------------

## 6. EXTENDED ANTI-PATTERNS (December 2025 telemetry)

✗ `program_id(3)` or higher.  
✗ `tl.cast` anywhere in the kernel.  
✗ Slice assignment (`buf[:, i] = …`).  
✗ Building or mutating python containers inside a kernel.  
✗ Combining python booleans with Triton masks via `&`.  
✗ Using `tl.num_programs()` inside pointer arithmetic.  
✗ Forgetting to `tl.astype` after `tl.arange`.  
✗ Reductions without `axis=`.

--------------------------------------------------------------------

## 7. FIXED EXAMPLES (diff-checked against November failures)

### 7.1 Windowed Quantized Self-Attention (1-D)
See `examples/020_windowed_quant_qk_attn.py`.  The diff versus the failed version:
* Replaced every `tl.cast` with `tl.astype`.
* Broadcasted `pid0` into vectors via `tl.full` before pointer arithmetic.
* Built `valid_win_vec` with vector comparisons – no mixing with python bools.
* Applied a 2-D flatten so only `program_id(0)` and `program_id(1)` are used.

### 7.2 Point-wise Conv2D + Residual + GLU
* Switched to **three-dimensional grid** `(N, H, triton.cdiv(W*C_out_half, BLOCK))`.
* The output channel and spatial indices are decoded from a single `pid2` (flattened) instead of using `program_id(3)`.
* All accumulators use `acc = acc + val`, never `+=`.

### 7.3 Masked LayerNorm + Gaussian Noise
* Affine `weight`/`bias` are 1-D; broadcast inside the kernel, not expanded on the host – fixes the numerical drift.

### 7.4 Median Pool 2-D + ReLU + Add
* Switched to **register-level sorting network** that returns a *new* tensor each stage – no slice writes.
* If `k > 32` the kernel drops into shared memory + bitonic sort instead of gigantic `[BLOCK,k]` register matrices.

### 7.5 Multi-resolution Affine Warp + Blend
* Bilinear sampler rewritten with vector masks only – scalar comparisons are lifted with `tl.full`.
* Uses `tl.atomic_add` to accumulate K grids into a single output in one launch (no K separate launches).

### 7.6 Multiplexed Shift-Gather + LayerNorm + Add
* Fixed the decorator typo, flattened the 3-D `(batch, seq_len, k)` space onto `program_id(0)`, and removed all `tl.cast`.

--------------------------------------------------------------------

## 8. VERSION NOTES
Triton 3.1 keeps the 4-arg `make_block_ptr`.  Triton 3.2 (ETA Q1 2026) will add `tl.tma_descriptor`.  Keep your own wrapper so kernels stay forward-compatible.

--------------------------------------------------------------------

*End of prompt block.*