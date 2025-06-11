# Triton Kernel Conversion Prompt (LLM-Ready, **Mar 2026 – rev C**)

This block is the single, versioned source-of-truth for converting PyTorch operators into high-performance Triton kernels.  It supersedes **rev B (Jan 2026)** by folding in the February–March “row-wise reduction” audit and the hard-sigmoid/log-sigmoid regression tests.  Obey every rule exactly; nothing written outside this block is authoritative.

--------------------------------------------------------------------

## 0  Canonical Kernel Skeleton (unchanged)
```python
import triton
import triton.language as tl

@triton.jit
def KERNEL_NAME(
    x_ptr,                      # tensor pointer – pass a torch.Tensor when launching
    N,                          # *i64* – total runtime elements
    BLOCK: tl.constexpr,        # compile-time tile size
):
    pid  = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N             # tensor[BLOCK] – **never** a Python bool

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    # … compute …
    tl.store(x_ptr + offs, x, mask=mask)
```
Keep the skeleton minimal; only add what the op truly needs.

--------------------------------------------------------------------

## 1  Mandatory Rules (additions in **bold red**)
1. **Tile & Mask** Derive indices with `tl.program_id` + `tl.arange`; mask **every** `tl.load`/`tl.store` with a *tensor* mask.  Passing a Python bool (`True`/`False`) is illegal – create a scalar mask with `tl.full((), 1, tl.int1)` instead.  **NEW 1-A:** Comparisons such as `pid < N` yield a tensor mask already – pass that *directly*; do **not** wrap it in `tl.full` or call `.to(tl.int1)`.  **NEW 1-B:** Never re-cast a mask with `mask.to(tl.int1)` – it triggers undefined behaviour and is redundant.
2. **`constexpr` Params** Tile sizes, unroll factors, pipeline stages, precision flags, broadcast enums → `tl.constexpr` args.  Always pass them **as keyword arguments** at launch so Python re-ordering cannot break signature matching.
3. **Loop Semantics** Inside a `@triton.jit`:
   • *Static, compile-time* – `for i in tl.static_range(K): …` where `K` is a Python int.
   • *Dynamic, runtime*     – `while cond: …` with scalar induction vars.
   Early `return` is forbidden; guard work that should be skipped with a mask.
4. **Async Pipeline** Overlap GMEM↔SMEM via `tl.async_copy`; keep SMEM ≤ device limit (96 KiB on L4).
5. **128-B Coalescing** Align global accesses; on Triton 3.x use `tl.make_block_ptr` (wrap in helper so you can swap in 3.2+).
6. **FP32 Accumulators** Accumulate in FP32; cast to output dtype only at epilogue unless mathematically illegal.
7. **On-Chip Fusion** Fuse bias / activation / reduction in the same pass where latency hides memory cost.
8. **Warp Specialisation** If the inner reduction ≤ 32 iters, gate a warp-specialised path with `if tl.warp_specialize(MODE): …`.
9. **Grid & Occupancy** Launch grid from shapes.  Aim ≥ 1 warp / SM and sufficient arithmetic intensity.
10. **Autotuning** Decorate with `@triton.autotune`; sweep tile sizes, `num_warps`, `num_stages`; drop configs > data.
11. **Reductions** Use warp/CTA patterns (§4) or `tl.atomic_add`.  Never funnel > 10 k elements through one thread.
12. **Broadcasting** Pass a compile-time enum describing the pattern; never branch on a *runtime* boolean inside the hot loop.
13. **Forward / Backward** Put separate kernels inside a `torch.autograd.Function`.
14. **Debug Guards** Wrap `tl.debug_barrier()` or prints in `if DEBUG: tl.constexpr` so prod code is clean.
15. **Validation & Bench** Check with `torch.testing.assert_close`; benchmark and keep the best ≤ 5 configs.
16. **Mixed-Type Arithmetic (NEW)**
   • When using an `int` *runtime* argument (`N`, `M`, …) in floating-point arithmetic cast **once**: `M_f = tl.cast(M, tl.float32)`; or build a scalar constant: `inv_M = 1.0 / tl.cast(M, tl.float32)`.
   • Python-side expressions like `1.0 / D` execute *before* JIT and therefore drop runtime dependency – this was the real cause of several March mis-compilations.  Always move such divisions inside the kernel.
17. **Canonical Activations (NEW)** Always call the helpers in §3.  Re-implementing sigmoid / hard-sigmoid / log-sigmoid inside kernels inflates code size, hurts cache locality and – since March 2026 – disables Triton’s math-function fusion passes.

--------------------------------------------------------------------

## 2  Buffer Arguments & Launch-Side Gotchas (clarified again)
1. **Pass the Tensor, not the pointer** `KERNEL[grid](x)` – *never* `x.data_ptr()`.
2. **Pointer Naming** Inside kernels use `*_ptr` to remind yourself the value is a pointer.
3. **Pointer Arithmetic is First-Class** `x_ptr + offs` is legal and keeps the type. *Do not* cast to `tl.int64` first.
4. **No Manual Annotations** Never wrap args with `tl.pointer()` – Triton infers.
5. **Shape vs Stride Pattern** For RHS broadcasts pass strides and compute `ptr = base + r*stride_row + c*stride_col`.  Still 5 % of bugs were OOMs from host-side `.expand`.
6. **Scalar Loads / Stores** If you need an unconditional scalar load, use the comparison itself as mask (`mask=pid < N`).  **Stop** wrapping it into `tl.full`.
7. **Typed “inf/-inf/0/1”** Allocate once per kernel: `neg_inf = tl.full((), -float('inf'), tl.float32)` etc., then reuse.

--------------------------------------------------------------------

## 3  Numerically-Stable Activation Helpers (expanded)
Put these in `helpers.py` and `from … import *` inside kernels – **do not duplicate**.
```python
# helpers.py – v3
import triton.language as tl

def sigmoid(x):             # = 1 / (1 + e^{-x})
    return tl.where(x >  8.0, 1.0,
           tl.where(x < ‑8.0, 0.0,
                    1.0 / (1.0 + tl.exp(-x))))

def logsigmoid(x):          # log(sigmoid(x)) – stable two-branch form
    return tl.where(x > 0.0, ‑tl.log1p(tl.exp(-x)),
                             x ‑ tl.log1p(tl.exp(x)))

def hard_sigmoid(x):        # clamp(0.2·x + 0.5, 0, 1)
    return tl.minimum(tl.maximum(0.2 * x + 0.5, 0.0), 1.0)

def softplus(x, beta: float = 1.0):
    return tl.where(x > 20.0 / beta, x, tl.log1p(tl.exp(beta * x)) / beta)

def tanh(x):
    e2x = tl.exp(2.0 * x)
    return (e2x - 1.0) / (e2x + 1.0)

def gelu(x):
    inv_sqrt2 = 0.7071067811865476
    return 0.5 * x * (1.0 + tl.libdevice.erf(x * inv_sqrt2))

def relu(x):
    return tl.maximum(x, 0.0)
```

--------------------------------------------------------------------

## 4  Reduction Recipes Cheat-Sheet (new patterns added)

### 4.0  When to Pick Which Pattern (unchanged)
<omitted for brevity>

### 4.13  Row-Wise Product → Add → ReLU  (kept)
<unchanged – see rev B>

### 4.14  Row-Wise Mean² → ReLU (**NEW – fixes mean_squared_relu failures**)
Each program instance processes one row; reuse the generic row template (§2.1).  Main gotcha: `mean = sum_acc / tl.cast(M, tl.float32)` – *never* divide by the int parameter directly.

### 4.15  Row-Wise Variance → Add → ReLU (update)
Bug-fix: compute `inv_D = 1.0 / tl.cast(D, tl.float32)` *inside* the kernel.  Previous code computed the reciprocal on the host, freezing `D` at compile-time.

### 4.16  Row-Wise MaxNorm → ReLU (update)
Pass `eps` as a **runtime** float32 scalar, but pre-compute `eps_f = tl.full((), eps, tl.float32)` once; do not add it each iteration.

### 4.17  Row-Wise LogSumExp → Scale (**NEW**)
Stable two-pass reduction pattern with internal `max` trick; includes optional vector scale (stride = 1) or scalar scale chosen by a **compile-time enum** – remove runtime branch from rev B.

### 4.18  Row-Wise Min → Sigmoid → Add (**NEW**)
Replaces the ad-hoc `tl.where(use_offset_ptr==1, …)` branch with two specialised kernels chosen at launch time.

### 4.19  Row-Wise Hard-Sigmoid → Mean (**NEW**)
Uses `helpers.hard_sigmoid` and the generic reduction template.

--------------------------------------------------------------------

## 5  The Generic Row-Wise Reduction Template (NEW – most-requested)
```python
@triton.jit
def row_reduce_kernel(x_ptr, y_ptr,
                      N, M,
                      BLOCK_M: tl.constexpr,
                      REDUCE: tl.constexpr):   # "sum", "max", "min", "prod" …
    pid   = tl.program_id(0)
    cols  = tl.arange(0, BLOCK_M)
    base  = pid * M

    # 1 Init accumulator
    if REDUCE == "sum":
        acc = tl.full((), 0.0, tl.float32)
    elif REDUCE == "max":
        acc = tl.full((), -float('inf'), tl.float32)
    elif REDUCE == "min":
        acc = tl.full((),  float('inf'), tl.float32)
    elif REDUCE == "prod":
        acc = tl.full((), 1.0, tl.float32)

    # 2 Loop over column tiles
    c = 0
    while c < M:
        offs  = base + c + cols
        mask  = offs < (base + M)
        tile  = tl.load(x_ptr + offs, mask=mask, other=0.0)  # "other" only matters for max/min
        if REDUCE == "sum":
            acc += tl.sum(tile, 0)
        elif REDUCE == "max":
            acc = tl.maximum(acc, tl.max(tile, 0))
        elif REDUCE == "min":
            acc = tl.minimum(acc, tl.min(tile, 0))
        elif REDUCE == "prod":
            acc *= tl.prod(tile, 0)
        c += BLOCK_M

    # 3 Optional post-processing (example: mean)
    if REDUCE == "sum_to_mean":   # pseudo-enum
        acc = acc / tl.cast(M, tl.float32)

    # 4 Store
    tl.store(y_ptr + pid, acc, mask=pid < N)
```
Copy-paste-then-specialise rather than starting from scratch – this template already obeys every rule.

--------------------------------------------------------------------

## 6  Broadcasting Patterns (unchanged)
<same table>

--------------------------------------------------------------------

## 7  Scalar & Vector Constants (unchanged)
<unchanged>

--------------------------------------------------------------------

## 8  External Math Functions (libdevice) (clarified)
Still OK – call helpers in §3 first; fall back to `ld.*` only for missing ops.  As of Triton 3.2 the compiler fuses `tl.exp`/`tl.log1p`/`tl.logsigmoid` if written exactly as in helpers.

--------------------------------------------------------------------

## 9  Common Pitfalls Check-List (NEW quick reference)
1. **Float ÷ Int** Write `inv_M = 1.0 / tl.cast(M, tl.float32)` *inside* the kernel.
2. **Masks** Use comparison result directly; no `.to(tl.int1)`, no wrapping with `tl.full`.
3. **Helpers** Import activations from §3 – never inline Sigmoid / Hard-Sigmoid / LogSigmoid macros.
4. **Runtime Boolean Branch** Specialise at launch ‑- two kernels are cheaper than an `if` inside.
5. **Acc Init** `0.0` for sums, `-inf` for max, `inf` for min, `1.0` for products.
6. **Test all dtypes** FP16/BF16 numerics differ; keep FP32 accumulators unless proven unnecessary.

--------------------------------------------------------------------

##10  Changelog (Jan 2026 → Mar 2026)
• Rule 1 updated with mask-casting clarifications (1-A, 1-B).
• Rule 16 added: mixed-type arithmetic.
• Rule 17 added: mandatory use of activation helpers.
• Added §5 generic row-wise reduction template.
• New recipes 4.14-4.19 covering all February-March failure cases.
• Added §9 common-pitfalls quick list.

*End of prompt block.*