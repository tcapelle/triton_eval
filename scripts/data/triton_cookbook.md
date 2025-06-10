# Triton Kernel Conversion Prompt – COMPLETE EDITION
(LLM-Ready, **REV-AUG 2027.a**) – adds rules 76-80 and new Pooling mini-guide drafted from recent failure cases; clarifies pointer arithmetic for strided windows, broadcast bias add, and dynamic-size safeguards.  This document *completely* replaces every earlier edition, including JULY 2027.m.

--------------------------------------------------------------------

## 0. GOLDEN RULES – quick checklist before you hit “Run”
(NEW or changed since JULY 2027.m are ★★★-marked, **bold** where vital)

✓  **`tl.cast` is gone → use `tl.astype(x, tl.int32/float32)` for *all* scalar *and* vector casts.** (★ reminder)
✓  **All reduction intrinsics (`tl.sum`, `tl.max`, `tl.min`, …) take the *axis index as a positional-only argument*.** `tl.sum(x, 0)` is legal, `tl.sum(x, axis=0)` is *not*.
✓  Python numeric literals (`1`, `1.0`, etc.) default to *float64* in Triton.  Prefer `tl.full([], 1, dtype)` or `tl.asarray(1, dtype)`.
✓  ★★★ **Triton has no `view`/`reshape`.  If you need `x.view(N, D // W, W)` you must compute pointer math manually.**  (new)
✓  ★★★ **Pooling or windowed reductions must materialise the window with `tl.arange`-based offsets, not Python loops or `tl.broadcast_to`.  See §7b.**
✓  ★★★ **Broadcast add/mul uses standard arithmetic; *one dim must be length-1* in the active lanes.  Pre-compute a bias pointer and stride with `tl.arange` instead of fancy indexing.**

--------------------------------------------------------------------

## 1. KERNEL SKELETON — start here (unchanged except safer literals)
```python
import triton
import triton.language as tl

@triton.jit
def KERNEL_NAME(
    x_ptr,                # tensor, *not* pointer
    N,                    # rows
    D,                    # cols (optional if constexpr)
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_n  = tl.program_id(0)              # row index
    pid_d  = tl.program_id(1)              # tile along last-dim, ▢ optional

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    mask_n = offs_n < N
    mask_d = offs_d < D

    ptrs = x_ptr + offs_n[:, None] * D + offs_d[None, :]   # 2-D pointer grid
    ZEROS = tl.full([BLOCK_N, BLOCK_D], 0, tl.float32)
    x = tl.load(ptrs, mask=mask_n[:, None] & mask_d[None, :], other=ZEROS)

    # … do work …

    tl.store(ptrs, x, mask=mask_n[:, None] & mask_d[None, :])
```
Key take-away:  Reshape is emulated by mixed-rank pointer math.

--------------------------------------------------------------------

## 2. MANDATORY RULES (apply in this exact order)
…rules 17-67 unchanged…

68. **`tl.cast` has been removed.**  *Every* cast – tensor or scalar – now uses `tl.astype(x, dtype)`.

69-72 unchanged.

73. **Reductions are positional-only.**  Write `tl.sum(x, 0)` – *never* `tl.sum(x, axis=0)`.

74. **Typed literals only.**  Bare Python numerics are float64.  Always materialise a literal with `tl.full([], val, dtype)` or reuse an existing variable’s dtype.

75. **Avoid silent fp64→fp32 promotion.**  Keep everything consistently typed.

76. **No `.view` / `.reshape`.**  Any logical reshape must be reproduced through linear-index math (`row * stride + col`).  Attempting to port `x.view(N, new_D, W)` from PyTorch is the #1 cause of “placeholder kernels”.

77. **Fixed-window pooling requires compile-time window size** (`WINDOW: tl.constexpr`) *or* a fallback loop with `tl.arange` sized to the *maximum* supported window and masked lanes for smaller cases.

78. **Window offsets are always contiguous.**  Create them with `base + tl.arange(0, W)`, **never** via dynamic indexing of a global pointer list.

79. **Mean-pool = sum / window_size (typed).**  Use `inv_W = tl.full([], 1.0 / WINDOW, x.dtype)` then `out = tl.sum(win, 0) * inv_W`.  Do *not* rely on integer division.

80. **Broadcast bias addition.**  For a pooled output of shape `(N, new_D)` and a bias of `(new_D,)`, compute `bias_ptr = bias + offs_d` and load once per output tile.  Broadcasting by constructing a `[BLOCK_N, BLOCK_D]` bias tensor trom inside the kernel wastes memory bandwidth.

--------------------------------------------------------------------

## 5. QUICK REFERENCE – one-liners worth memorising
• `ZERO  = tl.full([], 0, x.dtype)` – typed zero scalar.
• `ONE   = tl.full([], 1, x.dtype)` – typed one scalar.
• `int32_val = tl.astype(val, tl.int32)` – legal scalar cast.
• `sum0 = tl.sum(x, 0)` – positional-only reduction.
• ★ `row_ptrs = base + offs_n * STRIDE` – canonical strided access.
• ★ `inv_W = tl.full([], 1.0 / WINDOW, x.dtype)` – typed reciprocal for mean.

--------------------------------------------------------------------

## 7. MINI-GUIDES

### 7a. “Why did my reduction blow up?”  (unchanged from prior rev)
…

### 7b. “How do I do mean-pool over fixed windows?”  (NEW)
Scenario: you have `(N, D)` and want `(N, D // W)` where `W` is a small compile-time window.

1. Declare `WINDOW: tl.constexpr` arg.  Run-time windows destroy vectorisation.
2. Tile rows with `BLOCK_N`, pooled columns with `BLOCK_P = 128` (or whatever).
3. Compute base column indices *in pooled space*: `offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)`.
4. Recover *input* column indices:  `offs_in = offs_p[:, None] * WINDOW + tl.arange(0, WINDOW)[None, :]`  ➜ shape `[BLOCK_P, WINDOW]`.
5. Build 2-D pointer: `ptrs = x_ptr + offs_n[:, None, None] * D + offs_in[None, :, :]`.
6. Load window `win = tl.load(ptrs, …)`; reduce `sums = tl.sum(win, -1)`.
7. Multiply by `inv_W` (typed reciprocal) to get the mean.
8. Add broadcast bias: `bias_vals = tl.load(bias_ptr + offs_p, mask=mask_p)`.
9. Store.

Pitfalls caught in recent failures:
• Forgetting the `[:, None]` & `[None, :]` rank alignment, leading to pointer-arithmetic shape errors.
• Using Python `for` over `WINDOW`; use vector arithmetic instead.
• Dividing by `WINDOW` (an *int*) instead of pre-multiplying by `1.0 / WINDOW` with correct dtype.

--------------------------------------------------------------------

## 8. VERSION NOTES
Revision **AUG 2027.a** introduces rules 76-80 and §7b after analysing multiple recent auto-conversions that failed whenever PyTorch code used `.view` + `.mean(dim=-1)` for pooling.  Guidance now shows the exact pointer arithmetic pattern and emphasises typed reciprocal for averages.

--------------------------------------------------------------------
*End of prompt block.*