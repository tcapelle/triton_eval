# Triton Kernel Conversion Prompt
(LLM-Ready, **REV-JULY 2027.m**) – adds rules 73-75 clarifying reduction-API positional arguments, constant-literal typing, and numeric-literal promotion; minor Golden-Rule and Quick-Reference expansions.

This document *completely* replaces every earlier edition. Anything not shown here is **obsolete**.

--------------------------------------------------------------------

## 0. GOLDEN RULES – **quick checklist before you hit “Run”**
(NEW or changed since June 2027.k+1 are ★★★-marked, **bold** where vital)

…(previous bullets unchanged)…
✓  **`tl.cast` is gone → use `tl.astype(x, tl.int32/float32)` for *all* scalar *and* vector casts.**  Typical pattern for runtime tensor dimensions: `channels_i = tl.astype(channels, tl.int32)` (★ example added) ★★★
✓  **All reduction intrinsics (`tl.sum`, `tl.max`, `tl.min`, …) take the *axis index as a positional-only argument*.**  `tl.sum(x, 0)` is legal, `tl.sum(x, axis=0)` is *not*.  (★ new) ★★★
✓  Python numeric literals (`1`, `1.0`, etc.) default to *float64* in Triton.  Prefer `tl.full([], 1, dtype)` or `tl.asarray(1, dtype)` to avoid silent fp64→fp32 down-cast penalties. (★ new) ★★★
…(rest unchanged)…

--------------------------------------------------------------------

## 1. KERNEL SKELETON — start here *(unchanged except for safer literals)*
```python
import triton
import triton.language as tl

@triton.jit
def KERNEL_NAME(
    x_ptr,                # tensor, *not* pointer
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid  = tl.program_id(0)              # 0 ≤ pid < grid[0]
    offs = tl.arange(0, BLOCK_SIZE)      # thread-lane ids (≠ program_id)
    idx  = pid * BLOCK_SIZE + offs       # *** in-bounds pattern ***
    mask = idx < N

    SAFE_IDX = tl.where(mask, idx, 0)    # keeps address 0 in masked lanes
    ZERO = tl.full([], 0, tl.float32)    # safe literal – DON’T write 0.0
    data = tl.load(x_ptr + SAFE_IDX, mask=mask, other=ZERO)

    # TODO: compute – use only tl.math.* intrinsics or tl.where or `/` for division

    tl.store(x_ptr + SAFE_IDX, data, mask=mask)
```

…(sections 2 → 4 identical except where noted)…

--------------------------------------------------------------------

## 2. MANDATORY RULES (apply in this exact order)
…(rules 17-67 unchanged)…

68. **`tl.cast` has been removed.**  *Every* cast – tensor or scalar – now uses `tl.astype(x, dtype)`.

69-72 …(unchanged)…

73. **Reductions are positional-only.**  Write `tl.sum(x, 0)` – *never* `tl.sum(x, axis=0)`.  Same for `tl.max`, `tl.min`, `tl.dot`, etc.  Violating this throws “unexpected keyword argument ‘axis’”.

74. **Typed literals only.**  Bare Python numerics are float64.  Always materialise a literal with `tl.full([], val, dtype)` or reuse an existing variable’s dtype:  
   • `ONE = tl.full([], 1, x.dtype)`  
   • `INV_GROUP = tl.full([], 1.0/group_size, tl.float32)`

75. **Avoid silent fp64→fp32 promotion.**  Mixing a float64 literal with fp32 tensors forces a down-convert and may disable certain HW paths.  Keep everything consistently typed.

--------------------------------------------------------------------

## 5. QUICK REFERENCE – one-liners worth memorising *(new lines ★★★)*

• `ZERO  = tl.full([], 0, x.dtype)` – typed zero scalar. ★★★  
• `ONE   = tl.full([], 1, x.dtype)` – typed one scalar. ★★★  
• `int32_val = tl.astype(val, tl.int32)` – legal way to cast runtime scalars; never use `tl.cast`. ★★★ NEW ★★★  
• `sum0 = tl.sum(x, 0)` – **positional-only reduction; drop the keyword!** ★★★ NEW ★★★  
• `SAFE = tl.where(idx < N, idx, 0)` – clamp pointer address for masked lanes.
…(rest unchanged)…

--------------------------------------------------------------------

## 7. MINI-GUIDE – “Why did my reduction blow up?” *(NEW section)*
1. Make sure `axis` is passed positionally.  `tl.sum(a, 0)` compiles; `tl.sum(a, axis=0)` does not.  
2. `tl.sum` returns the same dtype as `a`.  If you later divide by an *int* you will get integer division!  Cast or use typed literal `tl.full([], denom, a.dtype)`.
3. Remember that the reduction happens *within the current program*, not across the grid.  Design your grid/block so that each lane sees the entire reduction domain.

--------------------------------------------------------------------

## 8. VERSION NOTES (incremental)
Revision **JULY 2027.m** adds rules 73-75 after observing widespread compile failures involving positional-only reduction signatures and fp64 literal leakage.  Also notes the correct idiom for typed constants.

--------------------------------------------------------------------
*End of prompt block.*