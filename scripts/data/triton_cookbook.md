# Triton Kernel Conversion Prompt (LLM‑Ready, June 2025)

**Context block for an LLM that must rewrite a PyTorch op into a high‑performance Triton kernel. Follow every rule exactly.**

---

## Kernel Skeleton — start here

```python
import triton.language as tl

@triton.jit
def KERNEL_NAME(
    x_ptr,                # *T*  – tensor pointer(s)
    N,                    # *i64* – total elements
    BLOCK_SIZE: tl.constexpr,
):
    pid  = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    data = tl.load(x_ptr + offs, mask=mask)
    # TODO: compute
    tl.store(x_ptr + offs, data, mask=mask)
```

---

## Mandatory Rules (obey in order)

1. **Tile & Mask**  Compute indices with `tl.program_id` + `tl.arange`; mask **every** `tl.load`/`tl.store`.
2. **constexpr Params**  Expose *all* tile sizes, unroll factors, stage counts, precision flags via `tl.constexpr` args.
3. **Async Pipeline**  Overlap GMEM→SMEM with compute via `tl.async_copy` (or `tl.tma_async_copy` on Hopper). Pick `num_stages` so `shared_mem ≤ 96 KiB` (L4) or device limit.
4. **128‑B Coalescing**  Keep global loads/stores 128‑byte aligned; on Triton 3.0 use `tl.make_block_ptr`, and be aware that upcoming 3.1+ may switch to `tl.make_tensor_descriptor`/TMA descriptors — write a thin helper so you can swap APIs easily. ([github.com](https://github.com/pytorch/pytorch/issues/154025?utm_source=chatgpt.com))
5. **FP32 Accumulators**  Accumulate in FP32, cast to output dtype only at the epilogue.
6. **On‑Chip Fusion**  Fuse bias/activation/reduction in the same pass when possible.
7. **Warp Specialization**  If the inner loop fits ≤ 32 iterations, guard a warp‑specialized path with `if tl.warp_specialize(WARP_MODE): …`.
8. **Grid & Occupancy**  Derive launch grid from input shapes. Query GPU props (NUM\_SMS, SMEM\_MAX, REGS\_MAX, WARPS\_MAX) and ensure occupancy ≥ 1 warp/SM.
9. **Autotuning**  Wrap the kernel with `@triton.autotune`, sweeping over tile sizes, `num_stages`, and `num_warps`; prune configs larger than the data.
10. **Reductions**  Use `tl.atomic_add` (FP32 supported on Hopper+) or a two‑phase reduction (pre‑Hopper).
11. **Forward / Backward**  Implement separate Triton kernels inside a `torch.autograd.Function`.
12. **Debug Guards**  Wrap `tl.debug_barrier()` or prints inside `if DEBUG: tl.constexpr` to strip them from prod builds.
13. **Validation & Bench**  Verify with `torch.testing.assert_close`; benchmark and keep the best ≤ 5 configs.

---

## Buffer Arguments & `tl.constexpr` Meta‑Parameters

**Directive to LLM**  For every kernel you generate:

1. **Buffer Arguments**  Pass tensor buffers **un‑annotated** (e.g., `x_ptr`, `y_ptr`). Triton infers they are device pointers; no `tl.pointer(...)` annotation exists in the Python API.
2. **Compile‑time Meta‑Parameters**  Mark tile sizes, unroll factors, stage counts, precision flags, and other design‑time constants as `tl.constexpr`. Do **not** mark runtime sizes (`M`, `N`, batch length) as `constexpr`.
3. \`\`\*\* Base Argument\*\*  Supply the **raw device pointer** (`tensor.data_ptr()`) as the first argument—never a `torch.Tensor`, dtype object, or literal integer offset.
4. **Pointer Safety**  If a pointer is produced inside Triton (e.g., shared‑memory alias), keep it as a pointer type; do not cast to `int` before reuse.

### Example

```python
@triton.jit
def fp16_relu(
    x_ptr,  # tensor pointer (no annotation)
    y_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid  = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.maximum(x, 0.0)
    tl.store(y_ptr + offs, y, mask=mask)
```

---

## Anti‑Patterns (✗ never do)

- Unmasked out‑of‑bounds memory ops
- Excessive register pressure from huge tiles / unrolls
- Increasing `num_stages` beyond SMEM limit
- Hard‑coding grid sizes or launch bounds
- Using deprecated APIs (`make_tensor_descriptor`); use `make_block_ptr`
- Shipping kernels with `tl.debug_barrier()` or print statements

---

### Conversion Workflow for the LLM

1. **Insert the skeleton** and rename `KERNEL_NAME`.
2. **Map PyTorch tensor shapes** → choose initial tile sizes (64‑128 range).
3. **Apply Rules 1‑5** to flesh out the kernel body.
4. **Add async pipeline** and `num_stages` knob (Rule 3).
5. **Create an autotune sweep** (Rule 9).
6. **Wrap** kernel(s) in a `torch.autograd.Function` (Rule 11).
7. **Return**: Triton kernel(s) + autotuner wrapper + PyTorch glue.

## Libdevice & External Math Functions

**Directive to LLM**  Use `tl.extra.libdevice` whenever a kernel needs a transcendental math function (e.g., `asin`, `acos`, `erf`) that Triton does not provide natively.

### Usage Rules

1. `from triton.language.extra import libdevice` inside your Python module.
2. Call the function inside the kernel (`libdevice.asin`, `libdevice.erf`, …); Triton auto‑routes to the correct bit‑code implementation based on the tensor dtype.
3. The default NVIDIA/AMD bit‑code paths are shipped with Triton and are resolved automatically—no user action needed.
4. Functions are aggregated by semantics, not dtype (`__nv_asin` + `__nv_asinf` → `libdevice.asin`).

### Kernel Example — `asin`

```python
@triton.jit
def asin_kernel(
    x_ptr, y_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid   = tl.program_id(0)
    offs  = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask  = offs < N

    x = tl.load(x_ptr + offs, mask=mask)
    y = libdevice.asin(x)                 # external math call
    tl.store(y_ptr + offs, y, mask=mask)
```



**Validation**   Compare against `torch.asin` using `torch.testing.assert_close` (see Rule 13).

---

### Conversion Workflow for the LLM

1. **Insert the skeleton** and rename `KERNEL_NAME`.
2. **Map PyTorch tensor shapes** → choose initial tile sizes (64‑128 range).
3. **Apply Rules 1‑5** to flesh out the kernel body.
4. **Add async pipeline** and `num_stages` knob (Rule 3).
5. **Create an autotune sweep** (Rule 9).
6. **Wrap** kernel(s) in a `torch.autograd.Function` (Rule 11).
7. \*\*If an external math op is needed, integrate \*\*\`\` per this section.
8. **Return**: Triton kernel(s) + autotuner wrapper + PyTorch glue.

*End of prompt block.*

