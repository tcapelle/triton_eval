# Triton Kernel Conversion Recipe (v2)

A concise, actionable checklist and code snippets to guide you through writing high-performance Triton kernels. Focuses on kernel-authoring steps themselves, without extraneous benchmarking details.

---

## 1. Define Your SPMD Tile (1D)

**Goal**: Process a contiguous tile of data per program instance.

```python
import triton.language as tl

@triton.jit
def kernel_1d(x_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid   = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offs  = start + tl.arange(0, BLOCK_SIZE)
    mask  = offs < N
    data  = tl.load(x_ptr + offs, mask=mask)
    # ... compute ...
    tl.store(x_ptr + offs, data, mask=mask)
```

* Use `tl.program_id(axis)` and `tl.arange` for per-program indexing.
* Mark tile dimensions as `constexpr` for compile-time unrolling and optimization.

---

## 2. Multi-Dimensional Grids (2D/3D)

**Goal**: Map 2D/3D problem domains to Triton grids.

```python
@triton.jit
def kernel_2d(A_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    start_m = pid_m * BLOCK_M
    start_n = pid_n * BLOCK_N
    off_m = start_m + tl.arange(0, BLOCK_M)
    off_n = start_n + tl.arange(0, BLOCK_N)
    # broadcast to 2D
    idx = off_m[:, None] * N + off_n[None, :]
    mask = (off_m[:, None] < M) & (off_n[None, :] < N)
    block = tl.load(A_ptr + idx, mask=mask)
    # ... compute ...
    tl.store(A_ptr + idx, block, mask=mask)
```

* Compute `num_pid_m = tl.cdiv(M, BLOCK_M)` and similarly for `N` to launch 2D grid.
* For 3D, use `program_id(2)`.

---

## 3. Safety with Masks

**Goal**: Prevent out-of-bounds memory accesses on partial tiles.

```python
offs = start + tl.arange(0, BLOCK)
mask = offs < length
x    = tl.load(x_ptr + offs, mask=mask)
# ... compute ...
tl.store(y_ptr + offs, result, mask=mask)
```

* Guard **every** `load`, `store`, and reduction (e.g., `tl.sum(..., mask=mask)`).

---

## 4. Vectorization & Memory Coalescing

**Goal**: Optimize memory bandwidth via unit-stride loads and shared-memory reuse.

```python
# Use make_block_ptr for non-contiguous strides:
block_ptr = tl.make_block_ptr(
    base_ptr=x_ptr,
    shape=(BLOCK_M, BLOCK_N),
    strides=(stride_m, stride_n),
    offsets=(start_m, start_n),
)
# load a tile in one call:
A = tl.load(block_ptr, mask=mask)
```

* Aim for unit-stride (contiguous) accesses whenever possible.
* Use `tl.cdiv(n, BLOCK)` to compute tile counts.
* On Ampere+ GPUs, consider `cp.async` for async shared-memory prefetch.

---

## 5. Meta-Parameter Specialization

**Goal**: Expose tunable constants so Triton can compile optimized variants.

```python
@triton.jit
def gemm(A, B, C, M, N, K,
           BM: tl.constexpr,
           BN: tl.constexpr,
           BK: tl.constexpr,
           UNROLL: tl.constexpr):
    # use BM,BN,BK,UNROLL in loops and indexing
    ...
```

* Promote tile sizes, unroll factors, warp counts, and flags to `constexpr`.
* Enables `@triton.autotune` to search over them.

---

## 6. Autotune Configuration

**Goal**: Define a search space over meta-parameters and let Triton pick the best.

```python
def make_configs():
    configs = []
    for BM in [64,128]:
      for BN in [64,256]:
        for BK in [32,64]:
          configs.append(
            triton.Config({'BLOCK_M':BM,'BLOCK_N':BN,'BLOCK_K':BK},
                          num_stages=4, num_warps=8)
          )
    return configs

@triton.autotune(
  configs=make_configs(),
  key=['M','N','K'],
)
@triton.jit
def gemm(...):
    ...
```

* Prune configs where tile dims exceed problem dims.
* Use `@triton.autotune` for runtime specialization.

---

## 7. On-Chip Fusion & Mixed Precision

**Goal**: Fuse multiple operations in one pass; use higher precision for accumulators.

```python
# 1) Load tiles
A = tl.load(A_ptrs, mask=mask_a)
B = tl.load(B_ptrs, mask=mask_b)
# 2) Compute in FP32
acc = tl.dot(A, B, tl.zeros((BM,BN), dtype=tl.float32))
# 3) Epilogue: bias, activation, cast
C = acc + bias
C = tl.relu(C)
out = C.to(output_dtype)
tl.store(C_ptrs, out, mask=mask_c)
```

* Cast only at store; use `FP8` or `FP16` flags via `constexpr`.
* Fuse bias-add, activation, reduction in one kernel where possible.

---

## 8. Software Pipelining & Persistence

**Goal**: Overlap memory and compute; use fewer programs than tiles.

```python
@triton.jit
def persistent(..., NUM_SMS: tl.constexpr):
    tid = tl.program_id(0)
    total = num_pid_m * num_pid_n
    for t in tl.range(tid, total, NUM_SMS, flatten=True):
        pid_m, pid_n = divmod(t, num_pid_n)
        # load, compute, store
```

* Launch grid size = `min(NUM_SMS, total_tiles)` for one program per SM.
* Use `tl.range(..., flatten=True)` to assign multiple tiles per program.

---

## 9. Two-Phase Parallel Reductions

**Goal**: Efficiently reduce across rows or program instances.

1. **Partial Reduction** with shared buffers & `atomic_cas`:

   ```python
   # each program writes partial sums
   while tl.atomic_cas(Lock, 0, 1): pass
   tl.store(buf + idx, partial, mask=mask)
   tl.atomic_xchg(Lock, 0)
   ```
2. **Final Reduction** kernel scans buffers:

   ```python
   @triton.jit
   ```

def finalize(DW, FINAL, M, N, BM: tl.constexpr, BN: tl.constexpr):
pid = tl.program\_id(0)
cols = pid\*BN + tl.arange(0,BN)
acc = tl.zeros(\[BN], tl.float32)
for i in range(0, M, BM):
rows = i + tl.arange(0,BM)
idx  = rows\[:,None]\*N + cols\[None,:]
m    = (rows\[:,None]\<M)&(cols\[None,:]\<N)
acc += tl.load(DW+idx, mask=m)
out = tl.sum(acc, axis=0)
tl.store(FINAL+cols, out, mask=cols\<N)

````

- Choose buffer group size to fit L2 and minimize contention.

---

## 10. Descriptor API & Stride Handling

**Goal**: Support arbitrary N-dimensional layouts (e.g. NHWC, NCHWc).

```python
# Host side
desc = torch.tensor_strides(x)
# Device side
ptr = tl.make_tensor_descriptor(
 base_ptr=x_ptr,
 shape=desc.shape,
 strides=desc.strides,
 order=desc.order,
)
A = tl.load(ptr, offsets=(i,j), mask=...)
````

* Avoid manual offset arithmetic when handling complex strides.

---

## 11. Libdevice & External Math Functions

**Goal**: Invoke library math routines (sin, cos, asin, etc.) via `tl.extra.libdevice`.

Triton can call external device functions—wrapped in `tl.extra.libdevice`—to compute transcendental operations. Triton automatically picks the correct double/float variant.

```python
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

DEVICE = "cuda"

@triton.jit
def asin_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid         = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets     = block_start + tl.arange(0, BLOCK_SIZE)
    mask        = offsets < n_elements
    x           = tl.load(x_ptr + offsets, mask=mask)
    x           = libdevice.asin(x)  # compute arc-sine
    tl.store(y_ptr + offsets, x, mask=mask)
```

* In `libdevice.py`, functions like `__nv_asin` (double) and `__nv_asinf` (float) are grouped. Triton picks the right one based on input dtype.
* For full list and semantics, refer to CUDA Libdevice User Guide or HIP device-lib source.
* Other examples: `libdevice.sin(x)`, `libdevice.exp(x)`, `libdevice.sqrt(x)`.

---

## 12. Debugging & Testing

**Goal**: Verify correctness before performance tuning.

* **Print**: `tl.debug_print(x, y, "msg")` inside kernels.
* **Barrier**: `tl.debug_barrier()` to synchronize and inspect.
* **Host Assertions**: compare kernel output against PyTorch reference for small inputs.
* **Unit Tests**: use `pytest` with edge-case shapes (odd sizes, N < BLOCK).

---

## 13. Profiling & Benchmarking

**Goal**: Measure performance and identify bottlenecks.

* **Triton Profiler**: `triton.testing.perf_report` or `triton.profiler`.
* **External Tools**: NVIDIA Nsight Compute for detailed metrics.
* **Benchmark Protocol**: warm-up runs, multiple iterations, report GFLOPS/GB/s vs. roofline.

---

## 14. Device-Aware Resource Queries & Occupancy

**Goal**: Tune grid, stages, and warps to GPU limits.

```python
from triton.runtime.driver import active
props = active.utils.get_device_properties(DEVICE.index)
NUM_SMS = props['multiprocessor_count']
SMEM    = props['max_shared_mem']
REGS    = props['max_num_regs']
```

* Compute max programs per SM:

  ```python
  max_progs = min(NUM_SMS,
                  SMEM // smem_per_stage,
                  REGS // regs_per_program)
  ```
* Choose `num_stages` and `num_warps` to balance occupancy vs. register/SMEM usage.

---

## 15. Dynamic Shapes & Conditional Dispatch

**Goal**: Handle variable workloads efficiently.

* Branch in Python wrapper to dispatch specialized kernels:

  ```python
  if N % 128 == 0:
      kernel_128[grid](...)
  else:
      kernel_generic[grid](...)
  ```
* Or write a single kernel with shape-dependent branches on `constexpr` flags.

---

## 16. Forward & Backward Separation

**Goal**: Keep backward kernels focused and low-pressure.

```python
class MyOp(torch.autograd.Function):
  @staticmethod
def forward(ctx, x):
    y = kernel_fwd[x.shape, ...](x, ...)
    ctx.save_for_backward(x, y)
    return y

  @staticmethod
def backward(ctx, dy):
    x, y = ctx.saved_tensors
    dx   = kernel_bwd[dy.shape, ...](x, y, dy)
    return dx
```

* Save only minimal state.
* Separate Triton kernels for each gradient path.

---

## Final Checklist & Anti-Patterns

1. Prototype small tile → add masks → parametrize → autotune.
2. Verify with unit tests and profiler.
3. Tune occupancy via device props.
4. Separate ops and use mixed precision.
5. Integrate libdevice calls for math functions.
6. Dispatch dynamically for varied shapes.

**Anti-Patterns**

* Unmasked OOB accesses
* Excessive register pressure
* Unnecessary barriers
* Hard-coded grid sizes
* Ignoring device limits
* Over-specializing early
* Monolithic kernels without modularity

> Keep this recipe as your go-to reference for authoring Triton kernels. Follow the checklist, start from the small prototype, and use autotuning plus profiling to reach top performance.
