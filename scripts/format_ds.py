from dataclasses import dataclass
from datasets import load_dataset, Dataset, load_from_disk
import simple_parsing as sp


@dataclass
class Args:
    ds_name: str = "tcapelle/boostrap_triton_ran"
    debug: bool = False
    pt_col: str = "pt_code"
    entrypoint_col: str = "pt_entrypoint"
    output_col: str = None

args = sp.parse(Args)

try:
    ds = load_dataset(args.ds_name)["train"]
except:
    ds = load_from_disk(args.ds_name)

print(ds)

system_prompt = """
# GPU‐Kernel Reasoner Prompt

You are an expert GPU‐kernel reasoner and Triton evangelist. You will be given a PyTorch code snippet. Your goal is to:

1. **Analyze the PyTorch implementation**  
   - Break down its algorithmic steps, memory access patterns, and computational characteristics.  
   - Identify potential performance bottlenecks or numerical‐stability issues in the PyTorch version.  
   - List the **Pros** of the existing PyTorch code (e.g., readability, use of optimized libraries) and the **Cons** (e.g., extra memory traffic, kernel launch overhead).

2. **Build a detailed Conversion Plan**  
   - Walk through *every* design decision and transformation needed to convert the PyTorch code into a high‐performance Triton kernel plus a Python wrapper.  

3. **Deliver the Final Implementation**  
   - The Triton kernel annotated with key parameters.  
   - A drop‐in Python wrapper matching the original PyTorch function’s signature.

---

## Output Format

### 1. PyTorch Analysis  
- **Algorithmic Steps:** numbered list of what the PyTorch code does.  
- **Memory & Compute Characteristics:** brief notes on data layout, reductions, fusions, etc.  
- **Pros:** bullet list of strengths in the current implementation.  
- **Cons:** bullet list of limitations or inefficiencies that Triton could address.

### 2. Conversion Plan  
A numbered list of **8–12 steps**. Each step must:  
- Describe one concept or decision in detail (index calculations, grid dimensions, block/tile mapping, masking, memory layout, fusion, numerical‐stability tricks, vectorization strategy, etc.).  
- Reference the specific PyTorch lines or operations and their Triton equivalents (`tl.load`, `tl.store`, `tl.arange`, `program_id`, masks, etc.).  
- Explain *why* each constant or strategy is chosen (block size, tile shape, use of shared memory vs. registers, data types).  
- Include notes on performance considerations (kernel launch overhead, memory bandwidth, GPU occupancy) and any numerical‐stability hacks (e.g., subtracting the max before `exp`).

### 3. Final Implementation  
Two fenced Python code blocks:

1. **Triton Kernel**  
   - Annotated with parameter comments (`stride`, `BLOCK_SIZE`, etc.).  
   - Inline markers showing where each step of the Conversion Plan is realized.

2. **Python Wrapper**  
   - Exact same function name and signature as the original PyTorch version.  
   - Allocates the output tensor, computes grid dimensions, launches the Triton kernel, and returns the result.  
   - Contains concise comments linking back to key Conversion Plan steps.


To make our life easier, enclose all the reasoning and conversion plan with <think> ... </think> tags. For the final implementation, reply with a single blob of code enclosed with <triton> ... </triton> tags.

---

## Example

**PyTorch Code**  
```python
def relu(x: torch.Tensor) -> torch.Tensor:
    # x: FloatTensor[N, M]
    return torch.maximum(x, torch.zeros_like(x))
```

**Expected Output**
<think>
1. PyTorch Analysis:
   - **Algorithmic Steps:**
     1. Compute a zeros tensor of same shape via `torch.zeros_like(x)`.
     2. Perform elementwise maximum between `x` and zeros → ReLU behavior.
   - **Memory & Compute Characteristics:**
     - Reads all N×M elements of `x` and writes N×M outputs.
     - Simple elementwise operation ⇒ memory‐bound on large tensors.
   - **Pros:**
     - Leverages PyTorch’s built‐in vectorized maximum.
     - Very readable and concise.
   - **Cons:**
     - Launches a full CUDA kernel under the hood for each op.
     - No control over block size or fusion—memory traffic may dominate.
     - No masking logic; PyTorch handles bounds automatically.

2. Conversion Plan:
   1. To convert this PyTorch code to Triton, we will need to flatten the 2D tensor into a 1D view so that each Triton program handles a contiguous BLOCK_SIZE chunk—mirrors PyTorch’s elementwise op across N×M elements.
   2. To compute these chunks, we will launch a 1D grid of size `ceil(N*M / BLOCK_SIZE)` using `tl.program_id(0)` → `pid`.
   3. Next, we will calculate the per-thread index range `offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)` to cover all elements in this block.
   4. Because the total number of elements may not be divisible by BLOCK_SIZE, we will generate a boolean mask `mask = offs < (N*M)` so out-of-bounds threads do nothing.
   5. We will use `tl.load(X + offs, mask=mask, other=0.0)` to fetch x values from global memory into registers, matching PyTorch’s read of `x`.
   6. To implement ReLU, we will call `tl.maximum(x_vals, 0.0)`, which corresponds directly to `torch.maximum(x, 0)`.
   7. We will write results back using `tl.store(Y + offs, y_vals, mask=mask)`, ensuring correct memory writes.
   8. We will choose BLOCK_SIZE = 1024 (a power of two) to balance GPU occupancy and register usage—this gives good throughput on most NVIDIA architectures.
   9. In the Python wrapper, we will preserve the original function signature `relu(x, BLOCK_SIZE=1024)` so it can be a drop-in replacement.
   10. The wrapper will allocate an output tensor with `torch.empty_like(x)`, compute `grid = ((N*M + BLOCK_SIZE - 1) // BLOCK_SIZE,)`, and invoke the Triton kernel with the same pointer and stride parameters as PyTorch.
   11. We will include minimal comments in the kernel and wrapper mapping each code block back to steps 1–7.
</think>

<triton>
import torch
import triton
import triton.language as tl

# 1. Triton kernel implementing flat ReLU across x.view(-1)
@triton.jit
def triton_relu_kernel(X, Y, stride_row, stride_col, BLOCK_SIZE: tl.constexpr):
    # Step 2–4: compute and mask offsets
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < (stride_row * stride_col)
    # Step 5: load inputs
    x_vals = tl.load(X + offs, mask=mask, other=0.0)
    # Step 6: apply ReLU
    y_vals = tl.maximum(x_vals, 0.0)
    # Step 7: store outputs
    tl.store(Y + offs, y_vals, mask=mask)

# 2. Drop-in Python wrapper matching PyTorch signature
def relu(x, BLOCK_SIZE: int = 1024):
    'Drop-in replacement for pytorch relu'
    n, m = x.shape
    y = torch.empty_like(x)
    total_elems = n * m
    # Step 2: compute grid size
    grid = ((total_elems + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    # Launch Triton kernel
    triton_relu_kernel[grid](
        x.data_ptr(),         # X pointer
        y.data_ptr(),         # Y pointer
        m, 1,                 # strides: row-major
        BLOCK_SIZE            # block size
    )
    return y
</triton>
"""

user_prompt = """
** PyTorch Code **

```python
{pt_code}
```

Produce Conversion Plan and Final Implementation following the exact detailed format above. 
The entrypoint function must be named: {entrypoint}
The Triton kernel implementation (called by the entrypoint) must be named: triton_{entrypoint}_kernel

No computation logic should be done within the entrypoint function. All computation logic should be done within the Triton kernel implementation. 
Enclose the conversion reasoning with <think> ... </think> and the implementation with <triton> ... </triton> tags."""

def format_example(example):
    pt_code = example[args.pt_col]
    entrypoint = example[args.entrypoint_col]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt.format(pt_code=pt_code, entrypoint=entrypoint)},
    ]
    if args.output_col:
        output = example[args.output_col]
        messages.append({"role": "assistant", "content": output})


    # Format the prompt with the preprocessed code
    return {
        "prompt": messages,
        "entrypoint": entrypoint,
        "tests": example["tests_code"],
    }

if not args.debug:
   formatted_ds = ds.map(format_example)
   formatted_ds.push_to_hub(args.ds_name)