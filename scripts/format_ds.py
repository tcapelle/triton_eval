from dataclasses import dataclass
from datasets import load_dataset, Dataset
import simple_parsing as sp


@dataclass
class Args:
    ds_name: str = "tcapelle/train_ds_triton_sft"
    debug: bool = False
    pt_col: str = "pt_code_without_tests"
    entrypoint_col: str = "entrypoint"
    output_col: str = "reasoning"

args = sp.parse(Args)

ds = load_dataset(args.ds_name)["train"]


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
    output = example[args.output_col]

    # Format the prompt with the preprocessed code
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt.format(pt_code=pt_code, entrypoint=entrypoint)},
            {"role": "assistant", "content": output},
        ],
    }

formatted_ds = ds.map(format_example)
formatted_ds.push_to_hub(args.ds_name)