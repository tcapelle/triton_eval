import weave

from datasets import load_dataset, DatasetDict
from dataclasses import dataclass
import simple_parsing as sp

@dataclass
class Args:
    dataset_name: str = "tcapelle/train_ds_triton_v2f2"#"GPUMODE/Inductor_Created_Data_Permissive"
    prepared_dataset_name: str = "tcapelle/train_ds_triton"
    code_column: str = "pt_code_without_tests"
    split: str = "train"


SYSTEM_PROMPT = """
# GPU‐Kernel Reasoner Prompt

You are an expert GPU‐kernel reasoner and Triton programmer. You will be given a PyTorch code snippet. Your goal is to:

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
- Begin with “To convert this PyTorch code to Triton, we will need to…”  
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


To make our life easier, enclose all the reasoning and conversion plan with <think> ... </think> tags. 
For the final implementation, reply with a single blob of code enclosed with <triton> ... </triton> tags.
---

## Example

**PyTorch Code**  
```python
import torch
def relu(x: torch.Tensor) -> torch.Tensor:
    # x: FloatTensor[N, M]
    return torch.maximum(x, torch.zeros_like(x))
```

**Expected Output**
<think>
1. PyTorch Analysis:
   ...

2. Conversion Plan:
   ...
</think>

3. Final Implementation
<triton>
import torch
import triton
import triton.language as tl

# 1. Triton kernel implementing flat ReLU across x.view(-1)
@triton.jit
def triton_relu_kernel(X, Y, stride_row, stride_col, BLOCK_SIZE: tl.constexpr):
    ...
def relu(x, BLOCK_SIZE: int = 1024):
    ...
</triton>
"""

USER_PROMPT = """
** PyTorch Code **
{pytorch_code}

Produce Conversion Plan and Final Implementation following the exact detailed format above. 
The entrypoint function must be named: {entrypoint}
The Triton kernel implementation (called by the entrypoint) must be named: triton_{entrypoint}_kernel

No computation logic should be done within the entrypoint function. All computation logic should be done within the Triton kernel implementation. 
Enclose the conversion reasoning with <think> ... </think> and the implementation with <triton> ... </triton> tags."""


def get_dataset(dataset_name, code_column, split="train"):
    # Load the dataset - this is expected to be preprocessed by prepare_dataset.py
    data = load_dataset(dataset_name)[split]
    print(f"Loaded {len(data)} examples from {dataset_name} split {split}")
    data = data.filter(lambda x: x[code_column] is not None)
    print(f"Filtered to {len(data)} examples with non-None {code_column}")
    data = data.filter(lambda x: x["entrypoint"] is not None)
    print(f"Filtered to {len(data)} examples with non-None entrypoint")
    data = data.filter(lambda x: x["tests"] is not None)
    print(f"Filtered to {len(data)} examples with non-None tests")
    data = data.filter(lambda x: x["pt_code_runs"])
    print(f"Filtered to {len(data)} examples with non-None pt_code_runs")

    # Format the prompt with the preprocessed code
    def format_example(example):
        # Format the prompt with the preprocessed code
        pytorch_code = example[code_column]
        entrypoint = example["entrypoint"]
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT.format(pytorch_code=pytorch_code, entrypoint=entrypoint)},
                {"role": "assistant", "content": "<think>\n"},
            ],
        }
    
    # Format each example in the dataset
    formatted_data = data.map(format_example)

    return formatted_data

if __name__ == "__main__":
    args = sp.parse(Args)
    dataset = get_dataset(args.dataset_name, args.code_column, split=args.split)
    dataset_dict = DatasetDict({
        "train": dataset,
        "debug": dataset.select(range(24)),
    })

    dataset_dict.push_to_hub(args.prepared_dataset_name, commit_message="push prepared")

    weave.init("grpo-cuda/axolotl-grpo")
    wds = weave.Dataset(rows=dataset.to_list(), name="new_format")
    weave.publish(wds)
