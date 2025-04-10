"A collection of prompts for the annotation of the triton dataset"

from pydantic import BaseModel, Field

class KernelInfo(BaseModel):
    description: str = Field(description="A concise description of the triton code")


description_prompt = """
You are an expert in Triton and PyTorch.
Let's generate a description of the following triton code. Make it concise and to the point. Explain the reasoning behind the code.

## Triton Code

```python
{code}
```

"""

class PytorchCodeWithTestCases(BaseModel):
    pytorch_code_with_test_cases: str = Field(description="A complete pure PyTorch implementation of the Triton code along with test cases")

generate_pytorch_prompt = """You are given a Triton kernel implementation along with a description of what it does. Your task is to generate:
1. A complete pure PyTorch equivalent of the provided functionality.
2. A comprehensive list of test cases in code form that validate the behavior of the function.

The input will be provided in the following format:
---
description: <Description of what the Triton kernel does (e.g., "A simple gelu Triton implementation")>
code:
<The Triton kernel code here>
---

Your output should include:
1. The original Triton kernel code as provided.
2. A pure PyTorch implementation that performs the same computation as the Triton kernel. This should be a complete, self-contained function.
3. A set of test cases (wrapped in a function like `test_<function>()`) that test various scenarios and demonstrate the correctness of the PyTorch implementation. The test cases should include examples for different input sizes and parameter variations when applicable.

# Example:
Given the Triton kernel and description:

## description: 

A simple gelu Triton implementation

## triton kernel code:

```python
import math
import torch
import triton
import triton.language as tl

@triton.jit
def _gelu_kernel(
    in_ptr, 
    out_ptr, 
    n, 
    approximate_tanh: tl.constexpr, 
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    
    if approximate_tanh:
        # tanh approximation
        # 0.5 * x * (1 + Tanh(√(2/π) * (x + 0.044715 * x^3)))
        sqrt_2_over_pi = math.sqrt(2.0 / math.pi)
        y = x + 0.044715 * (x ** 3)
        out_val = 0.5 * x * (1.0 + tl.tanh(sqrt_2_over_pi * y))
    else:
        # exact version
        # x * Φ(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        out_val = x * 0.5 * (1.0 + tl.erf(x * inv_sqrt2))

    tl.store(out_ptr + offsets, out_val, mask=mask)
```

Your output must include:

1. **Pure PyTorch Implementation:**  
   Write a function called `gelu` that mimics the behavior of the Triton kernel without using any PyTorch higher-level GELU function. The function should support an `approximate` argument that can be either `'none'` or `'tanh'`, corresponding to the two branches in the kernel. Specifically, for each element `x` in the input tensor, the function should compute:
   
   - For `approximate == 'tanh'`:  
     `0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))`
     
   - For `approximate == 'none'`:  
     `x * 0.5 * (1 + erf( x / sqrt(2) ))`
   
   This implementation should operate element-wise on the input tensor.

2. **Test Cases:**  
   Create several test cases (wrapped in a function, e.g. `test_gelu()`) that:
   - Test both variants (`'none'` and `'tanh'`) on small input tensors.
   - Return the results of the test cases as a dictionary.
   - add a main section that runs the test cases: `test_results = test_gelu()`

Your final output should be a complete, self-contained single file example, similar to the structure below:

# PyTorch Implementation

```python
import torch
import math

def gelu(input: torch.Tensor, approximate: str = 'none') -> torch.Tensor:
    if approximate not in ['none', 'tanh']:
        raise ValueError("approximate must be 'none' or 'tanh'")
    
    # Compute element-wise GELU:
    if approximate == 'tanh':
        sqrt_2_over_pi = math.sqrt(2.0 / math.pi)
        # Compute the polynomial part: x + 0.044715 * x^3
        y = input + 0.044715 * input.pow(3)
        return 0.5 * input * (1.0 + torch.tanh(sqrt_2_over_pi * y))
    else:
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        return input * 0.5 * (1.0 + torch.erf(input * inv_sqrt2))

######################################################################

import torch
import torch.nn.functional as F

def gelu(input: torch.Tensor, approximate: str='none') -> torch.Tensor:
    return F.gelu(input, approximate=approximate)

def test_gelu():
    results = {{}}
    
    # Test case 1: Default approximate='none'
    input_tensor_1 = torch.tensor([-1.0, 0.0, 1.0], device='cuda')
    results["test_case_1"] = gelu(input_tensor_1)
    
    # Test case 2: approximate='tanh'
    input_tensor_2 = torch.tensor([-1.0, 0.0, 1.0], device='cuda')
    results["test_case_2"] = gelu(input_tensor_2, approximate='tanh')
    
    # Test case 3: Larger tensor with default approximate='none'
    input_tensor_3 = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], device='cuda')
    results["test_case_3"] = gelu(input_tensor_3)
    
    # Test case 4: Larger tensor with approximate='tanh'
    input_tensor_4 = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], device='cuda')
    results["test_case_4"] = gelu(input_tensor_4, approximate='tanh')
    
    return results

if __name__ == "__main__":
    test_results = test_gelu()
```

# Here is another input code and description:

## triton kernel code:
```python
{code}
```

## description:
{description}
"""
