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
    entrypoint: str = Field(description="The name of the function")

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

fixer_pytorch_prompt = """You are an expert PyTorch developer. Your goal is to fix the test cases for a given PyTorch function. The function code is correct, but the tests were generated by an LLM and may contain errors. Do not modify the function code—only adjust the tests.

# Instructions:
- Try running the code first, to see what the error is.
- Fix the tests, make sure that they are fixed and the logic of the tests is preserved.
- If a test cannot be fixed, remove it or create a simpler one.
- At the last step, return only the code with the fixed tests. It should be basically the same code as the original code, but with the tests fixed.

# Regarding the tests:
- Write a test code in Python for the above code. Ensure that all branch tests are in a single function starting with
“test_”, with no parameters.
- Particular attention should be paid to the fact that tensor parameters are of GPU type.
- Try to limit the number of branches to no more than 4.
- In branch tests, avoid modifying parameters that are later in the argument list with default values (especially if
they have out parameters, do not assign them).
- Store the results of all branch calculations in a dictionary, where the dictionary key is "test_case_n", with n
representing the test case number.
- Ensure that the import paths match exactly as described in the operator documentation to maintain accuracy.
- The code should run directly, without if __name__ == "__main__".
- Remember to run the code one last time to make sure the tests are fixed before returning the code.
- The tests are meant to be run on the GPU, so use device='cuda' when creating the tensors and when appropriate.
- Remove any unnecesary comments or commented out code.

IMPORTANT: Never change the code, only the tests.

# A perfect example would look like this:

```python
import torch
from typing import Optional
def add(input: torch.Tensor, other: torch.Tensor, alpha: float=1, out: Optional[torch.Tensor]=None):
    \"\"\"
    Adds the tensor or number 'other', scaled by 'alpha', to the 'input' tensor.
    
    Args:
        input (Tensor): The input tensor.
        other (Tensor or Number): The tensor or number to add to input.
        alpha (Number, optional): The multiplier for 'other'. Default is 1.
        out (Tensor, optional): The output tensor. If provided, the result will be stored in this tensor.
        
    Returns:
        Tensor: The result of adding 'other' scaled by 'alpha' to 'input'.
    \"\"\"
    return torch.add(input, other, alpha=alpha, out=out)

##################################################################################

import torch
torch.manual_seed(42)

def test_add():
    results = {{}}

    # Test case 1: Adding two tensors with default alpha
    input1 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    other1 = torch.tensor([4.0, 5.0, 6.0], device='cuda')
    results["test_case_1"] = add(input1, other1)

    # Test case 2: Adding a tensor and a scalar with default alpha
    input2 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    other2 = 2.0
    results["test_case_2"] = add(input2, other2)

    # Test case 3: Adding two tensors with a specified alpha
    input3 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    other3 = torch.tensor([4.0, 5.0, 6.0], device='cuda')
    results["test_case_3"] = add(input3, other3, alpha=0.5)

    # Test case 4: Adding a tensor and a scalar with a specified alpha
    input4 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    other4 = 2.0
    results["test_case_4"] = add(input4, other4, alpha=0.5)

    return results

test_results = test_add()
print(test_results)
```
"""


formatting_prompt = """You are an expert developer with deep expertise in PyTorch implementations. Your primary objective is to organize and refactor the code to be more readable and easier to maintain.
1.	Set a global device standard:
	-	Add a global variable at the top of your scripts:
    ```py
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    ```
    - Ensure every PyTorch tensor created during tests explicitly uses device=DEVICE. This ways we can easily switch between CPU and GPU.
2.	Code Cleanup & Consistency:
	-	Maintain consistency in commenting style and clarity.
    -   Add typing to the function entrypoint, make sure to use the correct types.
    -   Add docstrings to the functions, and abide to the docstring format of the example below.
	-	Remove unnecessary commented code, redundant imports, print statements or unused variables.
3.	Integration & Testing:
	-   Separate the Code and the tests by a line of `######################################`
    -   Store all test results in a dictionary (e.g., results["test_case_n"]).
    -   Do not include an if __name__ == "__main__": section.
    -   Add a single print statement at the end of the tests, printing the test_results dictionary.
    -   Make sure the signature of the test function is `test_<function_name>()`
4.  Entrypoint:
    -   The entrypoint should be the name of the function. Make sure they align with the function name.
    -   If the name of the function and the tests_<function_name> do not match, change the name of the tests function to match the function name.

# Example inputs and outputs:

## Input:

```python
import torch

def addmm(input, mat1, mat2, beta=1, alpha=1, out=None):
    \"\"\"
    Performs matrix multiplication of mat1 and mat2, and adds input to the result.
    \"\"\"
    return torch.addmm(input, mat1, mat2, beta=beta, alpha=alpha, out=out)



def test_addmm():
    # Test case 1: Default beta and alpha
    input1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    mat1_1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device='cuda')
    mat2_1 = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device='cuda')
    print(addmm(input1, mat1_1, mat2_1))

    # Test case 2: Custom beta and alpha
    input2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    mat1_2 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device='cuda')
    mat2_2 = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device='cuda')
    print(addmm(input2, mat1_2, mat2_2, beta=0.5, alpha=2.0))

    # Test case 3: Zero beta
    input3 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    mat1_3 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device='cuda')
    mat2_3 = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device='cuda')
    print(addmm(input3, mat1_3, mat2_3, beta=0.0))
```

## Example output: You should only retutn the code and the entrypoint.
- The test part should have the imports required to run the tests.
- Only print the test_results dictionary, not any other intermediate results.
- have the DEVICE variable at the top of the file.
- For reproducibility alway add a `torch.manual_seed(42)` at the top of the test part.

entrypoint: addmm

code:

```python
import torch

def addmm(
        input: torch.Tensor, 
        mat1: torch.Tensor, 
        mat2: torch.Tensor, 
        beta: float=1, 
        alpha: float=1, 
        out: torch.Tensor=None
        ) -> torch.Tensor:
    \"\"\"
    Performs matrix multiplication of mat1 and mat2, and adds input to the result.

    Parameters:
        input (torch.Tensor): Matrix to be added.
        mat1 (torch.Tensor): The first matrix to be matrix-multiplied.
        mat2 (torch.Tensor): The second matrix to be matrix-multiplied.
        beta (float, optional): Multiplier for input (default is 1).
        alpha (float, optional): Multiplier for mat1 @ mat2 (default is 1).
        out (torch.Tensor, optional): The output tensor to store the result.

    Returns:
        torch.Tensor: The resulting tensor after performing the operation.
    
    This function performs the matrix multiplication of mat1 and mat2, scales the result by alpha,
    and then adds it to the input matrix scaled by beta. The resulting matrix is returned.
    
    If input is sparse, the result will have the same layout as input. If out is provided,
    it must have the same layout as input. If beta is 0, the input will be ignored, and nan or inf
    in input will not be propagated.
    \"\"\"
    return torch.addmm(input, mat1, mat2, beta=beta, alpha=alpha, out=out)

#################################################################################


import torch
torch.manual_seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

def test_addmm():
    results = {{}}

    # Test case 1: Default beta and alpha
    input1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=DEVICE)
    mat1_1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=DEVICE)
    mat2_1 = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device=DEVICE)
    results["test_case_1"] = addmm(input1, mat1_1, mat2_1)

    # Test case 2: Custom beta and alpha
    input2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=DEVICE)
    mat1_2 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=DEVICE)
    mat2_2 = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device=DEVICE)
    results["test_case_2"] = addmm(input2, mat1_2, mat2_2, beta=0.5, alpha=2.0)

    # Test case 3: Zero beta
    input3 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=DEVICE)
    mat1_3 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=DEVICE)
    mat2_3 = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device=DEVICE)
    results["test_case_3"] = addmm(input3, mat1_3, mat2_3, beta=0.0)

    # Test case 4: bigger tensors
    input4 = torch.randn(30, 20, device=DEVICE)

    return results

test_results = test_addmm()
print(test_results)
```

Return your results in JSON format with keys: 
- entrypoint: the name of the function
- code: A complete pure PyTorch code along with test cases, no backticks or ```python ```.

# Final notes:
Make sure the tests are not printing intermediate results. As you can see there is a single print statement at the end of the tests.
"""


extract_pytorch_tests_prompt = """You are an expert developer with deep expertise in PyTorch and Triton kernels. Your task is to:
1.	Extract the PyTorch tests from the provided code.
2.	Return only the PyTorch tests, no other code.
3.	Ensure the tests are formatted correctly and can be run directly.
4.	Make sure the tests are not printing intermediate results. As you can see there is a single print statement at the end of the tests.
This is really important if we want to compare the 2 implementations in the future.
"""

annotator_prompt = """You are an expert developer with deep expertise in PyTorch. Your task is to:
1. We have a pytorch code with some errors.
2. You have the fixed pytorch code
2. You havethe original errors.
3. YOu aghve to explain what was the error and how it was fixed.

Don't chanage the naming of the functions, just fix the errors.
Make sure to preserve the structure, with a single print statement at the end of the tests.
"""

pytorch_agent_prompt = """
You are an expert PyTorch developer and test-engineering agent.  
For **each file in the dataset** follow the pipeline below.

───────────────────────────────  WORKFLOW  ────────────────────────────────
1. **Run the file** exactly as received.  
2. **Investigate any failure** (exceptions, wrong values, style lint, etc.).  
3. **Fix whatever is broken** – preferably the tests, but patch the operator
   itself if that is the real defect.  Keep changes minimal and clean.
4. **Refactor & format** both implementation and tests so they meet the
   standards in the next section.  
5. **Re-execute** the fixed code **and** its tests.  
6. **Emit** your answer as a JSON object that matches the
   `PytorchCodeWithTests` model (see “OUTPUT FORMAT”).

───────────────────────────────  STANDARDS  ───────────────────────────────
A.  Global device switch  
    ```py
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ```

B.  Implementation code (`format_pt_code`)  
    • Full type hints and a Google-style docstring.  
    • Remove dead code, redundant imports, prints.  
    • Keep logic minimal; only change behaviour when required to satisfy
      correct semantics and tests.

C.  Tests (`tests`)  
    • Single function named `test_<entrypoint>()`.  
    • ≤ 4 branch checks, each result stored in
      `results["test_case_n"]`.  
    • Every tensor uses `device=DEVICE`; seed once with
      `torch.manual_seed(42)`.  
    • No intermediate prints; only  
      ```py
      print(test_results)
      ```  
      at the very end.  
    • Don't add a __main__ section at the end of the file.
    • If a faulty branch cannot be repaired, drop or simplify it.

D.  File anatomy *inside the combined working file* (you will **return the
   parts separately**, but keep this structure while fixing):  
    1. Imports & `DEVICE` declaration  
    2. Operator / library code  
    3. A separator line of exactly 40 `#` characters  
    4. Tests

E.  Allowed changes  
    ✓ Fix wrong logic, defaults, dtype/device mismatches, etc.  
    ✓ Edit / delete / add tests as needed for meaningful coverage.  
    ✗ No third-party libs beyond Python & PyTorch.


──────────────────────────────  OUTPUT FORMAT  ─────────────────────────────

Reply with the code, a brief description of the changes you made, if it ran and the entrypoint

entrypoint: addmm
ran: True
code:

```python
import torch

def addmm(
        input: torch.Tensor, 
        mat1: torch.Tensor, 
        mat2: torch.Tensor, 
        beta: float=1, 
        alpha: float=1, 
        out: torch.Tensor=None
        ) -> torch.Tensor:
    \"\"\"
    Performs matrix multiplication of mat1 and mat2, and adds input to the result.

    Parameters:
        input (torch.Tensor): Matrix to be added.
        mat1 (torch.Tensor): The first matrix to be matrix-multiplied.
        mat2 (torch.Tensor): The second matrix to be matrix-multiplied.
        beta (float, optional): Multiplier for input (default is 1).
        alpha (float, optional): Multiplier for mat1 @ mat2 (default is 1).
        out (torch.Tensor, optional): The output tensor to store the result.

    Returns:
        torch.Tensor: The resulting tensor after performing the operation.
    
    This function performs the matrix multiplication of mat1 and mat2, scales the result by alpha,
    and then adds it to the input matrix scaled by beta. The resulting matrix is returned.
    
    If input is sparse, the result will have the same layout as input. If out is provided,
    it must have the same layout as input. If beta is 0, the input will be ignored, and nan or inf
    in input will not be propagated.
    \"\"\"
    return torch.addmm(input, mat1, mat2, beta=beta, alpha=alpha, out=out)

#################################################################################


import torch
torch.manual_seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

def test_addmm():
    results = {{}}

    # Test case 1: Default beta and alpha
    input1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=DEVICE)
    mat1_1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=DEVICE)
    mat2_1 = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device=DEVICE)
    results["test_case_1"] = addmm(input1, mat1_1, mat2_1)

    # Test case 2: Custom beta and alpha
    input2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=DEVICE)
    mat1_2 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=DEVICE)
    mat2_2 = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device=DEVICE)
    results["test_case_2"] = addmm(input2, mat1_2, mat2_2, beta=0.5, alpha=2.0)

    # Test case 3: Zero beta
    input3 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    mat1_3 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=DEVICE)
    mat2_3 = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device=DEVICE)
    results["test_case_3"] = addmm(input3, mat1_3, mat2_3, beta=0.0)

    return results

test_results = test_addmm()
print(test_results)
```

"""


test_creator_prompt = """You are an expert PyTorch developer. Your goal is to create test cases for a given PyTorch function.

You have access to tools to run code, make sure to use it to validate your tests run.

# Instructions:
- Try running the code first without the tests.
- Create a series of simple tests to validate the code.
- Don't change the code, only the tests.

# Tests formattin instructions:
- Write a test code in Python for the above code. Ensure that all branch tests are in a single function starting with
“test_”, with no parameters.
- Particular attention should be paid to the fact that tensor parameters are of GPU type.
- Try to limit the number of branches to no more than 4.
- In branch tests, avoid modifying parameters that are later in the argument list with default values (especially if
they have out parameters, do not assign them).
- Store the results of all branch calculations in a dictionary, where the dictionary key is "test_case_n", with n
representing the test case number.
- Make sure to add one test with larger inputs, you can use torch.randn to create them.
- Ensure that the import paths match exactly as described in the operator documentation to maintain accuracy.
- The code should run directly, without if __name__ == "__main__".
- Remember to run the code one last time to make sure the tests are fixed before returning the code.
- The tests are meant to be run on the GPU, so use device='cuda' when creating the tensors and when appropriate.
- Remove any unnecesary comments or commented out code.
- Add a single print statement at the end of the tests, printing the test_results dictionary.
- Make sure the signature of the test function is `test_<function_name>()`
- Use `torch.manual_seed(42)` to seed the random number generator.



IMPORTANT: Never change the code, only the tests.

# A perfect example would look like this:

Pytorch code:
```python
import torch
from typing import Optional

def add(input: torch.Tensor, other: torch.Tensor, alpha: float=1, out: Optional[torch.Tensor]=None):
    \"\"\"
    Adds the tensor or number 'other', scaled by 'alpha', to the 'input' tensor.
    
    Args:
        input (Tensor): The input tensor.
        other (Tensor or Number): The tensor or number to add to input.
        alpha (Number, optional): The multiplier for 'other'. Default is 1.
        out (Tensor, optional): The output tensor. If provided, the result will be stored in this tensor.
        
    Returns:
        Tensor: The result of adding 'other' scaled by 'alpha' to 'input'.
    \"\"\"
    return torch.add(input, other, alpha=alpha, out=out)
```

Tests:
```python
import torch
torch.manual_seed(42)

def test_add():
    results = {{}}

    # Test case 1: Adding two tensors with default alpha
    input1 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    other1 = torch.tensor([4.0, 5.0, 6.0], device='cuda')
    results["test_case_1"] = add(input1, other1)

    # Test case 2: Adding a tensor and a scalar with default alpha
    input2 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    other2 = 2.0
    results["test_case_2"] = add(input2, other2)

    # Test case 3: Adding two tensors with a specified alpha
    input3 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    other3 = torch.tensor([4.0, 5.0, 6.0], device='cuda')
    results["test_case_3"] = add(input3, other3, alpha=0.5)

    # Test case 4: Larger inputs
    input4 = torch.randn(30, 20, device=DEVICE)
    other4 = torch.randn(30, 20, device=DEVICE)
    alpha = 0.5
    results["test_case_4"] = add(input4, other4, alpha=alpha)

    return results

test_results = test_add()
print(test_results)
```
"""


sft_system_prompt = """
# GPU‐Kernel Reasoner Prompt

You are an expert GPU‐kernel reasoner and Triton evangelist. You will be given two code snippets—one in PyTorch and one in Triton that implement the same tensor operation. Your goal is to:

1. **Analyze the PyTorch implementation**  
   - Break down its algorithmic steps, memory access patterns, and computational characteristics.  
   - Identify potential performance bottlenecks or numerical‐stability issues in the PyTorch version.  
   - List the **Pros** of the existing PyTorch code (e.g., readability, use of optimized libraries) and the **Cons** (e.g., extra memory traffic, kernel launch overhead).

2. **Build a detailed Conversion Plan**  
   - Walk through *every* design decision and transformation needed to convert the PyTorch code into a high‐performance Triton kernel plus a Python wrapper.  

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

Don't return the code, just the Analysis and Conversion plan.

---

## Example

**PyTorch Code**  
```python
def relu(x: torch.Tensor) -> torch.Tensor:
    # x: FloatTensor[N, M]
    return torch.maximum(x, torch.zeros_like(x))
```

**Triton Code**
```python
import triton
import triton.language as tl

@triton.jit
def triton_relu_kernel(X, Y, stride_row, stride_col, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < stride_row * stride_col
    x_vals = tl.load(X + offs, mask=mask, other=0.0)
    y_vals = tl.maximum(x_vals, 0.0)
    tl.store(Y + offs, y_vals, mask=mask)

def relu(x, BLOCK_SIZE: int = 1024):
    n, m = x.shape
    y = torch.empty_like(x)
    grid = ((n * m + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    triton_relu_kernel[grid](
        x.data_ptr(), y.data_ptr(),
        m, 1,
        BLOCK_SIZE
    )
    return y
```

**Expected Output**

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

"""


class TorchTritonReasoning(BaseModel):
    reasoning: str = Field(description="The reasoning for the conversion from the PyTorch code to the Triton code.")


sft_user_prompt = """
** PyTorch Code **
{pt_code}

** Triton Code **
{triton_code}

As we already have the Triton Implementation, produce the Conversion Plan and reasoning for the conversion from the PyTorch code to the Triton code. Don't return any code."""

system_prompt_remove_func = """You are an expert Python programmer. Your task is to fix the tests"""


user_prompt_remove_func = """
You are presented with a Python test code like this:

```python
import torch

def window_max(x: torch.Tensor, k: int) -> torch.Tensor:
    \"\"\"
    Computes the sliding window maximum of a 1D tensor with window size k.
    Returns a tensor of length len(x) - k + 1, where each element i is the max over x[i:i+k].
    \"\"\"
    return x.unfold(0, k, 1).max(dim=1)[0]

torch.manual_seed(42)

def test_window_max():
    results = {{}}

    # Test case 1: Simple increasing sequence, k=2
    x1 = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32, device='cuda')
    k1 = 2
    results["test_case_1"] = window_max(x1, k1)

    # Test case 2: Sequence with negative values, k=3
    x2 = torch.tensor([-1, -3, 2, 4, 0], dtype=torch.float32, device='cuda')
    k2 = 3
    results["test_case_2"] = window_max(x2, k2)

    # Test case 3: All elements equal, k=4
    x3 = torch.tensor([7, 7, 7, 7, 7], dtype=torch.float32, device='cuda')
    k3 = 4
    results["test_case_3"] = window_max(x3, k3)

    # Test case 4: Large random input, k=10
    x4 = torch.randn(1000, device='cuda')
    k4 = 10
    results["test_case_4"] = window_max(x4, k4)

    return results

test_results = test_window_max()
print(test_results)
```

I want you to remove the function to test if it is present, returning onlty the tests:

```python
import torch
torch.manual_seed(42)

def test_window_max():
    results = {{}}

    # Test case 1: Simple increasing sequence, k=2
    x1 = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32, device='cuda')
    k1 = 2
    results["test_case_1"] = window_max(x1, k1)

    # Test case 2: Sequence with negative values, k=3
    x2 = torch.tensor([-1, -3, 2, 4, 0], dtype=torch.float32, device='cuda')
    k2 = 3
    results["test_case_2"] = window_max(x2, k2)

    # Test case 3: All elements equal, k=4
    x3 = torch.tensor([7, 7, 7, 7, 7], dtype=torch.float32, device='cuda')
    k3 = 4
    results["test_case_3"] = window_max(x3, k3)

    # Test case 4: Large random input, k=10
    x4 = torch.randn(1000, device='cuda')
    k4 = 10
    results["test_case_4"] = window_max(x4, k4)

    return results

test_results = test_window_max()
print(test_results)
```

If the function is not present, just return the tests as is.

# Tests:
```python
{tests}
```
"""

class Tests(BaseModel):
    tests: str = Field(description="The tests to run, with the function to test. Only the code without any ```python or ``` needed, just the code")
    triton_stderr: str = Field(description="The stderr of the Triton code.")
    triton_stdout: str = Field(description="The stdout of the Triton code.")


eval_system_prompt = """
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


To make our life easier, enclose all the reasoning and conversion plan with <reasoning> ... </reasoning> tags. For the final implementation, reply with a single blob of code enclosed with <triton> ... </triton> tags.

---

## Example

**PyTorch Code**  
```python
def relu(x: torch.Tensor) -> torch.Tensor:
    # x: FloatTensor[N, M]
    return torch.maximum(x, torch.zeros_like(x))
```

**Expected Output**
<reasoning>
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
</reasoning>

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

eval_user_prompt = """
** PyTorch Code **

```python
{pt_code}
```

Produce Conversion Plan and Final Implementation following the exact detailed format above. 
The entrypoint function must be named: {entrypoint}
The Triton kernel implementation (called by the entrypoint) must be named: triton_{entrypoint}_kernel

No computation logic should be done within the entrypoint function. All computation logic should be done within the Triton kernel implementation. 
Enclose the conversion reasoning with <reasoning> ... </reasoning> and the implementation with <triton> ... </triton> tags."""
