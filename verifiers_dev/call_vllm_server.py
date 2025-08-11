from dataclasses import dataclass

import openai
import simple_parsing as sp

@dataclass
class ScriptArgs:
    model_name: str = "Qwen/Qwen3-4B"
    ip: str = "127.0.0.1"
    port: int = 8000

args = sp.parse(ScriptArgs)

openai_client = openai.OpenAI(
    base_url=f"http://{args.ip}:{args.port}/v1",
    api_key="NoKey",
)

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

Enclose all the reasoning and conversion plan with <think> ... </think> tags.

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

def make_call(prompt: str):
    response = openai_client.chat.completions.create(
        model=args.model_name,
        messages=[{"role": "user", "content": test_creator_prompt}],
        max_tokens=12000,

    )
    print(response)
    return response.choices[0].message.content

print(f"Making call to: {args.model_name} at {args.ip}:{args.port}")
out = make_call("Hello, world!")
print(f"\noutput: {out}")

