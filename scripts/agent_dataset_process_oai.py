import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from enum import Enum

from datasets import Dataset, load_dataset, load_from_disk
from pydantic import BaseModel, Field
from rich.console import Console
import simple_parsing as sp
import weave
import openai

from agents import Agent, Runner, RunContextWrapper, function_tool

from triton_eval.agents.tools import run_python_code_on_gpu
from triton_eval.utils import compare_outputs

console = Console()

client = openai.AsyncOpenAI()

@dataclass
class Args:
    debug: bool = False
    input_dataset: str = "tcapelle/boostrap_triton_ran"
    output_dataset: str = "tcapelle/boostrap_triton_ran"
    weave_project: str = "grpo-cuda/dataset_agent"
    push: bool = False
    num_proc: int = 10
    max_turns: int = 10

args = sp.parse(Args)

def load_ds(dataset_name, init=True):
    if init:
        return load_dataset("json", data_files="./data/simple_samples.jsonl")["train"]
    elif "/" in dataset_name:
        return load_dataset(dataset_name)["train"]
    else:
        return load_from_disk(dataset_name)["train"]

console.rule(f"[bold blue]Loading dataset: {args.input_dataset}[/bold blue]")

input_ds = load_ds(args.input_dataset, init=True)

console.print("[bold blue]Input dataset[/bold blue]")
console.print(input_ds)

console.rule("[bold blue]Fixing code with Agent[/bold blue]")

weave.init(args.weave_project)

############################################

class FunctionNameAndDescription(BaseModel):
    function_name: str = Field(description="The name of the Pytorch function")
    function_description: str = Field(description="The description of what the Pytorch function does and why it's interesting to convert to Triton")

creativity_system_prompt = """You are an expert in Pytorch and Triton. You are given a dataset of Pytorch functions and their descriptions.
You are tasked with generating a new function that is not in the dataset.
You must generate the function name and a short description of what the function does and why it's interesting to convert to Triton.

Some examples of functions that are interesting to convert to Triton:
- fused_matmul_add: a fused operation that combines matrix multiplication and addition
- conv_transpose2d: a transposed convolution operation
- transpose_2d: a 2D transpose operation
- matmul_relu_add: a fused operation that combines matrix multiplication, ReLU, and addition
- one_dimensional_attention: a 1D attention operation

Be creative and make our dataset diverse and rich.
"""

creativity_user_prompt = """
Current Dataset:
{dataset}

Generate a new function that is not in the dataset.
"""

def dump_ds(ds):
    return "\n".join([f"{row['entrypoint']}: {row['description']}" for row in ds])

async def generate_function_name_and_description(input_ds):
    current_ds_rows = dump_ds(input_ds)
    messages = [
        {"role": "system", "content": creativity_system_prompt},
        {"role": "user", "content": creativity_user_prompt.format(dataset=current_ds_rows)}
    ]
    response = await client.responses.parse(
        model="gpt-4.1",
        input=messages,
        temperature=1.0,
        text_format=FunctionNameAndDescription,
    )
    return response.output_parsed

############################################

class ExecutionType(str, Enum):
    PYTORCH = "pytorch"
    TRITON = "triton"

class UnifiedExecutionContext(BaseModel):
    """Simplified flat context for both PyTorch and Triton execution"""
    
    # Function metadata
    function_name: str = ""
    function_description: str = ""
    
    # PyTorch execution
    pt_code: str = ""
    pt_entrypoint: str = ""
    pt_tests: str = ""
    pt_returncode: int = -1
    pt_stdout: str = ""
    pt_stderr: str = ""
    pt_runs: bool = False
    pt_has_output: bool = False
    pt_error_summary: str = ""
    
    # Triton execution
    triton_code: str = ""
    triton_entrypoint: str = ""
    triton_returncode: int = -1
    triton_stdout: str = ""
    triton_stderr: str = ""
    triton_runs: bool = False
    triton_has_output: bool = False
    triton_error_summary: str = ""
    triton_is_correct: bool = False
    
    # Shared
    tests: str = ""
    
    def store_execution_result(self, exec_type: ExecutionType, result: dict):
        """Store execution result for given type"""
        prefix = exec_type.value if exec_type == ExecutionType.PYTORCH else "triton"
        if exec_type == ExecutionType.PYTORCH:
            prefix = "pt"
        
        setattr(self, f"{prefix}_returncode", result.get("returncode", -1))
        setattr(self, f"{prefix}_stdout", result.get("stdout", ""))
        setattr(self, f"{prefix}_stderr", result.get("stderr", ""))
        setattr(self, f"{prefix}_runs", result.get("returncode", -1) == 0)
        setattr(self, f"{prefix}_has_output", bool(result.get("stdout", "").strip()))
        setattr(self, f"{prefix}_error_summary", result.get("stderr", ""))
    
    def get_execution_summary(self, exec_type: ExecutionType) -> dict:
        """Get execution summary for LLM"""
        prefix = "pt" if exec_type == ExecutionType.PYTORCH else "triton"
        
        runs = getattr(self, f"{prefix}_runs")
        has_output = getattr(self, f"{prefix}_has_output")
        stderr = getattr(self, f"{prefix}_stderr")
        
        return {
            "runs": runs,
            "has_output": has_output,
            "has_error": bool(stderr.strip()),
            "error_summary": stderr,
        }
    
    def to_flat_dict(self) -> dict:
        """Convert to flat dictionary - no more unpack_row needed"""
        return self.model_dump()

@function_tool
async def run_code(
    wrapper: RunContextWrapper[UnifiedExecutionContext], 
    code: str, 
    exec_type: ExecutionType,
    tests: Optional[str] = None
) -> str:
    """Generic tool to run code and store results in context"""
    
    if tests:
        full_code = f"{code}\n\n############\nimport torch\ntorch.set_printoptions(threshold=int(1e9))\n\n{tests}"
    else:
        full_code = code
    
    result = run_python_code_on_gpu(full_code)
    wrapper.context.store_execution_result(exec_type, result)
    
    summary = wrapper.context.get_execution_summary(exec_type)
    
    if summary["runs"]:
        return f"Code executed successfully. Has output: {summary['has_output']}"
    else:
        return f"""Code failed with error: {summary['error_summary']}.
        Reflect on your previous attempts at fixing the error, then try fixing the error."""

@function_tool
async def run_pytorch_code_and_tests(
    wrapper: RunContextWrapper[UnifiedExecutionContext], 
    code: str, 
    tests: str
) -> str:
    """Run PyTorch code and tests"""
    return await run_code(wrapper, code, ExecutionType.PYTORCH, tests)

@function_tool  
async def compare_pytorch_triton_outputs(wrapper: RunContextWrapper[UnifiedExecutionContext]) -> str:
    """Compare PyTorch and Triton outputs"""
    ctx = wrapper.context
    
    if not ctx.pt_runs or not ctx.triton_runs:
        return "Cannot compare - one or both implementations failed to run"
    
    match_results = compare_outputs(ctx.pt_stdout, ctx.triton_stdout)
    ctx.triton_is_correct = all(status == "PASS" for _, status, _, _ in match_results)
    
    results_str = "\n".join([f"{name}: {status} ({msg})" for name, status, msg, _ in match_results])
    return f"Test Results:\n{results_str}"

@function_tool
async def run_triton_code_and_compare(
    wrapper: RunContextWrapper[UnifiedExecutionContext], 
    triton_code: str
) -> str:
    """Run Triton code and compare with PyTorch output"""
    # First run the triton code
    await run_code(wrapper, triton_code, ExecutionType.TRITON, wrapper.context.tests)
    
    # Then compare outputs
    return await compare_pytorch_triton_outputs(wrapper)

### First Agent: Generate PyTorch/Triton pairs
triton_cookbook = Path("./data/triton_cookbook.md").read_text()

class PytorchOutput(BaseModel):
    pt_code: str = Field(description="The PyTorch code for the function")
    pt_entrypoint: str = Field(description="The entrypoint of the function in Pytorch")
    tests: str = Field(description="The tests for the function")
    pt_runs: bool = Field(description="Whether the pytorch code runs or not.")
    pt_has_output: bool = Field(description="Whether the pytorch code produced output.")
    pt_error_summary: str = Field(default="", description="Brief summary of any pytorch errors.")

class TritonOutput(BaseModel):
    conversion_reasoning: str = Field(default="", description="The reasoning step by step on how the conversion to triton should be done for this specific function")
    triton_code: str = Field(default="", description="The Triton code for the function, no tests are needed, just the triton code")
    triton_entrypoint: str = Field(default="", description="The entrypoint of the function in Triton")
    triton_runs: bool = Field(default=False, description="Whether the triton code runs or not.")
    triton_has_output: bool = Field(default=False, description="Whether the triton code produced output.")
    triton_error_summary: str = Field(default="", description="Brief summary of any triton errors.")
    triton_is_correct: bool = Field(default=False, description="Whether the triton code is correct or not.")

pytorch_generation_system_prompt = f"""We are generating a PyTorch/Triton pairs dataset. We want functions that have exactly the same functionalities.

Your task is to generate a new row for our dataset. Focus on clarity and simplicity, Triton can be very complex, the idea is generating pairs that are easy to understand and that can be used to learn Triton. With each new sample add more complexity to the code.

You have to generate the pytorch code and tests for a given function.

## Regarding the tests:

- Ensure that all branch tests are in a single function starting with
"test_", with no parameters.
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


A perfect example of the pytorch function and tests would look like this:

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

You must use the run_pytorch_code_and_tests tool to run your code and tests. You should return a working pytorch implementation of the function with the tests.
"""

pytorch_generation_user_prompt = """
Generate the pytorch code and tests for the function: {function_name}
description: {function_description}
"""

triton_generation_system_prompt = f"""
Your task is to convert the pytorch code into a Triton kernel, the code should be runnable and the output should be the same as the pytorch code. Use the `run_triton_code_and_compare` tool to check if the code is correct.

Here it's a best practice on writing Triton kernels:
{triton_cookbook}

Also return reasoning step by step on how the conversion to triton should be done for this specific function. Apply the best practices from the cookbook.

## Example conversion from pytorch to triton

Pytorch code input: 
```python
def relu(x: torch.Tensor) -> torch.Tensor:
    # x: FloatTensor[N, M]
    return torch.maximum(x, torch.zeros_like(x))
```

Expected Output:
```python
import torch
import triton
import triton.language as tl

@triton.jit
def triton_relu_kernel(
    X_ptr,         # pointer to the input float buffer
    Y_ptr,         # pointer to the output float buffer
    numel,         # total number of elements = n * m
    BLOCK_SIZE: tl.constexpr  # compile‐time block size
):
    pid   = tl.program_id(0)
    offs  = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask  = offs < numel
    x_vals = tl.load(X_ptr + offs, mask=mask, other=0.0)
    y_vals = tl.maximum(x_vals, 0.0)
    tl.store(Y_ptr + offs, y_vals, mask=mask)

def relu(x: torch.Tensor, BLOCK_SIZE: int = 1024) -> torch.Tensor:
    n, m   = x.shape
    numel  = n * m
    y      = torch.empty_like(x)
    grid   = ((numel + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    triton_relu_kernel[grid](
        x.data_ptr(), 
        y.data_ptr(),
        numel, 
        BLOCK_SIZE
    )
    return y
```

Why this version is "good":
- Single "numel" argument instead of confusing stride_row, stride_col.
- Mask is offs < numel, which correctly covers all n×m elements.
- All loads/stores use mask, so partial blocks at the end won't run out of bounds.
- It's clear that tl.maximum(x_vals, 0.0) implements ReLU.
"""

triton_generation_user_prompt = """Convert the following pytorch code into a Triton kernel.

Pytorch code:
```python
{pt_code}
```

The entrypoint function must be named: {entrypoint}
The Triton kernel implementation (called by the entrypoint) must be named: {entrypoint}_kernel

No computation logic should be done within the entrypoint function. All computation logic should be done within the Triton kernel implementation.

You must use the `run_triton_code_and_compare` tool to check if the code is correct.
"""

pt_agent = Agent[UnifiedExecutionContext](
    name="PyTorchAgent", 
    model="o4-mini",
    instructions=pytorch_generation_system_prompt,
    tools=[run_pytorch_code_and_tests],
    output_type=PytorchOutput
)

triton_agent = Agent[UnifiedExecutionContext](
    name="TritonAgent", 
    model="o4-mini",
    instructions=triton_generation_system_prompt,
    tools=[run_triton_code_and_compare],
    output_type=TritonOutput
)

console.rule("[bold blue]Processing dataset[/bold blue]")

def unpack_row(row: dict):
    """Unpack all nested dicts into a flat dict"""
    unpacked_data = {}
    for key, value in row.items():
        if isinstance(value, dict):
            unpacked_data.update(unpack_row(value))
        else:
            unpacked_data[key] = value
    return unpacked_data

@weave.op
async def generate_row(max_turns: int):
    # Create unified context
    context = UnifiedExecutionContext()
    
    # Generate function description
    function_name_desc = await generate_function_name_and_description(input_ds)
    context.function_name = function_name_desc.function_name
    context.function_description = function_name_desc.function_description
    
    try:
        # Run PyTorch agent
        pt_result = await Runner.run(
            starting_agent=pt_agent,
            input=pytorch_generation_user_prompt.format(
                function_name=context.function_name,
                function_description=context.function_description
            ),
            context=context,
            max_turns=max_turns
        )
        
        # Store PyTorch results in context
        context.pt_code = pt_result.final_output.pt_code
        context.pt_entrypoint = pt_result.final_output.pt_entrypoint
        context.tests = pt_result.final_output.tests
        
        # Run Triton agent with same context
        triton_result = await Runner.run(
            starting_agent=triton_agent,
            input=triton_generation_user_prompt.format(
                pt_code=context.pt_code,
                entrypoint=context.pt_entrypoint
            ),
            context=context,
            max_turns=max_turns
        )
        
        # Start with the flat context data
        row_data = context.to_flat_dict()
        
        # Add final output data (might contain nested dicts)
        row_data.update(pt_result.final_output.model_dump())
        row_data.update(triton_result.final_output.model_dump())
        
        # Ensure the final result is completely flat
        return unpack_row(row_data)
        
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        return None

@weave.op
async def generate_rows(n_rows: int, max_turns: int):
    tasks = []
    for _ in range(n_rows):
        tasks.append(generate_row(max_turns=max_turns))
    pds_list = await asyncio.gather(*tasks)
    pds_list = [pds for pds in pds_list if pds is not None]
    return pds_list

new_rows = asyncio.run(generate_rows(n_rows=3, max_turns=args.max_turns))

pds = Dataset.from_list(new_rows)
pds.save_to_disk(args.output_dataset.replace("/", "_"))

if args.push:
    input_ds.push_to_hub(args.output_dataset)
