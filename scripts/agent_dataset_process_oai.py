import asyncio
from dataclasses import dataclass, field
from pathlib import Path
import random
from typing import Any

from datasets import Dataset, load_dataset, load_from_disk
from pydantic import BaseModel, Field
from rich.console import Console
import simple_parsing as sp
import weave
import openai

from agents import Agent, Runner, RunContextWrapper, function_tool, RunHooks

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

def load_ds(dataset_name):
    if "/" in dataset_name:
        return load_dataset(dataset_name)["train"]
    else:
        return load_from_disk(dataset_name)["train"]

console.rule(f"[bold blue]Loading dataset: {args.input_dataset}[/bold blue]")

# input_ds = load_ds(args.input_dataset)
input_ds = [
    {"entrypoint": "softmax", "description": "Computes the softmax of a tensor. Interesting for Triton due to its common use in neural networks and potential for fused operations (e.g., with log or scaling) to improve memory bandwidth and reduce kernel launches."},
    {"entrypoint": "layer_norm", "description": "Applies Layer Normalization over a mini-batch of inputs. This is a good candidate for Triton as it involves multiple element-wise operations (mean, variance, normalization) and reductions that can be fused into a single kernel, significantly improving performance by reducing memory I/O and kernel launch overhead."},
    {"entrypoint": "fused_attention", "description": "Implements a fused attention mechanism, such as scaled dot-product attention. This is highly interesting for Triton as it combines multiple computationally intensive operations (matrix multiplications, softmax, dropout, masking) that benefit greatly from kernel fusion, optimized memory access patterns, and tiling strategies to maximize GPU utilization."},
    {"entrypoint": "silu_and_mul", "description": "Computes SiLU (Sigmoid Linear Unit) activation (x * sigmoid(x)) and then multiplies the result with another tensor (gate). This pattern, often found in models like LLaMA (SwiGLU), is interesting for Triton because fusing these element-wise operations (sigmoid, two multiplications) can reduce memory bandwidth usage and kernel launch overhead."},
    {"entrypoint": "rope_embedding", "description": "Applies Rotary Position Embedding (RoPE) to input tensors, a common technique in modern transformers. This is interesting for Triton as it involves complex-number-like manipulations, trigonometric functions, and specific slicing/reshaping operations that can be efficiently implemented in a custom kernel to optimize data movement and computation on the GPU."},
    {"entrypoint": "conv1d_relu", "description": "Performs a 1D convolution followed by a ReLU activation. Fusing these two operations in Triton is beneficial as it avoids writing the intermediate convolution output to global memory and then reading it back for the ReLU, thus saving memory bandwidth and reducing latency."},
    {"entrypoint": "group_norm", "description": "Applies Group Normalization over a mini-batch of inputs. Similar to LayerNorm, but with grouping, it involves reductions and element-wise ops. Triton can optimize this by fusing these steps and handling the group-wise calculations efficiently, which can be complex for generic library implementations to optimize perfectly."}
]

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
        text_format=FunctionNameAndDescription,
    )
    return response.output_parsed

############################################

class ExecutionOutput(BaseModel):
    returncode: int = Field(default=-1, description="The return code of the execution")
    stdout: str = Field(default="", description="The standard output of the execution")
    stderr: str = Field(default="", description="The standard error of the execution")

@dataclass
class CodeExecutionContext:
    """Context to store execution outputs locally without sending to LLM"""
    outputs: ExecutionOutput = field(default_factory=ExecutionOutput)
    
    def store_result(self, result: ExecutionOutput) -> None:
        """Store execution result"""
        self.outputs = result
    
    def get_execution_summary(self) -> dict[str, Any]:
        """Get a summary of execution for LLM"""
        if not self.outputs:
            return {"runs": False, "error_summary": "No execution attempted"}
        
        return {
            "runs": self.outputs.returncode == 0,
            "has_output": bool(self.outputs.stdout.strip()),
            "has_error": bool(self.outputs.stderr.strip()),
            "error_summary": self.outputs.stderr[:200] + "..." if len(self.outputs.stderr) > 200 else self.outputs.stderr
        }
    
@dataclass
class T2TContext:
    pt_ctx: CodeExecutionContext = field(default_factory=CodeExecutionContext)
    triton_ctx: CodeExecutionContext = field(default_factory=CodeExecutionContext)
    tests: str = Field(description="The tests to run")
    match_results: bool = False

@function_tool
async def run_code_and_tests(wrapper: RunContextWrapper[CodeExecutionContext], code: str, tests: str) -> str:
    """Run the code and tests, store results in context.
    Args:
        code: The code to run.
        tests: The tests to run.
    Returns:
        A summary of the execution for the LLM.
    """
    code_and_tests = f"{code}\n\n############import torch\ntorch.set_printoptions(threshold=int(1e9))\n\n{tests}"
    result = run_python_code_on_gpu(code_and_tests)
    wrapper.context.store_result(ExecutionOutput.model_validate(result))
    
    summary = wrapper.context.get_execution_summary()
    if summary["runs"]:
        return f"Code executed successfully. Has output: {summary['has_output']}"
    else:
        return f"Code failed. Error: {summary['error_summary']}"

@function_tool
async def run_triton_code_and_compare(wrapper: RunContextWrapper[T2TContext], triton_code: str) -> str:
    """Run the triton code and compare the output to the expected output.
    Args:
        triton_code: The triton code to run.
    Returns:
        A message indicating which test cases pass/fail.
    """
    triton_code_and_tests = f"{triton_code}\n\n############import torch\ntorch.set_printoptions(threshold=int(1e9))\n\n{wrapper.context.tests}"
    result = run_python_code_on_gpu(triton_code_and_tests)
    triton_output = wrapper.context.outputs.stdout
    wrapper.context.triton_ctx.store_result(ExecutionOutput.model_validate(result))
    pt_sdout = wrapper.context.pt_ctx.outputs.expected_stdout
    
    summary = wrapper.context.triton_ctx.get_execution_summary()
    if summary["runs"]:
        match_results = compare_outputs(pt_sdout, triton_output)
        for name, status, msg, _ in match_results:
            return "Test Results:\n" + "\n".join([f"{name}: {status} ({msg})"])
    else:
        return f"Code failed. Error: {summary['error_summary']}"

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
    conversion_reasoning: str = Field(description="The reasoning step by step on how the conversion to triton should be done for this specific function")
    triton_code: str = Field(description="The Triton code for the function, no tests are needed, just the triton code")
    triton_entrypoint: str = Field(description="The entrypoint of the function in Triton")
    triton_runs: bool = Field(description="Whether the triton code runs or not.")
    triton_has_output: bool = Field(description="Whether the triton code produced output.")
    triton_error_summary: str = Field(default="", description="Brief summary of any triton errors.")
    triton_is_correct: bool = Field(description="Whether the triton code is correct or not.")

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

You must use the run_code_and_tests tool to run your code and tests. You should return a working pytorch implementation of the function with the tests.
"""

pytorch_generation_user_prompt = """
Generate the pytorch code and tests for the function: {function_name}
description: {function_description}
"""

triton_generation_system_prompt = """
Your task is to convert the pytorch code into a Triton kernel.

Here it's a best practice on writing Triton kernels: {triton_cookbook}

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
"""

triton_generation_user_prompt = """Convert the following pytorch code into a Triton kernel.

Pytorch code:
```python
{pt_code}
```

The entrypoint function must be named: {entrypoint}
The Triton kernel implementation (called by the entrypoint) must be named: {entrypoint}_kernel

No computation logic should be done within the entrypoint function. All computation logic should be done within the Triton kernel implementation.
"""

pt_agent = Agent[CodeExecutionContext](
    name="PyTorchAgent", 
    model="o4-mini",
    instructions=pytorch_generation_system_prompt,
    tools=[run_code_and_tests],
    output_type=PytorchOutput)

triton_agent = Agent[CodeExecutionContext](
    name="TritonAgent", 
    model="o4-mini",
    instructions=triton_generation_system_prompt,
    tools=[run_triton_code_and_compare],
    output_type=TritonOutput)

console.rule("[bold blue]Processing dataset[/bold blue]")

def unpack_row(row: dict):
    "unpack all nested dicts into a flat dict"
    unpacked_data = {}
    for key, value in row.items():
        if isinstance(value, dict):
            unpacked_data.update(unpack_row(value))
        else:
            unpacked_data[key] = value
    return unpacked_data

@weave.op
async def generate_row(max_turns: int):
    pt_execution_context = CodeExecutionContext()    
    try:    
        function_name, function_description = await generate_function_name_and_description(input_ds)

        pt_result = await Runner.run(
            starting_agent=pt_agent, 
            input=pytorch_generation_user_prompt.format(
                function_name=function_name, 
                function_description=function_description),
            context=pt_execution_context,
            max_turns=max_turns
        )

        triton_execution_context = T2TContext(
            pt_ctx=pt_execution_context,
            tests=pt_result.final_output.tests
        )

        triton_result = await Runner.run(
            starting_agent=triton_agent, 
            input=triton_generation_user_prompt.format(
                pt_code=pt_result.final_output.pt_code,
                entrypoint=pt_result.final_output.pt_entrypoint),
            context=triton_execution_context,
            max_turns=max_turns
        )
        
        # Get the basic row data from the agent output
        row_data = unpack_row(pt_result.final_output.model_dump())
        
        print (row_data)
        # return final_row
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

# console.print(new_rows[-1])

pds = Dataset.from_list(new_rows)
pds.save_to_disk(args.output_dataset.replace("/", "_"))

if args.push:
    input_ds.push_to_hub(args.output_dataset)
