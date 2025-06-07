import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from enum import Enum

from datasets import Dataset, load_dataset, load_from_disk
from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
import simple_parsing as sp
import weave
import openai

from agents import Agent, Runner, RunContextWrapper, function_tool
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX

from triton_eval.agents.tools import run_python_code_on_gpu
from triton_eval.utils import compare_outputs

console = Console()

client = openai.AsyncOpenAI()

@dataclass
class Args:
    debug: bool = False
    input_dataset: str = "tcapelle/boostrap_oai"
    output_dataset: str = "tcapelle/boostrap_oai"
    weave_project: str = "grpo-cuda/dataset_agent_oai"
    push: bool = False
    num_proc: int = 10
    max_turns: int = 20
    verbose: bool = False
    init: bool = False
    data_path: Path = Path("./data")
    n_rows: int = 3
    entrypoint: str = "pt_entrypoint"
    description: str = "function_description"
    batch_size: int = 10

args = sp.parse(Args)

def console_print(text, verbose_only=args.verbose, **kwargs):
    """Helper function to handle verbose printing"""
    if not verbose_only or args.verbose:
        console.print(text, **kwargs)

def load_ds(dataset_name, init=True):
    if init:
        return load_dataset("json", data_files=str(args.data_path / "simple_samples.jsonl"))["train"]
    elif "/" in dataset_name:
        return load_dataset(dataset_name)["train"]
    else:
        return load_from_disk(dataset_name)

console.print(Panel.fit(f"[bold blue]Loading dataset: {args.input_dataset}[/bold blue]", border_style="blue"))

input_ds = load_ds(args.input_dataset, init=args.init)

console.print("[bold blue]Input dataset[/bold blue]")
console.print(input_ds)

console.print(Panel.fit("[bold blue]Fixing code with Agent[/bold blue]", border_style="blue"))

weave.init(args.weave_project)

def get_current_cookbook() -> str:
    """Get the current cookbook content, used to reload after updates"""
    return (args.data_path / "triton_cookbook.md").read_text()

############################################

# ERROR REFLECTION SYSTEM
# 
# This system automatically learns from Triton conversion errors and improves the cookbook:
# 1. During generation, all errors are collected and stored in collected_errors list
# 2. After all rows are processed, the ErrorReflectionAgent analyzes ALL errors together
# 3. It reviews the full cookbook and generates an improved version
# 4. The updated cookbook incorporates lessons learned from all failures
# 
# This creates a holistic improvement system that learns from patterns across all errors.
#
# WORKFLOW:
# - Individual errors → collected_errors list (during generate_row)
# - Batch analysis → analyze_errors_and_improve_cookbook() (after all rows)
# - Pattern recognition → ErrorReflectionAgent analyzes all errors together
# - Cookbook replacement → Full cookbook updated with improvements
# - Future runs benefit from accumulated knowledge across all errors

############################################

# Global error collection for batch analysis
collected_errors = []

# Row counter for init mode
current_row_index = 0

def reset_global_state():
    """Reset global state for clean execution"""
    global collected_errors, current_row_index
    collected_errors.clear()
    current_row_index = 0

def recreate_triton_agent():
    """Recreate triton agent with updated cookbook content"""
    global triton_agent
    triton_agent = create_triton_agent()
    console_print("[blue]🔄 Triton agent recreated with updated cookbook[/blue]")

class ErrorContext(BaseModel):
    function_name: str
    function_description: str
    pt_code: str
    triton_code: str
    triton_error_summary: str
    triton_stderr: str
    conversion_reasoning: str = ""

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
    return "\n".join([f"{row[args.entrypoint]}: {row[args.description]}" for row in ds])

async def generate_function_name_and_description(input_ds):
    current_ds_rows = dump_ds(input_ds)
    messages = [
        {"role": "system", "content": creativity_system_prompt},
        {"role": "user", "content": creativity_user_prompt.format(dataset=current_ds_rows)}
    ]
    response = await client.responses.parse(
        model="gpt-4.1",
        input=messages,
        temperature=1.5,
        text_format=FunctionNameAndDescription,
    )
    return response.output_parsed

############################################

class ExecutionType(str, Enum):
    PYTORCH = "pytorch"
    TRITON = "triton"

class ExecutionContext(BaseModel):
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
        
        # Store brief error summary (first few lines) for output agents, not full stderr
        stderr = result.get("stderr", "")
        if stderr.strip():
            # Take first 2 lines or first 200 chars of stderr as brief summary
            error_lines = stderr.strip().split('\n')[:2]
            brief_error = '\n'.join(error_lines)
            if len(brief_error) > 200:
                brief_error = brief_error[:200] + "..."
            setattr(self, f"{prefix}_error_summary", brief_error)
        else:
            setattr(self, f"{prefix}_error_summary", "")
    
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

@weave.op
async def run_code(
    wrapper: RunContextWrapper[ExecutionContext], 
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
        return f"""Code failed with error:\n{summary['error_summary']}.
        Reflect on your previous attempts at fixing the error, then try fixing the error."""

@function_tool
async def run_pytorch_code_and_tests(
    wrapper: RunContextWrapper[ExecutionContext], 
    code: str, 
    tests: str
) -> str:
    """Run PyTorch code and tests"""
    # Store the code and tests in context
    wrapper.context.pt_code = code
    wrapper.context.tests = tests
    
    return await run_code(wrapper, code, ExecutionType.PYTORCH, tests)

async def compare_pytorch_triton_outputs(wrapper: RunContextWrapper[ExecutionContext]) -> str:
    """Compare PyTorch and Triton outputs"""
    ctx = wrapper.context
    
    if not ctx.pt_runs or not ctx.triton_runs:
        return "Cannot compare - one or both implementations failed to run"
    
    match_results = compare_outputs(ctx.pt_stdout, ctx.triton_stdout)
    ctx.triton_is_correct = all(status == "PASS" for name, status, msg, _ in match_results)
    
    results_str = "\n".join([f"{name}: {status} ({msg})" for name, status, msg, _ in match_results])
    return f"Test Results:\n{results_str}"

@function_tool
async def update_cookbook_with_error_knowledge(
    wrapper: RunContextWrapper[ExecutionContext],
    improved_cookbook: str
) -> str:
    """Update the Triton cookbook with the complete improved version"""
    cookbook_path = args.data_path / "triton_cookbook.md"
    
    # Write the new complete cookbook
    cookbook_path.write_text(improved_cookbook)
    
    return f"Successfully updated cookbook with improved version at {cookbook_path}"

@function_tool
async def run_triton_code_and_compare(
    wrapper: RunContextWrapper[ExecutionContext], 
    triton_code: str
) -> str:
    """Run Triton code and compare with PyTorch output"""
    # Store the triton code in context
    wrapper.context.triton_code = triton_code
    
    # First run the triton code
    result = await run_code(wrapper, triton_code, ExecutionType.TRITON, wrapper.context.tests)
    
    # If triton code failed, return the error instead of trying to compare
    if "Code failed with error" in result:
        return result
    
    # Then compare outputs
    return await compare_pytorch_triton_outputs(wrapper)

### First Agent: Generate PyTorch/Triton pairs
triton_cookbook = get_current_cookbook()


class PytorchOutput(BaseModel):
    pt_code: str = Field(description="The PyTorch code for the function")
    tests: str = Field(description="The test code for the function")
    pt_entrypoint: str = Field(description="The entrypoint/main function name in the PyTorch code")

class TritonOutput(BaseModel):
    triton_code: str = Field(description="The Triton code for the function")
    conversion_reasoning: str = Field(description="The step-by-step reasoning for converting PyTorch to Triton")
    triton_entrypoint: str = Field(description="The entrypoint/main function name in the Triton code")

pytorch_generation_system_prompt = f"""{RECOMMENDED_PROMPT_PREFIX}

We are generating a PyTorch/Triton pairs dataset. We want functions that have exactly the same functionalities.

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

You must use the run_pytorch_code_and_tests tool to run your code and tests. Once you have working pytorch code and tests, you MUST transfer to the PyTorch Output Agent to format the results. Do not provide any final response yourself - always transfer to the output agent.
"""

pytorch_output_system_prompt = """You are a specialized agent for extracting PyTorch code information into structured output.

Your task is to extract the following from the context:
1. pt_code: The final working PyTorch code
2. tests: The test code  
3. pt_entrypoint: The main function name (not the test function)

For the entrypoint, identify the main function that implements the actual operation. For example, if the code defines:
```python
def add(input: torch.Tensor, other: torch.Tensor, alpha: float=1):
    return torch.add(input, other, alpha=alpha)

def test_add():
    # test code
```

The entrypoint should be "add" (not "test_add").
"""

triton_generation_system_prompt = f"""{RECOMMENDED_PROMPT_PREFIX}

Your task is to convert the pytorch code into a Triton kernel, the code should be runnable and the output should be the same as the pytorch code. Use the `run_triton_code_and_compare` tool to check if the code is correct.

Here it's a best practice on writing Triton kernels:
{{triton_cookbook}}

Also provide reasoning step by step on how the conversion to triton should be done for this specific function. Apply the best practices from the cookbook.

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

Once you have working triton code that matches the pytorch output, you MUST transfer to the Triton Output Agent to format the results. Do not provide any final response yourself - always transfer to the output agent.
"""

triton_output_system_prompt = """You are a specialized agent for extracting Triton code information into structured output.

Your task is to extract the following:
1. triton_code: The final working Triton code
2. conversion_reasoning: The step-by-step reasoning for how the PyTorch to Triton conversion was done
3. triton_entrypoint: The main function name in the Triton code (not the kernel)

Look at the conversation history to find the reasoning that was provided during the conversion process.

For the entrypoint, look for the main callable function (not the @triton.jit kernel). For example:
- If there's `relu_kernel` (the kernel) and `relu` (the wrapper), the entrypoint is "relu"
- The entrypoint should match the original PyTorch function name
"""

pytorch_generation_user_prompt = """
Generate the pytorch code and tests for the function: {function_name}
description: {function_description}
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

class CookbookUpdate(BaseModel):
    analysis_summary: str = Field(description="Overall analysis of all the collected errors and patterns found")
    improved_cookbook: str = Field(description="The complete improved cookbook text incorporating lessons learned from all errors")
    should_update: bool = Field(description="Whether the cookbook should be updated based on the error analysis")
    key_improvements: list[str] = Field(description="List of key improvements made to the cookbook")

error_reflection_system_prompt = """You are an expert Triton kernel developer and cookbook improvement specialist. Your job is to analyze ALL collected Triton conversion errors from a batch generation process and create an improved version of the entire Triton cookbook.

You will receive:
1. The current complete Triton cookbook
2. A collection of ALL errors that occurred during the generation process
3. Context about each failed conversion (PyTorch code, attempted Triton code, error messages)

Your task is to:
1. **Analyze Error Patterns**: Look across all errors to identify common patterns, recurring issues, and systematic problems
2. **Extract Meta-Lessons**: Find higher-level insights about what makes Triton conversions fail
3. **Improve the Cookbook**: Generate a new, improved version of the COMPLETE cookbook that addresses these issues
4. **Preserve Good Content**: Keep all valuable existing content while adding new guidance

Focus on:
- Systematic patterns across multiple failures (not individual bugs)
- Common misconceptions about Triton that lead to errors
- Missing guidance in the current cookbook
- Better examples or clearer explanations needed
- New anti-patterns or best practices discovered

The improved cookbook should be:
- Complete and self-contained (full replacement of the original)
- More comprehensive based on real failure patterns
- Clearer in areas where errors were common
- Enhanced with new examples from the error cases
- Include code examples when possible with the knowledge you acquired to illustrate the new best practices

Only recommend updating the cookbook if there are meaningful patterns worth addressing across multiple errors."""

error_reflection_user_prompt = """Analyze these collected Triton conversion errors and improve the cookbook:

**Current Cookbook:**
```markdown
{current_cookbook}
```

**Collected Errors ({num_errors} total):**

{error_details}

Please analyze these errors for patterns and create an improved version of the complete cookbook that addresses the systematic issues you identified."""


# Create the output agents first
pt_output_agent = Agent[ExecutionContext](
    name="PyTorchOutputAgent",
    handoff_description="Specialist agent for extracting and formatting PyTorch code into structured output",
    model="o4-mini", 
    instructions=pytorch_output_system_prompt,
    output_type=PytorchOutput
)

triton_output_agent = Agent[ExecutionContext](
    name="TritonOutputAgent",
    handoff_description="Specialist agent for extracting and formatting Triton code into structured output with reasoning",
    model="o4-mini",
    instructions=triton_output_system_prompt, 
    output_type=TritonOutput
)

# Create the error reflection agent
error_reflection_agent = Agent[ExecutionContext](
    name="ErrorReflectionAgent",
    handoff_description="Specialist agent for analyzing Triton conversion errors and improving the cookbook",
    model="o3",
    instructions=error_reflection_system_prompt,
    tools=[update_cookbook_with_error_knowledge],
    output_type=CookbookUpdate
)

# Create the main agents with handoffs (no output_type)
pt_agent = Agent[ExecutionContext](
    name="PyTorchAgent", 
    model="o4-mini",
    instructions=pytorch_generation_system_prompt,
    tools=[run_pytorch_code_and_tests],
    handoffs=[pt_output_agent]
)

def create_triton_agent():
    """Create triton agent with current cookbook content"""
    current_cookbook = get_current_cookbook()
    return Agent[ExecutionContext](
        name="TritonAgent", 
        model="o4-mini",
        instructions=triton_generation_system_prompt.format(triton_cookbook=current_cookbook),
        tools=[run_triton_code_and_compare],
        handoffs=[triton_output_agent]
    )

# Initially create the triton agent
triton_agent = create_triton_agent()

console.print(Panel.fit("[bold blue]Processing dataset[/bold blue]", border_style="blue"))

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
async def generate_row(max_turns: int, function_name: str = None, function_description: str = None):
    global current_row_index
    
    # Create unified context
    context = ExecutionContext()
    
    if args.init:
        # Use existing sample from the dataset
        if current_row_index >= len(input_ds):
            raise IndexError(f"Row index {current_row_index} out of range for dataset with {len(input_ds)} samples")
        
        sample = input_ds[current_row_index]
        context.function_name = sample[args.entrypoint]
        context.function_description = sample[args.description]
        
        # Update pt_code in context if available
        if 'pt_code' in sample:
            context.pt_code = sample['pt_code']
        
        current_row_index += 1
        
        console_print(Panel(
            f"[bold cyan]🔄 Using existing sample: {context.function_name}[/bold cyan]\n"
            f"[dim]{context.function_description}[/dim]",
            title="Using Existing Sample",
            border_style="cyan"
        ))
    else:
        # Use pre-generated function info
        context.function_name = function_name
        context.function_description = function_description
        
        console_print(Panel(
            f"[bold cyan]🚀 Generating: {context.function_name}[/bold cyan]\n"
            f"[dim]{context.function_description}[/dim]",
            title="Function Generation",
            border_style="cyan"
        ))
    
    try:
        # Run PyTorch agent (will handoff to pt_output_agent to extract code, tests, and entrypoint)
        console_print(f"[yellow]📝 Running PyTorch agent...[/yellow]")
            
        pt_result = await Runner.run(
            starting_agent=pt_agent,
            input=pytorch_generation_user_prompt.format(
                function_name=context.function_name,
                function_description=context.function_description
            ),
            context=context,
            max_turns=max_turns
        )
        
        # Extract PyTorch code info from structured output
        pt_output = pt_result.final_output
        
        # Debug: Check what we got
        pt_analysis = f"Type: {type(pt_output)}"
        
        if isinstance(pt_output, str):
            pt_analysis += f"\n[red]❌ Got string instead of PytorchOutput[/red]"
            pt_analysis += f"\n[dim]Preview: {pt_output[:100]}{'...' if len(pt_output) > 100 else ''}[/dim]"
            # Fallback: extract from context instead
            pt_code = context.pt_code
            tests = context.tests  
            pt_entrypoint = context.function_name  # fallback to function name
        else:
            pt_analysis += f"\n[green]✅ Got PytorchOutput successfully[/green]"
            pt_analysis += f"\nEntrypoint: {pt_output.pt_entrypoint}"
            pt_analysis += f"\nCode length: {len(pt_output.pt_code)} chars"
            pt_analysis += f"\nTests length: {len(pt_output.tests)} chars"
            pt_code = pt_output.pt_code
            tests = pt_output.tests
            pt_entrypoint = pt_output.pt_entrypoint
        
        console_print(Panel(pt_analysis, title="🔍 PyTorch Analysis", border_style="yellow"))
        
        # Run Triton agent (will handoff to triton_output_agent to extract code, reasoning, and entrypoint)
        console_print(f"[blue]⚡ Running Triton agent...[/blue]")
            
        triton_result = await Runner.run(
            starting_agent=triton_agent,
            input=triton_generation_user_prompt.format(
                pt_code=pt_code,
                entrypoint=pt_entrypoint
            ),
            context=context,
            max_turns=max_turns
        )
        
        # Extract Triton code info from structured output
        triton_output = triton_result.final_output
        
        # Debug: Check what we got
        triton_analysis = f"Type: {type(triton_output)}"
        
        if isinstance(triton_output, str):
            triton_analysis += f"\n[red]❌ Got string instead of TritonOutput[/red]"
            triton_analysis += f"\n[dim]Preview: {triton_output[:100]}{'...' if len(triton_output) > 100 else ''}[/dim]"
            # Fallback: extract from context instead
            triton_code = context.triton_code
            conversion_reasoning = "Conversion reasoning not captured due to parsing issue"
            triton_entrypoint = pt_entrypoint  # fallback to same as pytorch
        else:
            triton_analysis += f"\n[green]✅ Got TritonOutput successfully[/green]"
            triton_analysis += f"\nEntrypoint: {triton_output.triton_entrypoint}"
            triton_analysis += f"\nCode length: {len(triton_output.triton_code)} chars"
            triton_analysis += f"\nReasoning length: {len(triton_output.conversion_reasoning)} chars"
            triton_code = triton_output.triton_code
            conversion_reasoning = triton_output.conversion_reasoning
            triton_entrypoint = triton_output.triton_entrypoint
        
        console_print(Panel(triton_analysis, title="🔍 Triton Analysis", border_style="blue"))
        
        # Collect error information for batch analysis if triton failed
        if not context.triton_runs and context.triton_stderr:
            error_info = ErrorContext(
                function_name=context.function_name,
                function_description=context.function_description,
                pt_code=pt_code,
                triton_code=triton_code,
                triton_error_summary=context.triton_error_summary,
                triton_stderr=context.triton_stderr,
                conversion_reasoning=conversion_reasoning
            )
            collected_errors.append(error_info)
            console_print(f"[yellow]📝 Collected error for batch analysis ({len(collected_errors)} total)[/yellow]")
        
        # Show execution summary
        summary = f"PyTorch runs: {'✅' if context.pt_runs else '❌'}"
        summary += f"\nTriton runs: {'✅' if context.triton_runs else '❌'}"
        summary += f"\nTriton correct: {'✅' if context.triton_is_correct else '❌'}"
        if context.pt_stderr:
            summary += f"\nPyTorch errors: {len(context.pt_stderr)} chars"
        if context.triton_stderr:
            summary += f"\nTriton errors: {len(context.triton_stderr)} chars"
        
        console_print(Panel(summary, title="📊 Execution Summary", border_style="magenta"))
        
        # Start with the flat context data (contains all execution details: stdout, stderr, runs, etc.)
        row_data = context.to_flat_dict()
        
        # Add the structured output data manually
        row_data.update({
            "pt_code": pt_code,
            "tests": tests,
            "pt_entrypoint": pt_entrypoint,
            "triton_code": triton_code,
            "conversion_reasoning": conversion_reasoning,
            "triton_entrypoint": triton_entrypoint
        })
        
        console_print(Panel(
            f"[green]🎉 Successfully generated row for {context.function_name}[/green]",
            title="Success",
            border_style="green"
        ))
        
        # Ensure the final result is completely flat
        return unpack_row(row_data)
        
    except Exception as e:
        error_msg = f"Error generating row for {context.function_name}: {str(e)}"
        console_print(Panel(f"[bold red]💥 {error_msg}[/bold red]", title="Error", border_style="red"))
        console_print(f"[red]❌ {error_msg}[/red]", verbose_only=False)  # Always show brief error
        
        if args.verbose:
            import traceback
            console_print(f"[red]Traceback: {traceback.format_exc()}[/red]")
        
        raise e
        return None

@weave.op
async def generate_new_functions_parallel(n_rows: int, max_turns: int):
    """Generate new functions sequentially, then run code generation in parallel"""
    console_print(f"[blue]🎲 Generating {n_rows} new random functions[/blue]")
    
    # First, generate all function names/descriptions sequentially to avoid repetition
    console_print(f"[yellow]📝 Generating {n_rows} unique function names/descriptions...[/yellow]")
    function_infos = []
    for i in range(n_rows):
        console_print(f"[dim]Generating function {i+1}/{n_rows}...[/dim]")
        function_info = await generate_function_name_and_description(input_ds)
        function_infos.append(function_info)
        console_print(f"[green]✅ Generated: {function_info.function_name}[/green]")
    
    # Now run the actual row generation in parallel with pre-generated function info
    console_print(f"[blue]⚡ Running parallel code generation for {n_rows} functions...[/blue]")
    tasks = []
    for i, function_info in enumerate(function_infos):
        console_print(f"[dim]Starting code generation for row {i+1}/{n_rows}: {function_info.function_name}...[/dim]")
        tasks.append(generate_row(
            max_turns=max_turns,
            function_name=function_info.function_name,
            function_description=function_info.function_description
        ))
    
    pds_list = await asyncio.gather(*tasks, return_exceptions=True)
    return pds_list

@weave.op
async def generate_rows(n_rows: int, max_turns: int):
    global current_row_index
    current_row_index = 0  # Reset counter
    
    console_print(Panel(
        f"[bold blue]🔄 Generating {n_rows} rows with max {max_turns} turns each[/bold blue]",
        title="Processing Configuration",
        border_style="blue"
    ))
    
    if args.init:
        # Limit to available samples
        available_samples = len(input_ds)
        if n_rows > available_samples:
            console_print(f"[yellow]⚠️ Requested {n_rows} rows but only {available_samples} samples available. Using all available samples.[/yellow]")
            n_rows = available_samples
        
        console_print(f"[blue]📋 Using existing dataset samples[/blue]")
        
        # For init mode, just create tasks directly
        tasks = []
        for i in range(n_rows):
            console_print(f"[dim]Starting row {i+1}/{n_rows}...[/dim]")
            tasks.append(generate_row(max_turns=max_turns))
        
        pds_list = await asyncio.gather(*tasks, return_exceptions=True)
    else:
        pds_list = await generate_new_functions_parallel(n_rows, max_turns)
    
    # Count successes and failures
    successes = [pds for pds in pds_list if pds is not None and not isinstance(pds, Exception)]
    failures = [pds for pds in pds_list if pds is None or isinstance(pds, Exception)]
    
    # Results summary
    mode_desc = "existing samples" if args.init else "generated functions"
    result_text = f"[bold green]✅ Successfully processed: {len(successes)}/{n_rows} {mode_desc}[/bold green]"
    if failures:
        result_text += f"\n[bold red]❌ Failed: {len(failures)}/{n_rows} {mode_desc}[/bold red]"
    
    console_print(Panel(result_text, title="📈 Final Results", border_style="green"))
    
    return successes

@weave.op
async def analyze_errors_and_improve_cookbook():
    """Analyze all collected errors and improve the cookbook if needed"""
    if not collected_errors:
        console_print("[yellow]📚 No errors collected - cookbook analysis skipped[/yellow]")
        return
    
    console_print(Panel(
        f"[bold blue]🔍 Analyzing {len(collected_errors)} collected errors for cookbook improvement[/bold blue]",
        title="Batch Error Analysis",
        border_style="blue"
    ))
    
    # Format error details for the prompt
    error_details_list = []
    for i, error in enumerate(collected_errors, 1):
        error_detail = f"""
**Error {i}: {error.function_name}**
- **Description:** {error.function_description}
- **PyTorch Code:**
```python
{error.pt_code}
```
- **Triton Code (failed):**
```python
{error.triton_code}
```
- **Error Message:** {error.triton_error_summary}
- **Conversion Reasoning:** {error.conversion_reasoning}
- **Stderr:** {error.triton_stderr[:300]}{'...' if len(error.triton_stderr) > 300 else ''}
"""
        error_details_list.append(error_detail)
    
    error_details = "\n".join(error_details_list)
    current_cookbook = get_current_cookbook()
    
    try:
        # Create a temporary context for the error analysis
        analysis_context = ExecutionContext()
        
        # Run error reflection agent with all collected errors
        error_result = await Runner.run(
            starting_agent=error_reflection_agent,
            input=error_reflection_user_prompt.format(
                current_cookbook=current_cookbook,
                num_errors=len(collected_errors),
                error_details=error_details
            ),
            context=analysis_context,
            max_turns=10  # Give more turns for comprehensive analysis
        )
        
        # Process the result
        cookbook_update = error_result.final_output
        
        if isinstance(cookbook_update, CookbookUpdate) and cookbook_update.should_update:
            console_print(Panel(
                f"[green]📚 Cookbook improvement recommended![/green]\n"
                f"[dim]Analysis: {cookbook_update.analysis_summary[:150]}...[/dim]\n"
                f"[bold]Key Improvements:[/bold]\n" + 
                "\n".join([f"• {improvement}" for improvement in cookbook_update.key_improvements[:3]]),
                title="Cookbook Analysis Complete",
                border_style="green"
            ))
            
            # The improved cookbook will be updated via the tool call that was made during agent execution
            console_print(f"[green]✅ Cookbook has been updated with improvements based on {len(collected_errors)} errors[/green]")
            
            # Recreate triton agent with updated cookbook
            recreate_triton_agent()
            
        else:
            console_print(Panel(
                f"[yellow]📚 No cookbook improvement needed[/yellow]\n"
                f"[dim]Analysis: {cookbook_update.analysis_summary[:150]}...[/dim]",
                title="Cookbook Analysis Complete", 
                border_style="yellow"
            ))
            
    except Exception as e:
        console_print(Panel(
            f"[red]❌ Error during batch cookbook analysis: {str(e)}[/red]",
            title="Analysis Error",
            border_style="red"
        ))

@weave.op
async def extend_dataset(n_rows: int, max_turns: int, batch_size: int):
    """Main async function to handle batch processing"""
    # Reset global state for clean execution
    reset_global_state()
    
    # Generate rows in batches and accumulate them
    all_new_rows = []
    total_batches = (n_rows + batch_size - 1) // batch_size  # Ceiling division

    console.print(Panel(
        f"[bold blue]🚀 Processing {n_rows} rows in {total_batches} batches of {batch_size}[/bold blue]",
        title="Batch Processing Configuration",
        border_style="blue"
    ))

    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_rows)
        current_batch_size = end_idx - start_idx
        
        console_print(Panel(
            f"[bold cyan]📦 Processing batch {batch_idx + 1}/{total_batches} ({current_batch_size} rows)[/bold cyan]",
            title=f"Batch {batch_idx + 1}",
            border_style="cyan"
        ))
        
        batch_rows = await generate_rows(n_rows=current_batch_size, max_turns=max_turns)
        all_new_rows.extend(batch_rows)
        
        console_print(f"[green]✅ Batch {batch_idx + 1} completed: {len(batch_rows)} rows generated[/green]")
        console_print(f"[blue]📊 Total rows so far: {len(all_new_rows)}/{n_rows}[/blue]")

    # After generating all rows, analyze errors and improve cookbook
    await analyze_errors_and_improve_cookbook()

    console.print(Panel(
        f"[bold green]🎉 Generated {len(all_new_rows)} new rows total[/bold green]",
        title="Generation Complete",
        border_style="green"
    ))

    # Load existing dataset and append new rows
    try:
        console_print(f"[yellow]📚 Loading existing dataset: {args.input_dataset}[/yellow]")
        existing_ds = load_ds(args.input_dataset, init=False)
        existing_rows = list(existing_ds)
        
        console_print(f"[blue]📋 Existing dataset has {len(existing_rows)} rows[/blue]")
        
    except Exception as e:
        console_print(f"[yellow]⚠️ Could not load existing dataset ({e}), starting fresh[/yellow]")
        existing_rows = []

    # Combine existing and new rows
    combined_rows = existing_rows + all_new_rows
    console_print(f"[green]📊 Combined dataset: {len(existing_rows)} existing + {len(all_new_rows)} new = {len(combined_rows)} total rows[/green]")

    # Create and save the combined dataset
    pds = Dataset.from_list(combined_rows)

    # Save to disk
    output_path = args.output_dataset.replace("/", "_")
    pds.save_to_disk(output_path)
    console_print(f"[green]💾 Saved combined dataset to disk: {output_path}[/green]")

    if args.push:
        console_print(f"[blue]🚀 Pushing to HuggingFace Hub: {args.output_dataset}[/blue]")
        pds.push_to_hub(args.output_dataset)
        console_print(f"[green]✅ Successfully pushed to Hub[/green]")

# Run the main async function
asyncio.run(extend_dataset(n_rows=args.n_rows, max_turns=args.max_turns, batch_size=args.batch_size))

