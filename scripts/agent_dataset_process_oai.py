import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Literal
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

"""
PyTorch to Triton Dataset Generation & Fix Script

This script supports three main modes:

1. **Generation Mode** (default): 
   - Generates new PyTorch functions and converts them to Triton
   - Creates completely new dataset entries
   - Usage: python agent_dataset_process_oai.py --n_rows 50

2. **Init Mode** (--init):
   - Uses existing samples from a simple dataset file
   - Processes pre-defined function names/descriptions  
   - Usage: python agent_dataset_process_oai.py --init --n_rows 20

3. **Fix Mode** (--fix):
   - Loads existing dataset and finds rows where PyTorch runs but Triton conversion failed
   - Re-attempts only the Triton conversion (skips PyTorch generation)
   - Updates the dataset with fixed results
   - Usage: python agent_dataset_process_oai.py --fix --input_dataset my_dataset --batch_size 10

Key Features:
- Batch processing with configurable batch sizes
- Error collection and cookbook improvement after each batch
- Parallel processing within batches
- Automatic dataset saving and optional HuggingFace Hub pushing
- Rich console output with progress tracking

Example commands:
# Generate 100 new rows in batches of 10
python agent_dataset_process_oai.py --n_rows 100 --batch_size 10 --push

# Fix failed Triton conversions in existing dataset  
python agent_dataset_process_oai.py --fix --input_dataset username/my_dataset --batch_size 5

# Use existing samples with verbose output
python agent_dataset_process_oai.py --init --n_rows 50 --verbose --batch_size 20
"""

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
    fix: bool = False
    data_path: Path = Path("./data")
    n_rows: int = 200
    entrypoint: str = "pt_entrypoint"
    description: str = "function_description"
    batch_size: int = 10

args = sp.parse(Args)

# Global placeholder for the reporter instance (populated by DatasetProcessor)
_GLOBAL_CONSOLE_REPORTER: "ConsoleReporter | None" = None

def console_print(text, verbose_only=args.verbose, **kwargs):
    """Helper function that delegates to the active ConsoleReporter (if any)."""
    global _GLOBAL_CONSOLE_REPORTER  # noqa: PLW0603 – needed for global mutation

    reporter = _GLOBAL_CONSOLE_REPORTER
    if reporter is not None:
        reporter.print(text, verbose_only=verbose_only, **kwargs)
    else:
        # Fallback to raw rich.console when no reporter is available (e.g. unit tests)
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

if args.fix:
    console.print(Panel.fit("[bold magenta]Fix Mode: Re-attempting failed Triton conversions[/bold magenta]", border_style="magenta"))
elif args.init:
    console.print(Panel.fit("[bold cyan]Init Mode: Using existing samples[/bold cyan]", border_style="cyan"))
else:
    console.print(Panel.fit("[bold green]Generation Mode: Creating new functions[/bold green]", border_style="green"))

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

# No row index tracking required now that init-mode is gone

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

creativity_system_prompt = """You are an expert in PyTorch and Triton.

# DATASET  
You have a list of (function name, description) pairs that already exist.  
Your task is to invent **one new PyTorch function that is *not* in that list**.

# STRICT CONSTRAINTS  
• Use **no more than 3 primitive PyTorch ops**.  
• Allowed ops:  
  – element-wise arithmetic (add, sub, mul, div)  
  – point-wise activations (relu, gelu, sigmoid, tanh)  
  – small 2-D matmul ≤ 128 × 128  
  – reductions over the last dimension (sum, mean, max)  
  – `conv2d` / `conv_transpose2d` with stride = 1 and padding = 0  
• No `for` / `while` loops or other control-flow; inputs must be contiguous tensors of rank ≤ 4.  
• Do **not** use RNG, autograd hooks, or dynamic shapes.  
• Keep the kernel to a **single warp-synchronous phase** (no multi-stage pipelines).

# OUTPUT FORMAT (return valid JSON)  
```json
{
  "name":        "<function_name_in_snake_case>",
  "description": "<1–2 sentences explaining what the function does>",
  "feasibility": "<1 sentence on why this maps cleanly to Triton>"
}

# GOOD EXAMPLES

fused_matmul_add: matrix multiplication followed immediately by bias add.

transpose_2d: transposes the last two dims of a 2-D tensor.

# BAD EXAMPLE (too complex)

hierarchical_multi_head_attention: combines multiple attention mechanisms and variable sequence lengths.

# THINK STEP-BY-STEP
1. Pick 1–2 allowed primitives
2. fuse them if helpful, (3) craft a concise name, (4) write the description, (5) add a one-sentence feasibility rationale.

Generate exactly one JSON object and nothing else.
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

# ---------------------------------------------------------------------------
# Generation helpers to shrink the massive `generate_row` coroutine
# ---------------------------------------------------------------------------

async def _create_execution_context(
    *,
    max_turns: int,
    existing_row: dict | None = None,
    function_name: str | None = None,
    function_description: str | None = None,
):
    """Prepare an `ExecutionContext` according to the current run‐mode.

    Returns a tuple of (context, pt_code, tests, pt_entrypoint) where the latter
    three are *None* until the PyTorch phase runs (unless we are in fix-mode and
    the row already contains them).
    """
    ctx = ExecutionContext()

    # ----------------------------------------------------
    # 1. Fix-mode – hydrate from existing row
    # ----------------------------------------------------
    if args.fix and existing_row:
        ctx.function_name = existing_row.get("function_name", "")
        ctx.function_description = existing_row.get("function_description", "")

        # Copy PyTorch fields
        for key in (
            "pt_code",
            "tests",
            "pt_entrypoint",
            "pt_runs",
            "pt_stdout",
            "pt_stderr",
            "pt_returncode",
            "pt_has_output",
            "pt_error_summary",
        ):
            setattr(ctx, key, existing_row.get(key, getattr(ctx, key)))

        # Copy Triton fields we might look at later (they will be overwritten)
        for key in (
            "triton_code",
            "triton_entrypoint",
            "triton_runs",
            "triton_stdout",
            "triton_stderr",
            "triton_returncode",
            "triton_has_output",
            "triton_error_summary",
            "triton_is_correct",
        ):
            setattr(ctx, key, existing_row.get(key, getattr(ctx, key)))

        console_print(Panel(
            f"[bold magenta]🔧 Fixing: {ctx.function_name}[/bold magenta]\n"
            f"[dim]{ctx.function_description}[/dim]\n"
            f"[yellow]PyTorch: {'✅ Working' if ctx.pt_runs else '❌ Failed'}[/yellow]\n"
            f"[blue]Triton: {'✅ Correct' if ctx.triton_is_correct else '❌ Needs Fix'}[/blue]",
            title="Fix Mode",
            border_style="magenta",
        ))
        return ctx, ctx.pt_code, ctx.tests, ctx.pt_entrypoint

    # ----------------------------------------------------
    # 3. Fresh generation
    # ----------------------------------------------------
    ctx.function_name = function_name or ""
    ctx.function_description = function_description or ""

    console_print(Panel(
        f"[bold cyan]🚀 Generating: {ctx.function_name}[/bold cyan]\n"
        f"[dim]{ctx.function_description}[/dim]",
        title="Function Generation",
        border_style="cyan",
    ))
    return ctx, None, None, None


async def _run_pytorch_phase(ctx: ExecutionContext, max_turns: int, skip: bool = False):
    """Run the PyTorch agent unless `skip` is True (used by fix-mode)."""
    if skip:
        # We already possess a verified PyTorch implementation.
        return ctx.pt_code, ctx.tests, ctx.pt_entrypoint

    console_print("[yellow]📝 Running PyTorch agent...[/yellow]")
    pt_result = await Runner.run(
        starting_agent=pt_agent,
        input=pytorch_generation_user_prompt.format(
            function_name=ctx.function_name,
            function_description=ctx.function_description,
        ),
        context=ctx,
        max_turns=max_turns,
    )
    pt_output = pt_result.final_output

    # Handle both structured and fallback cases
    if isinstance(pt_output, PytorchOutput):
        console_print(
            Panel(
                f"[green]✅ PyTorch agent succeeded[/green] — entrypoint: {pt_output.pt_entrypoint}",
                title="🔍 PyTorch Analysis",
                border_style="yellow",
            )
        )
        return pt_output.pt_code, pt_output.tests, pt_output.pt_entrypoint
    else:
        preview = str(pt_output)[:100]
        console_print(
            Panel(
                f"[red]❌ Unexpected PyTorch output type ({type(pt_output)}). Using context fallback.\n[/red]Preview: {preview}",
                title="🔍 PyTorch Analysis",
                border_style="yellow",
            )
        )
        return ctx.pt_code, ctx.tests, ctx.function_name  # fallback


async def _run_triton_phase(ctx: ExecutionContext, pt_code: str, pt_entrypoint: str, max_turns: int):
    """Invoke Triton agent and return (triton_code, reasoning, entrypoint, triton_output)."""
    console_print("[blue]⚡ Running Triton agent...[/blue]")

    triton_result = await Runner.run(
        starting_agent=triton_agent,
        input=triton_generation_user_prompt.format(pt_code=pt_code, entrypoint=pt_entrypoint),
        context=ctx,
        max_turns=max_turns,
    )
    return triton_result.final_output


# ---------------------------------------------------------------------------
# END helper section
# ---------------------------------------------------------------------------

async def generate_row(max_turns: int, function_name: str | None = None, function_description: str | None = None, existing_row: dict | None = None):
    """Smaller orchestrator that delegates heavy work to helper functions."""
    # ---------------------------------------------------------------------
    # 1. Prepare context depending on mode
    # ---------------------------------------------------------------------
    context, pt_code, tests, pt_entrypoint = await _create_execution_context(
        max_turns=max_turns,
        existing_row=existing_row,
        function_name=function_name,
        function_description=function_description,
    )

    try:
        # -----------------------------------------------------------------
        # 2. PyTorch phase (skipped in fix-mode)
        # -----------------------------------------------------------------
        skip_pt = args.fix  # Fix-mode means we skip PyTorch generation
        pt_code, tests, pt_entrypoint = await _run_pytorch_phase(context, max_turns, skip_pt)

        # -----------------------------------------------------------------
        # 3. Triton phase
        # -----------------------------------------------------------------
        triton_output = await _run_triton_phase(context, pt_code, pt_entrypoint, max_turns)

        if isinstance(triton_output, TritonOutput):
            triton_code = triton_output.triton_code
            conversion_reasoning = triton_output.conversion_reasoning
            triton_entrypoint = triton_output.triton_entrypoint
        else:
            triton_code = context.triton_code
            conversion_reasoning = "Conversion reasoning not captured due to parsing issue"
            triton_entrypoint = pt_entrypoint

        # Collect insight only when Triton implementation is correct
        if context.triton_is_correct:
            collected_errors.append(
                ErrorContext(
                    function_name=context.function_name,
                    function_description=context.function_description,
                    pt_code=pt_code,
                    triton_code=triton_code,
                    triton_error_summary="",  # no error on success
                    triton_stderr=context.triton_stdout,
                    conversion_reasoning=conversion_reasoning,
                )
            )

        # -----------------------------------------------------------------
        # 4. Build flat row data
        # -----------------------------------------------------------------
        row_data = context.to_flat_dict() | {
            "pt_code": pt_code,
            "tests": tests,
            "pt_entrypoint": pt_entrypoint,
            "triton_code": triton_code,
            "conversion_reasoning": conversion_reasoning,
            "triton_entrypoint": triton_entrypoint,
        }

        console_print(
            Panel(
                "[green]🎉 Row processed successfully[/green]",
                title="Success",
                border_style="green",
            )
        )
        return unpack_row(row_data)

    except Exception as exc:
        console_print(
            Panel(
                f"[bold red]💥 Error processing row: {exc}[/bold red]",
                title="Error",
                border_style="red",
            )
        )
        raise

@weave.op
async def generate_new_functions_parallel(n_rows: int, max_turns: int):
    """Generate new functions sequentially, then run code generation in parallel"""
    console_print(f"[blue]🎲 Generating {n_rows} new random functions[/blue]")
    
    # First, generate all function names/descriptions sequentially to avoid repetition
    console_print(f"[yellow]📝 Generating {n_rows} unique function names/descriptions...[/yellow]")
    function_infos = []
    
    # Create a dynamic dataset that includes both original and newly generated functions
    # Make a copy to avoid modifying the original dataset
    current_ds = input_ds.select(range(len(input_ds)))  # Creates a copy
    
    for i in range(n_rows):
        console_print(f"[dim]Generating function {i+1}/{n_rows}...[/dim]")
        
        function_info = await generate_function_name_and_description(current_ds)
        function_infos.append(function_info)
        
        # Add this function to our running dataset so next generations see it
        new_entry = {
            args.entrypoint: function_info.function_name,
            args.description: function_info.function_description
        }
        current_ds = current_ds.add_item(new_entry)
        
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

def filter_rows_needing_fix(dataset_rows: list) -> list:
    """Filter rows where PyTorch runs but Triton is incorrect"""
    fixable_rows = []
    for row in dataset_rows:
        pt_runs = row.get("pt_runs", False)
        triton_is_correct = row.get("triton_is_correct", False)
        
        if pt_runs and not triton_is_correct:
            fixable_rows.append(row)
    
    return fixable_rows

@weave.op
async def generate_fix_rows_parallel(rows_to_fix: list, max_turns: int):
    """Fix existing rows in parallel"""
    console_print(f"[magenta]🔧 Fixing {len(rows_to_fix)} rows with failed Triton conversions[/magenta]")
    
    # Run the fixes in parallel
    tasks = []
    for i, row in enumerate(rows_to_fix):
        function_name = row.get("function_name", f"function_{i}")
        console_print(f"[dim]Starting fix for row {i+1}/{len(rows_to_fix)}: {function_name}...[/dim]")
        tasks.append(generate_row(
            max_turns=max_turns,
            existing_row=row
        ))
    
    pds_list = await asyncio.gather(*tasks, return_exceptions=True)
    return pds_list

@weave.op
async def generate_rows(n_rows: int, max_turns: int, rows_to_fix: list = None):
    if args.fix and rows_to_fix is not None:
        # Fix mode: work on existing rows that need fixing
        console_print(Panel(
            f"[bold magenta]🔧 Fixing {len(rows_to_fix)} rows with failed Triton conversions[/bold magenta]",
            title="Fix Mode Configuration",
            border_style="magenta"
        ))
        
        pds_list = await generate_fix_rows_parallel(rows_to_fix, max_turns)
        
    else:
        # Generation mode: create new functions
        console_print(Panel(
            f"[bold blue]🔄 Generating {n_rows} rows with max {max_turns} turns each[/bold blue]",
            title="Processing Configuration",
            border_style="blue"
        ))
        
        pds_list = await generate_new_functions_parallel(n_rows, max_turns)
    
    # Count successes and failures
    valid_rows = [pds for pds in pds_list if pds is not None and not isinstance(pds, Exception)]
    exceptions = [pds for pds in pds_list if pds is None or isinstance(pds, Exception)]
    
    # Filter to only include rows where Triton is correct
    triton_correct_rows = [row for row in valid_rows if row.get("triton_is_correct", False)]
    triton_incorrect_rows = [row for row in valid_rows if not row.get("triton_is_correct", False)]
    
    # Results summary
    if args.fix:
        mode_desc = "fixed rows"
    else:
        mode_desc = "generated functions"
        
    result_text = f"[bold green]✅ Successfully processed with correct Triton: {len(triton_correct_rows)}/{len(pds_list)} {mode_desc}[/bold green]"
    if triton_incorrect_rows:
        result_text += f"\n[bold yellow]⚠️ Processed but Triton incorrect: {len(triton_incorrect_rows)}/{len(pds_list)} {mode_desc}[/bold yellow]"
    if exceptions:
        result_text += f"\n[bold red]❌ Failed with exceptions: {len(exceptions)}/{len(pds_list)} {mode_desc}[/bold red]"
    
    console_print(Panel(result_text, title="📈 Final Results", border_style="green"))
    
    # Only return rows where Triton is correct
    return triton_correct_rows

# ---------------------------------------------------------------------------
# Error reflection helpers (defined before first use)
# ---------------------------------------------------------------------------

def _format_error_details(errors: list["ErrorContext"]) -> str:
    blocks: list[str] = []
    for idx, err in enumerate(errors, 1):
        blocks.append(
            f"\n**Error {idx}: {err.function_name}**\n"
            f"- **Description:** {err.function_description}\n"
            f"- **PyTorch Code:**\n```python\n{err.pt_code}\n```\n"
            f"- **Triton Code (failed):**\n```python\n{err.triton_code}\n```\n"
            f"- **Error Message:** {err.triton_error_summary}\n"
            f"- **Conversion Reasoning:** {err.conversion_reasoning}\n"
            f"- **Stderr:** {err.triton_stderr[:300]}{'...' if len(err.triton_stderr) > 300 else ''}\n"
        )
    return "\n".join(blocks)


def _display_cookbook_update(cookbook_update: "CookbookUpdate") -> None:
    if isinstance(cookbook_update, CookbookUpdate) and cookbook_update.should_update:
        console_print(Panel(
            f"[green]📚 Cookbook improvement recommended![/green]\n"
            f"[dim]Analysis: {cookbook_update.analysis_summary[:150]}...[/dim]\n"
            f"[bold]Key Improvements:[/bold]\n" + "\n".join(
                [f"• {imp}" for imp in cookbook_update.key_improvements[:3]]
            ),
            title="Cookbook Analysis Complete",
            border_style="green",
        ))
    else:
        console_print(Panel(
            f"[yellow]📚 No cookbook improvement needed[/yellow]\n"
            f"[dim]Analysis: {cookbook_update.analysis_summary[:150]}...[/dim]",
            title="Cookbook Analysis Complete",
            border_style="yellow",
        ))


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
    
    # Build markdown summary for LLM
    error_details = _format_error_details(collected_errors)
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
        
        _display_cookbook_update(cookbook_update)
        
        # If cookbook changed, refresh the agent
        if isinstance(cookbook_update, CookbookUpdate) and cookbook_update.should_update:
            console_print("[green]✅ Cookbook has been updated with improvements.[/green]")
            recreate_triton_agent()

    except Exception as e:
        console_print(Panel(
            f"[red]❌ Error during batch cookbook analysis: {str(e)}[/red]",
            title="Analysis Error",
            border_style="red"
        ))

    # Clear collected errors for the next batch
    collected_errors.clear()
    console_print("[blue]🧹 Cleared collected errors for next batch[/blue]")

def save_and_push_dataset(dataset_rows: list, batch_info: str = ""):
    """Helper function to save dataset to disk and optionally push to hub"""
    # Create and save the dataset
    pds = Dataset.from_list(dataset_rows)
    
    # Save to disk
    output_path = args.output_dataset.replace("/", "_")
    pds.save_to_disk(output_path)
    console_print(f"[green]💾 Saved dataset to disk: {output_path} ({len(dataset_rows)} total rows){batch_info}[/green]")
    
    if args.push:
        console_print(f"[blue]🚀 Pushing to HuggingFace Hub: {args.output_dataset}{batch_info}[/blue]")
        pds.push_to_hub(args.output_dataset)
        console_print(f"[green]✅ Successfully pushed to Hub{batch_info}[/green]")

# ---------------------------------------------------------------------------
# Dataset-level helper functions (to shrink `extend_dataset`)
# ---------------------------------------------------------------------------

def _load_existing_dataset() -> list:
    """Return the existing dataset rows (or empty list if unavailable)."""
    try:
        console_print(f"[yellow]📚 Loading existing dataset: {args.input_dataset}[/yellow]")
        existing_ds = load_ds(args.input_dataset, init=False)
        rows = list(existing_ds)
        console_print(f"[blue]📋 Loaded {len(rows)} existing rows[/blue]")
        return rows
    except Exception as exc:
        if args.fix:
            console_print(f"[red]❌ Cannot load dataset for fix mode: {exc}[/red]")
            raise
        console_print(f"[yellow]⚠️ Could not load existing dataset ({exc}); starting fresh.[/yellow]")
        return []


async def _process_fix_batches(all_rows: list, rows_to_fix: list, batch_size: int, max_turns: int):
    """Iterate over batches of rows_to_fix, updating all_rows in-place."""
    total_batches = (len(rows_to_fix) + batch_size - 1) // batch_size
    total_fixed_rows = 0

    console_print(Panel(
        f"[bold magenta]🔧 Processing {len(rows_to_fix)} rows to fix in {total_batches} batches of {batch_size}[/bold magenta]",
        title="Fix Batch Processing Configuration",
        border_style="magenta",
    ))

    for batch_idx in range(total_batches):
        start, end = batch_idx * batch_size, min((batch_idx + 1) * batch_size, len(rows_to_fix))
        batch_rows = rows_to_fix[start:end]
        console_print(Panel(
            f"[bold magenta]🔧 Fix batch {batch_idx + 1}/{total_batches} – size {len(batch_rows)}[/bold magenta]",
            title=f"Fix Batch {batch_idx + 1}",
            border_style="magenta",
        ))

        fixed_rows = await generate_rows(n_rows=len(batch_rows), max_turns=max_turns, rows_to_fix=batch_rows)

        # Merge back into all_rows
        for original_row, fixed_row in zip(batch_rows, fixed_rows):
            if fixed_row is None or isinstance(fixed_row, Exception):
                continue
            for i, r in enumerate(all_rows):
                if (
                    r.get("function_name") == original_row.get("function_name")
                    and r.get("function_description") == original_row.get("function_description")
                ):
                    all_rows[i] = fixed_row
                    break

        total_fixed_rows += len([r for r in fixed_rows if r is not None and not isinstance(r, Exception)])
        console_print(f"[green]✅ Batch {batch_idx + 1} fixed {len(fixed_rows)} rows[/green]")

        # Post-batch housekeeping
        await analyze_errors_and_improve_cookbook()
        save_and_push_dataset(all_rows, f" (after fix batch {batch_idx + 1}/{total_batches})")

    console_print(Panel(
        f"[bold green]🎉 Fix complete – {total_fixed_rows}/{len(rows_to_fix)} rows fixed.[/bold green]",
        title="Fix Complete",
        border_style="green",
    ))


async def _process_generation_batches(all_rows: list, n_rows: int, batch_size: int, max_turns: int):
    """Generate new rows in batches and append to all_rows."""
    total_batches = (n_rows + batch_size - 1) // batch_size
    total_new = 0

    console_print(Panel(
        f"[bold blue]🚀 Processing {n_rows} rows in {total_batches} batches of {batch_size}[/bold blue]",
        title="Generation Batches",
        border_style="blue",
    ))

    for batch_idx in range(total_batches):
        current_size = min(batch_size, n_rows - batch_idx * batch_size)
        console_print(Panel(
            f"[bold cyan]📦 Gen batch {batch_idx + 1}/{total_batches} – size {current_size}[/bold cyan]",
            title=f"Batch {batch_idx + 1}",
            border_style="cyan",
        ))
        batch_rows = await generate_rows(n_rows=current_size, max_turns=max_turns)
        all_rows.extend(batch_rows)
        total_new += len(batch_rows)

        await analyze_errors_and_improve_cookbook()
        save_and_push_dataset(all_rows, f" (after gen batch {batch_idx + 1}/{total_batches})")

    console_print(Panel(
        f"[bold green]🎉 Generation complete – added {total_new} new rows.[/bold green]",
        title="Generation Complete",
        border_style="green",
    ))

# ============================================================================
# UTILITY CLASSES (ConsoleReporter) and MAIN PROCESSOR (DatasetProcessor)
# ============================================================================

class ConsoleReporter:
    """Thin wrapper around rich.Console to centralise verbosity handling."""
    def __init__(self, console: Console, verbose: bool = False):
        self._console = console
        self.verbose = verbose

    def print(self, *args, verbose_only: bool = False, **kwargs):
        if (not verbose_only) or self.verbose:
            self._console.print(*args, **kwargs)

    def panel(self, content: str, title: str = "", style: str = "blue"):
        self.print(Panel(content, title=title, border_style=style))


class DatasetProcessor:
    """Encapsulates the full dataset-generation workflow to avoid globals."""

    def __init__(self, args: Args):
        # Keep a reference to the original global *args* so that legacy helpers
        # still work.  In a later clean-up pass we can thread *args* explicitly
        # everywhere and delete the global, but for now we just mirror it so the
        # rest of the file keeps functioning unchanged.
        globals()["args"] = args  # noqa: SPL001 – intentional global write
        self.args = args
        self.console = ConsoleReporter(console, verbose=args.verbose)
        # Make the reporter globally discoverable for legacy helper functions.
        global _GLOBAL_CONSOLE_REPORTER  # noqa: PLW0603 – global mutation by design
        _GLOBAL_CONSOLE_REPORTER = self.console

        # Replace the legacy global error list with an instance attribute while
        # keeping backward compatibility for code that still references the
        # global name.
        self.collected_errors: list[ErrorContext] = []
        globals()["collected_errors"] = self.collected_errors  # type: ignore

        # Expose this processor for helper wrappers that live at module scope.
        globals()["_GLOBAL_DATASET_PROCESSOR"] = self  # type: ignore

    async def run(self):
        """Entry-point that simply forwards to the existing *extend_dataset* op."""
        # Reset any global state that previous runs might have left behind
        self.reset_state()

        # All heavy lifting is still performed by the pre-existing async
        # function.  Once that function is fully migrated into the class we can
        # delete this indirection.
        await process_dataset(
            mode="fix" if self.args.fix else "generate",
            n_rows=self.args.n_rows,
            batch_size=self.args.batch_size,
            max_turns=self.args.max_turns,
        )

    # ---------------------------------------------------------------------
    # Interim compatibility helpers – these still mutate the legacy globals,
    # but wrap them behind an instance-level convenience API so that callers
    # don't need to import the globals directly any more.  In a later pass we
    # will refactor the downstream functions to use these attributes instead
    # and then remove the globals entirely.
    # ---------------------------------------------------------------------

    @staticmethod
    def reset_state():
        """Clear legacy global state for a clean execution."""
        global collected_errors  # noqa: PLW0603
        collected_errors.clear()

    # ------------------------------------------------------------------
    # Batch-processing helpers (formerly module-level)
    # ------------------------------------------------------------------

    async def process_fix_batches(self, all_rows: list, rows_to_fix: list, batch_size: int, max_turns: int):
        total_batches = (len(rows_to_fix) + batch_size - 1) // batch_size
        total_fixed = 0

        self.console.panel(
            f"🔧 Processing {len(rows_to_fix)} rows to fix in {total_batches} batches of {batch_size}",
            title="Fix Batches",
            style="magenta",
        )

        for idx in range(total_batches):
            batch = rows_to_fix[idx * batch_size : (idx + 1) * batch_size]
            self.console.panel(
                f"🔧 Fix batch {idx + 1}/{total_batches} – size {len(batch)}",
                title=f"Fix Batch {idx + 1}",
                style="magenta",
            )

            fixed_rows = await generate_rows(n_rows=len(batch), max_turns=max_turns, rows_to_fix=batch)

            # Merge results
            for orig, fixed in zip(batch, fixed_rows):
                if fixed is None or isinstance(fixed, Exception):
                    continue
                for i, row in enumerate(all_rows):
                    if row.get("function_name") == orig.get("function_name") and row.get("function_description") == orig.get("function_description"):
                        all_rows[i] = fixed
                        break

            total_fixed += len([r for r in fixed_rows if r is not None and not isinstance(r, Exception)])

            await analyze_errors_and_improve_cookbook()
            save_and_push_dataset(all_rows, f" (after fix batch {idx + 1}/{total_batches})")

        self.console.panel(f"🎉 Fix complete – {total_fixed}/{len(rows_to_fix)} rows fixed.", title="Fix Complete", style="green")

    async def process_generation_batches(self, all_rows: list, n_rows: int, batch_size: int, max_turns: int):
        total_batches = (n_rows + batch_size - 1) // batch_size
        total_new = 0

        self.console.panel(
            f"🚀 Generating {n_rows} rows in {total_batches} batches of {batch_size}",
            title="Generation Batches",
            style="blue",
        )

        for idx in range(total_batches):
            size = min(batch_size, n_rows - idx * batch_size)
            self.console.panel(
                f"📦 Gen batch {idx + 1}/{total_batches} – size {size}",
                title=f"Batch {idx + 1}",
                style="cyan",
            )
            batch_rows = await generate_rows(n_rows=size, max_turns=max_turns)
            all_rows.extend(batch_rows)
            total_new += len(batch_rows)

            await analyze_errors_and_improve_cookbook()
            save_and_push_dataset(all_rows, f" (after gen batch {idx + 1}/{total_batches})")

        self.console.panel(f"🎉 Generation complete – added {total_new} new rows.", title="Generation Complete", style="green")

# ---------------------------------------------------------------------------
# Entry-point: instantiate the processor and launch the workflow.
# ---------------------------------------------------------------------------

@weave.op
async def process_dataset(
    mode: Literal["generate", "fix"],
    n_rows: int = 200,
    batch_size: int = 10,
    max_turns: int = 20,
):
    """Unified entry-point for dataset processing.

    mode="generate" → add `n_rows` new PyTorch/Triton pairs
    mode="fix"      → retry Triton conversion on failing rows
    """

    DatasetProcessor.reset_state()

    all_rows = _load_existing_dataset()

    proc = globals().get("_GLOBAL_DATASET_PROCESSOR") or DatasetProcessor(args)

    if mode == "fix":
        rows_to_fix = filter_rows_needing_fix(all_rows)
        if not rows_to_fix:
            console_print(Panel("[green]🎉 Nothing to fix![/green]", title="Skip", border_style="green"))
            return
        await proc.process_fix_batches(all_rows, rows_to_fix, batch_size, max_turns)
    else:
        await proc.process_generation_batches(all_rows, n_rows, batch_size, max_turns)

    save_and_push_dataset(all_rows, " (final)")

# ---------------------------------------------------------------------------
# Entry-point script execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    processor = DatasetProcessor(args)
    asyncio.run(processor.run())

