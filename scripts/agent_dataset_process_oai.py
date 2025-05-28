import asyncio
import random
from pathlib import Path
import openai
import weave
from dataclasses import dataclass
from rich.console import Console
from datasets import load_dataset, load_from_disk, Dataset
from pydantic import BaseModel, Field
import simple_parsing as sp

from agents import Agent, Runner, function_tool

from triton_eval.agents.tools import run_python_code_on_gpu

console = Console()

@dataclass
class Args:
    debug: bool = False
    input_dataset: str = "tcapelle/boostrap_triton_ran"
    output_dataset: str = "tcapelle/boostrap_triton_ran"
    weave_project: str = "grpo-cuda/dataset_agent"
    push: bool = False
    num_proc: int = 10

args = sp.parse(Args)

client = openai.OpenAI()

def load_ds(dataset_name):
    if "/" in dataset_name:
        return load_dataset(dataset_name)["train"]
    else:
        return load_from_disk(dataset_name)["train"]

console.rule(f"[bold blue]Loading dataset: {args.input_dataset}[/bold blue]")

input_ds = load_ds(args.input_dataset)

console.print("[bold blue]Input dataset[/bold blue]")
console.print(input_ds)

console.rule("[bold blue]Fixing code with Agent[/bold blue]")

weave.init(args.weave_project)


### First Agent: Generate PyTorch/Triton pairs

def join_past_rows(ds, num_past_rows):
    "sample `rows_to_sample` randomly from ds"
    rows = random.sample(ds.to_list(), num_past_rows)
    formatted_rows = "\n".join([f"====== \n{row['pt_code']}\n---\n{row['triton_code']}\n---\n{row['tests']}\n======\n" for i, row in enumerate(rows)])
    return formatted_rows

triton_cookbook = Path("/Users/tcapelle/work/triton_eval/scripts/data/triton_cookbook.md").read_text()

class RunnerOutput(BaseModel):
    code_runs: bool = Field(description="Whether the code runs or not.")
    stdout: str = Field(default="", description="The stdout of the code.")
    stderr: str = Field(default="", description="The stderr of the code.")

class PyTorchTritonRow(BaseModel):
    conversion_reasoning: str = Field(description="The reasoning step by step on how the conversion to triton should be done for this specific function")
    pt_code: str = Field(description="The PyTorch code for the function")
    triton_code: str = Field(description="The Triton code for the function")
    pt_entrypoint: str = Field(description="The entrypoint of the function in Pytorch")
    triton_entrypoint: str = Field(description="The entrypoint of the function in Triton")
    tests: str = Field(description="The tests for the function")
    pt_output: RunnerOutput = Field(description="The output of the PyTorch code.")
    triton_output: RunnerOutput = Field(description="The output of the Triton code.")

generation_system_prompt = f"""We are generating a PyTorch/Triton pairs dataset. We want functions that have exactly the same funcionalities.

Your task is to generate a new row for our dataset. Focus on clarity and simplicity, Triton can be very complex, the idea is generating pairs that are easy to understand and that can be used to learn Triton.

Here it's a best practice on writing Triton kernels: {triton_cookbook}

Return the reasoning step by step on how the conversion to triton should be done for this specific function.

Make also a set of tests that validate the functionality of the function.

## Regarding the tests:

- Ensure that all branch tests are in a single function starting with
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

You have access to tools to run code, make sure to use it to validate your tests run. Make sure the triton and pytorch code produce the same output.

A perfect example of the testswould look like this:

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

generation_user_prompt = """Our dataset is comprised now of the following rows:
{formatted_rows}

Be creative and think about dataset diversity, let's not make every row of the dataset row identical. Don't just re-implement top level torch operations like torch.sum, torch.floor, etc.. those have Triton primitives already.
Generate a new sample pair of PyTorch and Triton code that we can add to the dataset.

Keep the format of naming your triton entrypoint the same as the PyTorch entrypoint. This will enable us to use the same entrypoint for both PyTorch and Triton.
"""

boot_agent = Agent(
    name="BoostraperAgent", 
    model="o4-mini",
    instructions=generation_system_prompt,
    tools=[function_tool(run_python_code_on_gpu),],
    output_type=PyTorchTritonRow)

console.rule("[bold blue]Processing dataset[/bold blue]")


@weave.op
async def main(frows):
    pt_triton_row = await Runner.run(boot_agent, generation_user_prompt.format(formatted_rows=frows))
    
    return pt_triton_row.final_output.model_dump()

if args.debug:
    input_ds = input_ds.select(range(10))

frows = join_past_rows(input_ds, 10)

out = asyncio.run(main(frows))
print(out)

# pds = Dataset.from_list(pds_list)
# pds.save_to_disk(args.output_dataset.replace("/", "_"))

# if args.push:
#     pds.push_to_hub(args.output_dataset)
