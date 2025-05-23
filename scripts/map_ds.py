import asyncio
import weave
import openai
from dataclasses import dataclass
from pydantic import BaseModel, Field
from datasets import load_dataset, Dataset
import simple_parsing as sp
from triton_eval.utils import map
from prompts import sft_system_prompt, sft_user_prompt

client = openai.AsyncOpenAI()

@dataclass
class Args:
    ds_name: str = "tcapelle/boostrap_triton_ran"
    model: str = "gpt-4.1"
    num_proc: int = 20
    weave_project: str = "grpo-cuda/llm-tricks"
    output_ds_name: str = "tcapelle/boostrap_triton_ran"
    push: bool = False
    debug: bool = False

args = sp.parse(Args)

ds = load_dataset(args.ds_name)["train"]


system_prompt = """You are an expert Python programmer. Your tast is to fix the tests"""


user_prompt = """
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

class TorchTritonReasoning(BaseModel):
    reasoning: str = Field(description="The reasoning for the conversion from the PyTorch code to the Triton code.")


async def format_row(row):

    messages = [
        {"role": "system", "content": sft_system_prompt},
        {"role": "user", "content": sft_user_prompt.format(pt_code=row["pt_code"], triton_code=row["triton_code"])}
    ]
    response = await client.responses.parse(
        model=args.model,
        input=messages,
        text_format=TorchTritonReasoning,
    )
    row["reasoning"] = response.output_parsed.reasoning
    return row



weave.init(args.weave_project)

if args.debug:
    ds = ds.select(range(10))

out_ds = asyncio.run(map(ds, format_row, num_proc=args.num_proc))
out_ds = Dataset.from_list(out_ds)
out_ds.save_to_disk(args.output_ds_name.replace("/", "_"))

if args.push:
    out_ds.push_to_hub(args.output_ds_name)