import asyncio
import weave
import openai
from dataclasses import dataclass
from datasets import load_dataset, Dataset
import simple_parsing as sp
from triton_eval.utils import map
from pydantic import BaseModel, Field

client = openai.AsyncOpenAI()

@dataclass
class Args:
    ds_name: str = "tcapelle/boostrap_oai_pt_think"
    model: str = "gpt-4.1"
    num_proc: int = 20
    weave_project: str = "grpo-cuda/llm-tricks"
    output_ds_name: str = "tcapelle/boostrap_oai_pt_think"
    push: bool = False
    debug: bool = False

args = sp.parse(Args)

ds = load_dataset(args.ds_name)["train"]

system_prompt = """
I want to remove the __main__ part of the tests:

Input:

```python
import torch
# other imports
# test code

# after test code
if __name__ == "__main__":
    test_results = test_relu_max_sum()
    print(test_results)
```

And it should just print without any __main__:

Expected Output:

```python
import torch
torch.manual_seed(42)
# other imports

def test_code(...):
    test_results = {}
    test_restuls["test_case_1] = ...
    test_restuls["test_case_2] = ...
    ...
    return test_results

# after test code
test_results = test_relu_max_sum()
print(test_results)
```

Don't print inside the test_code function, it should just return the test results, and after the test_code function, you should print the test results.
Return the tests code without any ```python or ```.
"""

user_prompt = """
Fix the tests of the following code:

```python
{tests}
```
"""

class FormattedTests(BaseModel):
    fixed_tests: str = Field(description="The fixed test without any __main__,")

async def format_row(row):
    if "if __name__" not in row["tests"]:
        return {"tests": row["tests"]}
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt.format(tests=row["tests"])}
        ]
        response = await client.responses.parse(
            model=args.model,
            input=messages,
            text_format=FormattedTests,
        )
        print(response.output_parsed.fixed_tests)
        print("-"*100)
        return {"tests": response.output_parsed.fixed_tests}



weave.init(args.weave_project)

if args.debug:
    ds = ds.select(range(10))

out_ds = asyncio.run(map(ds, format_row, num_proc=args.num_proc))
out_ds = Dataset.from_list(out_ds)
out_ds.save_to_disk(args.output_ds_name.replace("/", "_"))

if args.push:
    out_ds.push_to_hub(args.output_ds_name, commit_message="Format tests")