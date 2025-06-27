from textwrap import dedent
from dataclasses import dataclass
from rich.console import Console
from pydantic import BaseModel, Field
import openai
import openai
from datasets import load_dataset
import simple_parsing as sp
from agents import Agent, Runner, RunContextWrapper, function_tool, WebSearchTool
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX

from triton_eval.agents.tools import run_python_code_on_gpu

client = openai.Client()

console = Console()

client = openai.OpenAI()

@dataclass
class Args:
    debug: bool = False
    input_dataset: str = "tcapelle/bootstrap_oai_pt"

args = sp.parse(Args)

ds = load_dataset(args.input_dataset, split="train")

if args.debug:
    ds = ds.select(range(10))

console.print(f"Loading {args.input_dataset} with {len(ds)} rows")


def format_sample(row):
    pt_code = row["pt_code"]
    test = row["tests"]
    pt_entrypoint = row["pt_entrypoint"]

    text = (f"# PT Code:\n{pt_code}\n\n"
        f"# Test:\n{test}\n\n"
        f"# PT Entrypoint:\n{pt_entrypoint}\n\n")

    return text

class NewSample(BaseModel):
    logic: str = Field(description="The logic of the refactor")
    pt_code: str = Field(description="The PT code of the refactor")
    tests: str = Field(description="The tests of the refactor")
    pt_entrypoint: str = Field(description="The new PT entrypoint of the refactor")


row_0 = ds[0]
console.print(format_sample(row_0)) 

console.rule("Refactor")

system_prompt = """
You are an expert PyTorch programmer, your mission is to refactor the code and tests so:
- Produce a different version of the code that produces the same output. For example, converting functional PyTorch to nn.Module.
- Keep the same tests, but rename the entrypoint accordigly. You may need to define intermediate variables to preserve the logic.
- The one liner printing logic must be preserved. That is how we compare correctness.
- You can rename the entrypoint, for example as `entrypoint_modular`, `entrypoint2`, etc.
- Don't add any __main__ to the code, just a simple print statement at the end of the tests is enough.

The idea is to use this sample as a way to augment our dataset, not change the logic.

Return the refactored code and tests, and the new entrypoint. Make sure the code runs, you can call `run_code` to test it.
"""

user_prompt = """
Refactor the following code and tests:
{format_sample}
"""

response = client.responses.parse(
    model="gpt-4.1",
    input=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt.format(format_sample=format_sample(row_0))}],
    text_format=NewSample,
)

out = response.output_parsed

console.print(format_sample(out.model_dump()))

console.rule("Run")



class ExecutionContext(BaseModel):
    stdout: str = Field(description="The stdout of the execution")
    stderr: str = Field(description="The stderr of the execution")
    returncode: int = Field(description="The returncode of the execution")

@function_tool
def run_code(wrapper: RunContextWrapper[ExecutionContext], code: str, tests: str) -> dict:
    "Run the code and tests on the GPU"
    full_code = f"{code}\n\n############\nimport torch\ntorch.set_seed(42)\ntorch.set_printoptions(threshold=int(1e9))\n\n{tests}"
    return run_python_code_on_gpu(full_code)


class AugmentContext(BaseModel):
    pt_code: str = Field(description="The PT code of the refactor")
    tests: str = Field(description="The tests of the refactor")
    pt_entrypoint: str = Field(description="The new PT entrypoint of the refactor")
    original_entrypoint: str = Field(description="The original PT entrypoint of the refactor")


augment_agent = Agent(
    name="AugmentAgent",
    tools=[run_code],
    instructions=system_prompt,
    output_type=NewSample,
)









