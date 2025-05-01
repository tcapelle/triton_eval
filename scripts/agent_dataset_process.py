"Reverse the jit triton kernel to pytorch code"

import openai
import weave
from rich.console import Console
from rich.pretty import pprint
from datasets import load_dataset, load_from_disk
from pydantic import BaseModel, Field
from my_smol_agent.agent import Agent
from my_smol_agent.tools import clear_temp_files

from prompts import pytorch_agent_prompt

console = Console()

DEBUG = False
INPUT_DATASET = "train_ds_triton_v2f"
OUTPUT_DATASET = "train_ds_triton_v2f2"
WEAVE_PROJECT = "grpo-cuda/dataset_agent"

client = openai.OpenAI()

input_ds = load_from_disk(INPUT_DATASET)
if DEBUG:
    input_ds = input_ds.select(range(10))

console.print("[bold blue]Input dataset[/bold blue]")
console.print(input_ds)

console.rule("[bold blue]Fixing code with Agent[/bold blue]")

clear_temp_files()

weave.init(WEAVE_PROJECT)

def func_to_map(row):
    class PytorchCodeWithTests(BaseModel):
        format_pt_code: str = Field(description="The pytorch file without the tests. No ```python or ``` needed, just the code.")
        tests: str = Field(description="The tests that were run on the pytorch code. No ```python or ``` needed, just the code.")
        pt_code_runs: bool = Field(description="Whether the pytorch code runs or not.")
        entrypoint: str = Field(description="The entrypoint of the pytorch code. It should match the tests naming test_<entrypoint>")

    pytorch_code = row["format_pt_code"]
    entrypoint = row["entrypoint"]
    try:
        agent = Agent(model_name="o4-mini", system_message=pytorch_agent_prompt, silent=True, response_format=PytorchCodeWithTests)
        agent_response = agent.run(
            user_prompt=f"Here is the pytorch code for the function {entrypoint} with the tests:\n\n{pytorch_code}. Make sure to fix the code and the tests.", max_steps=5)
        res = agent_response.final_response.model_dump()
        print(f"=============== Fixed code ==========================")
        return res
    except Exception as e:
        print(f"Error: {e}")
        return {"format_pt_code": pytorch_code, "tests": "", "pt_code_runs": False, "entrypoint": entrypoint}



