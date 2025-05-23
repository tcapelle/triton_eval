"Reverse the jit triton kernel to pytorch code"


import asyncio
from weave.flow.util import async_foreach
from weave.trace.op_caller import async_call
import openai
import weave
from dataclasses import dataclass
from rich.console import Console
from datasets import load_dataset, load_from_disk, Dataset
from pydantic import BaseModel, Field
from triton_eval.agents.agent import Agent
from triton_eval.agents.tools import clear_temp_files
from triton_eval.utils import map
import simple_parsing as sp

from prompts import test_creator_prompt

console = Console()

@dataclass
class Args:
    debug: bool = False
    input_dataset: str = "tcapelle/boostrap_triton"
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

clear_temp_files()

weave.init(args.weave_project)


system_message = """You are an expert PyTorch Triton programmer. Your task is to make sure the Triton code runs and is correct. Run it and check the output. If it doesn't work, fix it. You ahve access to tools to run code on GPU."""


class PytorchCodeWithTests(BaseModel):
    triton_code_runs: bool = Field(description="Whether the Triton code runs or not.")
    triton_code: str = Field(description="The Triton code to run. No ```python or ``` needed, just the code")
    triton_stdout: str = Field(description="The stdout of the Triton code.")
    triton_stderr: str = Field(description="The stderr of the Triton code.")

user_prompt = """Here is the triton code for the function {entrypoint}:
```py{triton_code}\n #### \n {tests}```. Run it and check the outputs. Don't change the signature or the format."
"""

@weave.op
def func_to_map(row):
    triton_code = row["triton_code"]
    entrypoint = row["entrypoint"]
    tests = row["tests"]
    try:
        agent = Agent(model_name="o4-mini", system_message=system_message, silent=True, response_format=PytorchCodeWithTests)
        agent_response = agent.run(
            user_prompt=user_prompt.format(triton_code=triton_code, entrypoint=entrypoint, tests=tests), max_steps=10)
        if agent_response.stop_reason == "done":
            res = agent_response.final_response.model_dump()
            res["stop_reason"] = agent_response.stop_reason
            return res
        else:
            console.print(f"Stop reason: {agent_response.stop_reason}")
            return {"triton_code_runs": False, 
                    "triton_code": triton_code, 
                    "triton_stdout": "", 
                    "triton_stderr": "", 
                    "stop_reason": agent_response.stop_reason}
    except Exception as e:
        print(f"Error: {e}")
        return {"triton_code_runs": False, 
                "triton_code": triton_code, 
                "triton_stdout": "", 
                "triton_stderr": "", 
                "stop_reason": str(e)}

console.rule("[bold blue]Processing dataset[/bold blue]")

if args.debug:
    input_ds = input_ds.select(range(10))

pds_list = asyncio.run(map(input_ds, func_to_map, num_proc=2 if args.debug else args.num_proc))
pds = Dataset.from_list(pds_list)
pds.save_to_disk(args.output_dataset.replace("/", "_"))

if args.push:
    pds.push_to_hub(args.output_dataset)
