"Reverse the jit triton kernel to pytorch code"


import asyncio
from weave.flow.util import async_foreach
import openai
import weave
from dataclasses import dataclass
from rich.console import Console
from datasets import load_dataset, load_from_disk, Dataset
from pydantic import BaseModel, Field
from triton_eval.agents.agent import Agent
from triton_eval.agents.tools import clear_temp_files
import simple_parsing as sp

from prompts import test_creator_prompt

console = Console()

@dataclass
class Args:
    debug: bool = False
    input_dataset: str = "tcapelle/boostrap_triton"
    output_dataset: str = "tcapelle/boostrap_triton"
    weave_project: str = "grpo-cuda/dataset_agent"
    push: bool = False


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

async def func_to_map(row):
    class PytorchCodeWithTests(BaseModel):
        tests_code: str = Field(description="The tests code for the pytorch code. No ```python or ``` needed, just the code")
        pt_code_runs: bool = Field(description="Whether the pytorch code runs or not.")
        entrypoint: str = Field(description="The entrypoint of the pytorch code. It should match the tests naming test_<entrypoint>")

    pytorch_code = row["pt_code"]
    entrypoint = row["pt_entrypoint"]
    runs = False
    if runs:
        return row
    try:
        agent = Agent(model_name="o4-mini", system_message=test_creator_prompt, silent=True, response_format=PytorchCodeWithTests)
        agent_response = agent.run(
            user_prompt=f"Here is the pytorch code for the function {entrypoint}:\n\n```py{pytorch_code}```. Create the tests for the code and make sure they run.", max_steps=10)
        if agent_response.stop_reason == "done":
            res = agent_response.final_response.model_dump()
            res["stop_reason"] = agent_response.stop_reason
            console.print(f"=============== Fixed code ==========================")
            return res
        else:
            console.print(f"=============== Failed to fix code ==========================")
            console.print(f"Stop reason: {agent_response.stop_reason}")
            return {"format_pt_code": pytorch_code, 
                    "pt_code_runs": False, 
                    "entrypoint": entrypoint, 
                    "stop_reason": agent_response.stop_reason}
    except Exception as e:
        print(f"Error: {e}")
        return {"format_pt_code": pytorch_code, 
                "pt_code_runs": False, 
                "entrypoint": entrypoint, 
                "stop_reason": e}


@weave.op
async def map(ds, func, num_proc=10):
    results = []
    n_complete = 0
    async for _, out_row in async_foreach(ds, func, max_concurrent_tasks=num_proc):
        results.append(out_row)
        n_complete += 1
        print(f"Completed {n_complete} / {len(ds)}")
    return results

console.rule("[bold blue]Processing dataset[/bold blue]")
if args.debug:
    ds = input_ds.select(range(10))
    _ = asyncio.run(map(input_ds.select(range(5)), func_to_map, num_proc=5))
else:
    pds_list = asyncio.run(map(input_ds, func_to_map, num_proc=10))
    pds = Dataset.from_list(pds_list)
    pds.save_to_disk(args.output_dataset)

    if args.push:
        pds.push_to_hub(args.output_dataset)
