"Reverse the jit triton kernel to pytorch code"

import openai
import weave
import json
from dataclasses import dataclass
from rich.console import Console
from datasets import load_dataset, load_from_disk, Dataset
from pydantic import BaseModel, Field
from my_smol_agent.agent import Agent
from my_smol_agent.tools import clear_temp_files
import simple_parsing as sp

from prompts import pytorch_agent_prompt

console = Console()

@dataclass
class Args:
    debug: bool = False
    input_dataset: str = "tcapelle/train_ds_triton_v2f"
    output_dataset: str = "train_ds_triton_v2f2"
    weave_project: str = "grpo-cuda/dataset_agent"


args = sp.parse(Args)

client = openai.OpenAI()

def load_ds(dataset_name):
    if "/" in dataset_name:
        return load_dataset(dataset_name)["train"]
    else:
        return load_from_disk(dataset_name)["train"]

console.rule(f"[bold blue]Loading dataset: {args.input_dataset}[/bold blue]")

input_ds = load_ds(args.input_dataset)
if args.debug:
    input_ds = input_ds.select(range(3))

console.print("[bold blue]Input dataset[/bold blue]")
console.print(input_ds)

console.rule("[bold blue]Fixing code with Agent[/bold blue]")

clear_temp_files()

if args.debug:
    weave.init(args.weave_project)

def func_to_map(row):
    class PytorchCodeWithTests(BaseModel):
        format_pt_code: str = Field(description="The pytorch file with tests. No ```python or ``` needed, just the code")
        pt_code_runs: bool = Field(description="Whether the pytorch code runs or not.")
        entrypoint: str = Field(description="The entrypoint of the pytorch code. It should match the tests naming test_<entrypoint>")

    pytorch_code = row["format_pt_code"]
    entrypoint = row["entrypoint"]
    try:
        agent = Agent(model_name="o4-mini", system_message=pytorch_agent_prompt, silent=True, response_format=PytorchCodeWithTests)
        agent_response = agent.run(
            user_prompt=f"Here is the pytorch code for the function {entrypoint} with the tests:\n\n```py{pytorch_code}```. Make sure to fix the code and the tests.", max_steps=10)
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
def process_dataset_safe(ds, output_file="./output.jsonl"):
    console.print(f"[bold blue]Processing safe to {output_file} [/bold blue]")
    with open(output_file, "w") as f:
        for i, row in enumerate(ds):
            print(f"Processing row {i} of {len(ds)-1}")
            res = func_to_map(row)
            row.update(res)
            row["row_num"] = i
            f.write(json.dumps(row) + "\n")

console.rule("[bold blue]Processing dataset[/bold blue]")
if args.debug:
    process_dataset_safe(input_ds, output_file="output.jsonl")
else:
    pds = input_ds.map(func_to_map, num_proc=10)
    pds.save_to_disk(args.output_dataset)