import asyncio
import weave
from dataclasses import dataclass
from rich.console import Console
from datasets import load_dataset, load_from_disk, Dataset
from triton_eval.utils import map
from triton_eval.agents.tools import clear_temp_files, run_python_code_on_gpu
import simple_parsing as sp


console = Console()

@dataclass
class Args:
    debug: bool = False
    input_dataset: str = "tcapelle/boostrap_triton"
    output_dataset: str = "tcapelle/boostrap_triton_ran"
    weave_project: str = "grpo-cuda/dataset_agent"
    push: bool = False
    num_proc: int = 10
    timeout: int = 60

args = sp.parse(Args)


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


def run_code(row):
    triton_code = row["triton_code"]
    tests = row["tests"]
    tests = tests.replace("_triton", "") # let's get rid of the triton naming..
    code = f"{triton_code}\nfrom typing import *\n{tests}"
    result = run_python_code_on_gpu(code, timeout=args.timeout)
    # result = {"status_code": 0, "stdout": result.stdout, "stderr": result.stderr}
    triton_runs = result["status_code"] == 0
    triton_stdout = result["stdout"]
    triton_stderr = result["stderr"]
    return {tests: tests, "triton_runs": triton_runs, "triton_stdout": triton_stdout, "triton_stderr": triton_stderr}

console.rule("[bold blue]Processing dataset[/bold blue]")

if args.debug:
    input_ds = input_ds.select(range(10))

pds_list = asyncio.run(map(input_ds, run_code, num_proc=2 if args.debug else args.num_proc))
pds = Dataset.from_list(pds_list)
pds.save_to_disk(args.output_dataset.replace("/", "_"))

if args.push:
    pds.push_to_hub(args.output_dataset)
