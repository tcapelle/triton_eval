import asyncio
import httpx
import weave
from dataclasses import dataclass
from rich.console import Console
from datasets import load_dataset, load_from_disk, Dataset
from triton_eval.utils import map
from triton_eval.agents.tools import clear_temp_files
import simple_parsing as sp

console = Console()

@dataclass
class Args:
    debug: bool = False
    input_dataset: str = "tcapelle/boostrap_oai_pt"
    output_dataset: str = None
    weave_project: str = "grpo-cuda/dataset_map"
    push: bool = False
    num_proc: int = 10
    timeout: int = 60
    server_url: str = "http://127.0.0.1:9347"
    code_row: str = "pt_code"
    tests_row: str = "tests"
    entrypoint_row: str = "pt_entrypoint"  # Optional column name containing the function to torch.compile
    # PyTorch-specific benchmarking options
    benchmark: bool = True
    benchmark_runs: int = 100
    torch_compile: bool = True
    torch_compile_mode: str = "default"  # "default", "reduce-overhead", "max-autotune"

args = sp.parse(Args)

output_dataset = args.input_dataset if args.output_dataset is None else args.output_dataset

def load_ds(dataset_name):
    if "/" in dataset_name:
        return load_dataset(dataset_name, revision="234e7f10b89ecbe46f293421349a88123cc92d99")["train"]
    else:
        ds = load_from_disk(dataset_name)
        # Handle both split and non-split datasets
        if hasattr(ds, 'keys') and 'train' in ds:
            return ds["train"]
        else:
            return ds

console.rule(f"[bold blue]Loading dataset: {args.input_dataset}[/bold blue]")

input_ds = load_ds(args.input_dataset)

console.print("[bold blue]Input dataset[/bold blue]")
console.print(input_ds)

console.rule("[bold blue]Processing PyTorch code with benchmarking[/bold blue]")

clear_temp_files()

weave.init(args.weave_project)


@weave.op
async def call_benchmark_server(pt_code, tests, entrypoint=None):
    """Run benchmark with PyTorch and torch.compile"""
    async with httpx.AsyncClient() as client:
        payload = {
            "code": pt_code,
            "tests": tests,
            "benchmark": args.benchmark,
            "benchmark_runs": args.benchmark_runs,
            "torch_compile": args.torch_compile,
            "torch_compile_mode": args.torch_compile_mode
        }
        if entrypoint:
            payload["entrypoint"] = entrypoint
            
        resp = await client.post(f"{args.server_url}/run_pytorch", 
                                   json=payload,
                                   timeout=args.timeout)
        resp.raise_for_status()
        return resp.json()

@weave.op
async def run_code(row):
    pt_code = row[args.code_row]
    tests = row[args.tests_row]
    
    # Extract entrypoint if specified
    entrypoint = None
    if args.entrypoint_row and args.entrypoint_row in row:
        entrypoint = row[args.entrypoint_row]

    # Run benchmark - this single call returns both PyTorch and torch.compile metrics
    result = await call_benchmark_server(pt_code, tests, entrypoint)
    


    # Use the server response directly and add our custom fields
    enhanced_result = {
        # Original fields
        "pt_code": pt_code,
        "tests": tests,
        # All server response fields
        **result,
        # Success flags for easy filtering
        "execution_success": result["status_code"] == 0,
        "has_benchmark_data": result.get("benchmark_mean_time_ms") is not None,
        "has_torch_compile_data": result.get("torch_compile_benchmark_mean_time_ms") is not None,
    }
    
    row.update(enhanced_result)
    return row
    

console.rule("[bold blue]Processing dataset[/bold blue]")

if args.debug:
    debug_size = min(10, len(input_ds))
    input_ds = input_ds.select(range(debug_size))

async def process_ds():
    pds_list = await map(input_ds, run_code, num_proc=2 if args.debug else args.num_proc)
    pds = Dataset.from_list(pds_list)
    return pds

pds = asyncio.run(process_ds())

# Save locally with benchmark info in the name
output_name = output_dataset.replace("/", "_")
if args.benchmark:
    output_name += "_benchmarked"

pds.save_to_disk(output_name)

if args.push:
    pds.push_to_hub(output_dataset)
    console.print(f"Pushed to hub: {output_dataset}")

console.print(f"[bold green]Dataset processing complete![/bold green]")
console.print(f"Processed {len(pds)} samples")
console.print(f"Local save: {output_name}")
    

# Print some statistics about the benchmark results
if args.benchmark:
    successful_executions = sum(1 for item in pds if item.get("execution_success", False))
    benchmark_data_count = sum(1 for item in pds if item.get("has_benchmark_data", False))
    torch_compile_count = sum(1 for item in pds if item.get("has_torch_compile_data", False))
    
    console.print(f"[cyan]Execution Statistics:[/cyan]")
    console.print(f"  Successful executions: {successful_executions}/{len(pds)} ({successful_executions/len(pds)*100:.1f}%)")
    console.print(f"  With benchmark data: {benchmark_data_count}/{len(pds)} ({benchmark_data_count/len(pds)*100:.1f}%)")
    console.print(f"  With torch.compile data: {torch_compile_count}/{len(pds)} ({torch_compile_count/len(pds)*100:.1f}%)")

    console.print(pds[0])
