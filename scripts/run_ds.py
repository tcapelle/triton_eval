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
    input_dataset: str = "tcapelle/boostrap_oai_pt_think"
    output_dataset: str = "tcapelle/boostrap_oai_pt_think"
    weave_project: str = "grpo-cuda/dataset_map"
    push: bool = False
    num_proc: int = 10
    timeout: int = 60
    server_url: str = "http://127.0.0.1:9347"
    code_row: str = "pt_code"
    tests_row: str = "tests"
    # PyTorch-specific benchmarking options
    benchmark: bool = True
    benchmark_runs: int = 10
    torch_compile: bool = True
    torch_compile_mode: str = "default"  # "default", "reduce-overhead", "max-autotune"

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

console.rule("[bold blue]Processing PyTorch code with benchmarking[/bold blue]")

clear_temp_files()

weave.init(args.weave_project)


@weave.op
async def call_benchmark_server(pt_code, tests):
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{args.server_url}/run_pytorch", 
                                   json={
                                       "code": pt_code,
                                       "tests": tests,
                                       "benchmark": args.benchmark,
                                       "benchmark_runs": args.benchmark_runs,
                                       "torch_compile": args.torch_compile,
                                       "torch_compile_mode": args.torch_compile_mode
                                   },
                                   timeout=args.timeout)
        resp.raise_for_status()
        return resp.json()

@weave.op
async def run_code(row):
    pt_code = row[args.code_row]
    tests = row[args.tests_row]

    result = await call_benchmark_server(pt_code, tests)

    # Extract all the benchmark information and add it to the result
    enhanced_result = {
        # Original fields
        "pt_code": pt_code,
        "tests": tests,
        # Server execution results
        "status_code": result["status_code"],
        "stdout": result["stdout"], 
        "stderr": result["stderr"],
        # System metrics
        "gpu_mem_used_gb": result.get("gpu_mem_used_gb"),
        "cpu_percent": result.get("cpu_percent"),
        "ram_percent": result.get("ram_percent"),
        # Benchmark metrics
        "benchmark_mean_time_ms": result.get("benchmark_mean_time_ms"),
        "benchmark_std_time_ms": result.get("benchmark_std_time_ms"),
        "benchmark_memory_peak_mb": result.get("benchmark_memory_peak_mb"),
        "benchmark_successful_runs": result.get("benchmark_successful_runs"),
        # PyTorch-specific metrics (torch.compile)
        "torch_compile_benchmark_mean_time_ms": result.get("torch_compile_benchmark_mean_time_ms"),
        "torch_compile_benchmark_std_time_ms": result.get("torch_compile_benchmark_std_time_ms"),
        "torch_compile_speedup": result.get("torch_compile_speedup"),
        # Success flag for easy filtering
        "execution_success": result["status_code"] == 0,
        "has_benchmark_data": result.get("benchmark_mean_time_ms") is not None,
        "has_torch_compile_data": result.get("torch_compile_benchmark_mean_time_ms") is not None,
    }
    
    # Add any other fields from the original row that we want to preserve
    for key, value in row.items():
        if key not in enhanced_result:
            enhanced_result[key] = value

    return enhanced_result
    

console.rule("[bold blue]Processing dataset[/bold blue]")

if args.debug:
    input_ds = input_ds.select(range(10))

async def process_ds():
    pds_list = await map(input_ds, run_code, num_proc=2 if args.debug else args.num_proc)
    pds = Dataset.from_list(pds_list)
    return pds

pds = asyncio.run(process_ds())

# Save locally with benchmark info in the name
output_name = args.output_dataset.replace("/", "_")
if args.benchmark:
    output_name += "_benchmarked"
if args.torch_compile:
    output_name += "_compiled"

pds.save_to_disk(output_name)

if args.push:
    pds.push_to_hub(args.output_dataset)
    console.print(f"Pushed to hub: {args.output_dataset}")

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
    if args.torch_compile:
        console.print(f"  With torch.compile data: {torch_compile_count}/{len(pds)} ({torch_compile_count/len(pds)*100:.1f}%)")

    # print the first row
    console.print(pds[0])
