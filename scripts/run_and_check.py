import simple_parsing as sp
from dataclasses import dataclass
import torch
from rich.console import Console
from rich.pretty import pprint
import os


from triton_eval.eval import eval_kernel_against_ref, detect_backend

"""
# Cuda
python scripts/run_and_check.py --ref_src tests/test_data/cuda/model_ex_add.py --custom_src tests/test_data/cuda/model_new_ex_add.py

# Triton
python scripts/run_and_check.py --ref_src tests/test_data/triton/embed_code.py --custom_src tests/test_data/triton/embed_triton.py --ref_entry_point LinearEmbedding --custom_entry_point LinearEmbeddingNew
"""

@dataclass
class ScriptArgs:
    ref_src: str  # Path to the reference Python source file
    custom_src: str  # Path to the custom Python source file with the kernel
    ref_entry_point: str = "Model"  # Class name of the reference nn.Module
    custom_entry_point: str = "ModelNew"  # Class name of the custom nn.Module
    device: str = "cuda:0"  # CUDA device to run on (e.g., 'cuda:0')
    verbose: bool = False  # Enable verbose output during evaluation
    measure_performance: bool = False  # Measure kernel performance
    num_correct_trials: int = 1  # Number of trials for correctness checking
    num_perf_trials: int = 10  # Number of trials for performance measurement
    build_dir_prefix: str = "/tmp/triton_eval_builds"  # Prefix for build directories
    clear_cache: bool = False  # Clear build cache before running


def main():
    # Directly parse the ScriptArgs dataclass
    args: ScriptArgs = sp.parse(ScriptArgs)

    console = Console()

    console.rule("[bold blue]Loading Source Code[/bold blue]")
    console.print(f"Loading reference source from: [cyan]{args.ref_src}[/cyan]")
    console.print(f"Loading custom source from: [cyan]{args.custom_src}[/cyan]")

    try:
        with open(args.ref_src, 'r', encoding='utf-8') as f:
            ref_src_code = f.read()
        with open(args.custom_src, 'r', encoding='utf-8') as f:
            custom_src_code = f.read()
    except FileNotFoundError as e:
        console.print(f"[bold red]Error:[/bold red] Could not read source file: {e}")
        exit(1)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] An error occurred reading files: {e}")
        exit(1)

    console.rule("[bold blue]Backend Detection & Device Setup[/bold blue]")
    backend = detect_backend(custom_src_code)
    console.print(f"Detected backend: [yellow]{backend}[/yellow]")

    if not torch.cuda.is_available():
        console.print("[bold red]Error:[/bold red] CUDA is not available.")
        exit(1)

    try:
        device = torch.device(args.device)
        console.print(f"Using device: [blue]{device} ({torch.cuda.get_device_name(device)})[/blue]")
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] Invalid device string '{args.device}': {e}")
        exit(1)

    # Create a dictionary for eval config matching evaluate_single_sample_src expectations
    # Although we call eval_kernel_against_ref directly, it expects build_dir etc.
    # NOTE: This dict isn't strictly necessary anymore as args holds the values directly,
    # but keeping it for clarity in mapping to build_dir logic if needed elsewhere.
    eval_configs = {
        "num_correct_trials": args.num_correct_trials,
        "num_perf_trials": args.num_perf_trials,
        "verbose": args.verbose,
        "measure_performance": args.measure_performance,
        "build_dir_prefix": args.build_dir_prefix,
        "clear_cache": args.clear_cache, # This specific flag is handled in evaluate_single_sample_src, but we pass build_dir below
    }

    console.rule("[bold blue]Build Directory Setup[/bold blue]")
    # Construct build directory path (similar logic to evaluate_single_sample_src)
    kernel_hash = str(hash(custom_src_code))
    # Use args directly for build_dir_prefix
    build_dir = os.path.join(args.build_dir_prefix, "run_script_build", kernel_hash)
    console.print(f"Build directory: [yellow]{build_dir}[/yellow]")


    if args.clear_cache:
        import shutil
        console.print(f"Clearing build cache directory: [yellow]{build_dir}[/yellow]")
        shutil.rmtree(build_dir, ignore_errors=True)
    os.makedirs(build_dir, exist_ok=True)


    console.rule("[bold green]Starting Evaluation[/bold green]")

    eval_result = eval_kernel_against_ref(
        original_model_src=ref_src_code,
        custom_model_src=custom_src_code,
        original_entry_point=args.ref_entry_point,
        custom_entry_point=args.custom_entry_point,
        seed_num=42, # Keep seed consistent for now
        num_correct_trials=args.num_correct_trials,
        num_perf_trials=args.num_perf_trials,
        verbose=args.verbose,
        measure_performance=args.measure_performance,
        build_dir=build_dir,
        device=device,
        backend=backend,
    )

    console.rule("[bold green]Evaluation Result[/bold green]")
    if eval_result:
        # Use rich's pretty print for the Pydantic model
        pprint(eval_result)
    else:
        console.print("[bold yellow]Warning:[/bold yellow] Evaluation function returned None (potentially due to a lock file error during compilation).")


if __name__ == "__main__":
    main()
