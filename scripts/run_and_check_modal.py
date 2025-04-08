"""
Runs kernel evaluation using Modal.

Example Usage:

# Cuda Example (on an H100 GPU, measuring performance):
modal run scripts/run_and_check_modal.py --ref-src tests/test_data/cuda/model_ex_add.py --custom-src tests/test_data/cuda/model_new_ex_add.py --gpu H100 --measure-performance

# Triton Example (on an H100 GPU, measuring performance):
modal run scripts/run_and_check_modal.py --ref-src tests/test_data/triton/embed_code.py --custom-src tests/test_data/triton/embed_triton.py --ref-entry-point LinearEmbedding --custom-entry-point LinearEmbeddingNew --gpu H100 --measure-performance

"""
import simple_parsing as sp
from dataclasses import dataclass, field
import torch
from rich.console import Console
from rich.pretty import pprint
import os
import modal


GPU_TYPE = "H100!"

# Configure Modal image
cuda_version = "12.8.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.10")
    .pip_install(
        "simple_parsing", # Keep for local parsing if needed outside main
        "rich",
        "torch", # Ensure torch is available in the Modal environment
        # Add other direct dependencies if needed
    )
    # Add the local triton_eval package.
    # Assumes 'triton_eval' directory is at the workspace root and contains an installable package.
    # If it's just a directory, use .add_local_dir("triton_eval", "/root/triton_eval")
    # and potentially manage sys.path inside the Modal function.
    .add_local_python_source("triton_eval")
)

app = modal.App("triton-eval-runner", image=image)

# Import necessary functions *inside* the Modal context
with image.imports():
    # Note: local_read_file is defined in main now
    from triton_eval.eval import eval_kernel_against_ref, detect_backend
    from triton_eval.utils import set_gpu_arch # Import the utility function
    import torch # Ensure torch is imported within the Modal context as well


# ScriptArgs dataclass removed as @app.local_entrypoint handles args


@app.cls(gpu=GPU_TYPE) # Request any available GPU, timeout set at instantiation
class ModalEvaluator:
    @modal.enter()
    def setup(self):
        # Pre-computation or setup can happen here if needed
        # e.g., check GPU availability
        print(f"Modal container started on GPU: {torch.cuda.get_device_name(0)}")
        pass

    @modal.method()
    def run_evaluation(
        self,
        ref_src_code: str,
        custom_src_code: str,
        ref_entry_point: str,
        custom_entry_point: str,
        num_correct_trials: int,
        num_perf_trials: int,
        verbose: bool,
        measure_performance: bool,
    ):
        """Runs the kernel evaluation remotely on Modal."""
        console = Console() # Use console inside modal for structured logging if needed
        console.rule("[bold blue]Inside Modal: Starting Evaluation[/bold blue]")

        backend = detect_backend(custom_src_code)
        console.print(f"Detected backend: [yellow]{backend}[/yellow]")

        # Device will always be cuda:0 within the Modal container
        device = torch.device("cuda:0")
        console.print(f"Using device: [blue]{device} ({torch.cuda.get_device_name(device)})[/blue]")

        # Build directory within the ephemeral Modal container filesystem
        kernel_hash = str(hash(custom_src_code))
        build_dir = os.path.join("/tmp", "modal_build", kernel_hash)
        console.print(f"Build directory: [yellow]{build_dir}[/yellow]")
        os.makedirs(build_dir, exist_ok=True) # Create it if it doesn't exist

        eval_result = eval_kernel_against_ref(
            original_model_src=ref_src_code,
            custom_model_src=custom_src_code,
            original_entry_point=ref_entry_point,
            custom_entry_point=custom_entry_point,
            seed_num=42, # Keep seed consistent
            num_correct_trials=num_correct_trials,
            num_perf_trials=num_perf_trials,
            verbose=verbose,
            measure_performance=measure_performance,
            build_dir=build_dir,
            device=device,
            backend=backend,
        )

        console.rule("[bold green]Inside Modal: Evaluation Complete[/bold green]")
        return eval_result


# Use local_entrypoint for easy CLI usage (e.g., --help)
@app.local_entrypoint()
def main(
    ref_src: str,
    custom_src: str,
    ref_entry_point: str = "Model",
    custom_entry_point: str = "ModelNew",
    verbose: bool = False,
    gpu: str = GPU_TYPE,
    measure_performance: bool = False,
    num_correct_trials: int = 1,
    num_perf_trials: int = 10,
    timeout: int = 600,
):
    """Runs kernel evaluation using Modal."""

    # Use local read_file utility
    from triton_eval.utils import read_file as local_read_file

    console = Console()

    console.rule("[bold blue]Loading Source Code Locally[/bold blue]")
    console.print(f"Loading reference source from: [cyan]{ref_src}[/cyan]")
    console.print(f"Loading custom source from: [cyan]{custom_src}[/cyan]")

    ref_src_code = local_read_file(ref_src)
    custom_src_code = local_read_file(custom_src)

    if not ref_src_code or not custom_src_code:
        console.print("[bold red]Error:[/bold red] Failed to read one or both source files locally.")
        exit(1)

    console.rule("[bold blue]Preparing Modal Evaluation[/bold blue]")
    console.print(f"Targeting Modal GPU: [yellow]{gpu}[/yellow]")
    console.print(f"Timeout set to: [yellow]{timeout}s[/yellow]")

    console.rule("[bold green]Starting Modal Evaluation Remotely[/bold green]")
    try:
        eval_result = ModalEvaluator.with_options(gpu=gpu, timeout=timeout)().run_evaluation.remote(
            ref_src_code=ref_src_code,
            custom_src_code=custom_src_code,
            ref_entry_point=ref_entry_point,
            custom_entry_point=custom_entry_point,
            num_correct_trials=num_correct_trials,
            num_perf_trials=num_perf_trials,
            verbose=verbose,
            measure_performance=measure_performance,
        )
    except Exception as e:
        # Catch Modal-specific exceptions if needed, or general ones
        console.print(f"[bold red]Error during Modal execution:[/bold red] {e}")
        # Consider more specific error handling if necessary
        exit(1)


    console.rule("[bold green]Modal Evaluation Result[/bold green]")
    if eval_result:
        # Use rich's pretty print for the result (which might be a Pydantic model or dict)
        pprint(eval_result)
    else:
        console.print("[bold yellow]Warning:[/bold yellow] Modal evaluation function returned None.")
