"""
Runs kernel evaluation using Modal.

Example Usage:

# Cuda Example (on an H100 GPU, measuring performance):
modal run scripts/run_and_check_modal.py \
    --ref_src tests/test_data/cuda/model_ex_add.py \
    --custom_src tests/test_data/cuda/model_new_ex_add.py \
    --gpu H100 

# Triton Example (on an H100 GPU, measuring performance):
modal run scripts/run_and_check_modal.py \
    --ref_src tests/test_data/triton/embed_code.py \
    --custom_src tests/test_data/triton/embed_triton.py \
    --ref_entry_point LinearEmbedding \
    --custom_entry_point LinearEmbeddingNew \
    --gpu H100

"""
import os
from dataclasses import dataclass
import torch
from rich.console import Console
from rich.pretty import pprint
import modal
import simple_parsing as sp

GPU_TYPE = "H100!"
TIMEOUT = 60


@dataclass
class ScriptArgs:
    ref_src: str
    custom_src: str
    ref_entry_point: str = "Model" # Default to "Model" if not provided
    custom_entry_point: str = "ModelNew" # Default to "ModelNew" if not provided
    verbose: bool = False # Print verbose output when running the script
    measure_performance: bool = True # Measure performance of the custom kernel
    num_correct_trials: int = 1 # Number of correct trials to run
    num_perf_trials: int = 10 # Number of performance trials to run
    
    # Modal specific arguments
    timeout: int = TIMEOUT # Timeout for the script
    gpu: str = GPU_TYPE # GPU type to use


# Configure Modal image
cuda_version = "12.6.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
python_version = "3.10"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python=python_version)
    .pip_install(
        "simple_parsing",
        "rich",
        "torch",
    )
    .add_local_python_source("triton_eval")
)

app = modal.App("triton-eval-runner", image=image)

# Import necessary functions *inside* the Modal context
with image.imports():
    from triton_eval.eval import eval_kernel_against_ref, detect_backend
    from triton_eval.utils import set_gpu_arch 
    import torch 


@app.cls(gpu=GPU_TYPE) # we will override this in the local_entrypoint
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


@app.local_entrypoint()
def main(*arglist):
    """Runs kernel evaluation using Modal."""
    
    args = sp.parse(ScriptArgs, args=arglist)
    # Use local read_file utility
    from triton_eval.utils import read_file as local_read_file

    console = Console()

    console.rule("[bold blue]Loading Source Code Locally[/bold blue]")
    console.print(f"Loading reference source from: [cyan]{args.ref_src}[/cyan]")
    console.print(f"Loading custom source from: [cyan]{args.custom_src}[/cyan]")

    ref_src_code = local_read_file(args.ref_src)
    custom_src_code = local_read_file(args.custom_src)

    if not ref_src_code or not custom_src_code:
        console.print("[bold red]Error:[/bold red] Failed to read one or both source files locally.")
        exit(1)

    console.rule("[bold blue]Preparing Modal Evaluation[/bold blue]")
    console.print(f"Targeting Modal GPU: [yellow]{args.gpu}[/yellow]")
    console.print(f"Timeout set to: [yellow]{args.timeout}s[/yellow]")

    console.rule("[bold green]Starting Modal Evaluation Remotely[/bold green]")
    try:
        eval_result = ModalEvaluator.with_options(gpu=args.gpu, timeout=args.timeout)().run_evaluation.remote(
            ref_src_code=ref_src_code,
            custom_src_code=custom_src_code,
            ref_entry_point=args.ref_entry_point,
            custom_entry_point=args.custom_entry_point,
            num_correct_trials=args.num_correct_trials,
            num_perf_trials=args.num_perf_trials,
            verbose=args.verbose,
            measure_performance=args.measure_performance,
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
