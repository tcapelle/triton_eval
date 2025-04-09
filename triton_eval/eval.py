"""
Helpers for Evaluations
"""

import importlib
import numpy as np
import os
import subprocess
import tempfile
import torch
import torch.nn as nn
from pydantic import BaseModel

from triton_eval.utils import to_device

def set_seed(seed: int):
    torch.manual_seed(seed)
    # NOTE: this only sets on current cuda device
    torch.cuda.manual_seed(seed)


def get_error_name(e: Exception) -> str:

    return f"{e.__class__.__module__}.{e.__class__.__name__}"


class KernelExecResult(BaseModel):
    """
    Single Kernel Execution
    """

    compiled: bool = False
    correctness: bool = False
    metadata: dict = {}
    runtime: float = -1.0  # in us, only recorded if we decide to measure performance
    runtime_stats: dict = {}  # only recorded if we decide to measure performance


def load_reference_model_and_inputs(
    model_original_src: str, entry_point: str, context: dict
) -> tuple[nn.Module, callable, callable]:
    """
    Load class from original NN.module pytorch code (reference implementation).

    Args:
        model_original_src: The source code string of the original model.
        entry_point: The name of the nn.Module class to load.
        context: The execution context (dictionary) for the source code.

    Returns:
        A tuple containing the loaded Model class, the get_init_inputs function,
        and the get_inputs function, or None if loading fails.
    """
    try:
        compile(model_original_src, "<string>", "exec")
    except SyntaxError as e:
        print(f"Syntax Error in original code {e}")
        return None

    try:
        exec(model_original_src, context)  # expose to current namespace
    except Exception as e:
        print(f"Error in executing original code {e}")
        return None

    # these should be defined in the original model code and present in the context
    get_init_inputs_fn = context.get("get_init_inputs")
    get_inputs_fn = context.get("get_inputs")
    # Get the class using the provided entry point name
    ModelClass = context.get(entry_point)
    if not ModelClass:
        # Print a more specific error message
        print(f"Error: Class '{entry_point}' not found in context after executing original source. Available keys: {list(context.keys())}")
        return None

    return (ModelClass, get_init_inputs_fn, get_inputs_fn)


def load_custom_cuda_model(
    model_custom_src: str, entry_point: str, context: dict, build_directory: str = None
) -> nn.Module:
    """
    Load class from custom NN.module pytorch code
    this is the code output by LLM with calls to custom cuda kernels

    Args:
        model_custom_src: The source code string of the custom model.
        entry_point: The name of the nn.Module class to load.
        context: The execution context (dictionary) for the source code.
        build_directory: Optional path to the build directory for extensions.
    """
    if build_directory:
        context["BUILD_DIRECTORY"] = build_directory
        # Add import at the start of the source code
        model_custom_src = (
            "import os\n" f"os.environ['TORCH_EXTENSIONS_DIR'] = '{build_directory}'\n"
        ) + model_custom_src

    try:
        compile(model_custom_src, "<string>", "exec")
        exec(model_custom_src, context)
        # DANGER: need to delete refernece from global namespace
    except SyntaxError as e:
        print(f"Syntax Error in custom generated code or Compilation Error {e}")
        return None
    except Exception as e: # Catch other execution errors
        print(f"Error executing custom code: {e}")
        return None

    # Get the class using the provided entry point name
    ModelClass = context.get(entry_point)
    if not ModelClass:
        print(f"Error: Class '{entry_point}' not found in context after executing custom source.")
        return None

    return ModelClass


def load_custom_triton_model(model_custom_src, entry_point="ModelNew"):
    """
    Writes the provided Python code string to a temporary .py file,
    dynamically imports the module so we can access the modified model class.
    Returns both a Model class and the temporary file. The temporary file must be
    deleted manually be the caller.
    This is a hack that is needed for triton code as compile / exec do not play well
    with the @triton.jit decorator. Need to provide the specific class name via entry_point.
    """
    temp_file_obj = None
    tempfile_path = None
    # Create a temporary named file with a .py extension
    # Need a try block to ensure cleanup even if writing fails
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp_file:
            tmp_file.write(model_custom_src)
            tempfile_path = tmp_file.name
            temp_file_obj = tmp_file # Keep the file object to return later

        # Create a module specification pointing to our temp file
        spec = importlib.util.spec_from_file_location("temp_module", tempfile_path)
        # Create a new module based on that spec
        temp_module = importlib.util.module_from_spec(spec)
        # Execute the code in the module's namespace
        spec.loader.exec_module(temp_module)

        # --- Added Specific Check ---
        try:
            ModelNew = getattr(temp_module, entry_point)
        except AttributeError:
            print(f"Error: Custom entry point '{entry_point}' not found in the loaded temporary module.")
            # Clean up before returning None
            if temp_file_obj:
                try:
                    temp_file_obj.close()
                    os.remove(temp_file_obj.name)
                except OSError as cleanup_e:
                    print(f"Error during cleanup of temp file {temp_file_obj.name}: {cleanup_e}")
            return None, None # Indicate failure to find entry point
        # --- End Specific Check ---

        # Return the object (class, function, etc.) that was defined in the code
        return ModelNew, temp_file_obj
    except Exception as e:
        # This will now catch errors other than AttributeError during getattr
        print(f"Error loading custom model from temp file (e.g., import error, syntax error): {e}")
        # Clean up if temp file was created and object exists
        if temp_file_obj:
            try:
                temp_file_obj.close()
                os.remove(temp_file_obj.name)
            except OSError as cleanup_e:
                print(f"Error during cleanup of temp file {temp_file_obj.name}: {cleanup_e}")
        elif 'tempfile_path' in locals() and os.path.exists(tempfile_path): # Handle case where file created but object not assigned
             try:
                 os.remove(tempfile_path)
             except OSError as cleanup_e:
                 print(f"Error during cleanup of temp file path {tempfile_path}: {cleanup_e}")

        return None, None # Return None for both model and tempfile


def graceful_eval_cleanup(
    curr_context: dict,
    device: torch.device,
    tempfile: tempfile.NamedTemporaryFile = None,
):
    """
    Clean up env, gpu cache, and compiled CUDA extensions after evaluation
    """  # delete ran-specific function definitions before next eval run
    del curr_context
    # Clear CUDA cache and reset GPU state
    with torch.cuda.device(device):
        torch.cuda.empty_cache()

        # does this help?
        torch.cuda.reset_peak_memory_stats(device=device)

        torch.cuda.synchronize(
            device=device
        )  # Wait for all CUDA operations to complete
    if tempfile:
        tempfile.close()
    return True


def _parse_inputs(inputs_raw: list | tuple, device: torch.device) -> tuple[list, dict]:
    """Parses raw inputs which can be a list of args or [args_list, kwargs_dict].

    Moves tensors to the specified device.

    Returns:
        tuple[list, dict]: A tuple containing the processed args list and kwargs dict.
    """
    if (
        isinstance(inputs_raw, (list, tuple))
        and len(inputs_raw) == 2
        and isinstance(inputs_raw[0], (list, tuple))
        and isinstance(inputs_raw[1], dict)
    ):
        args_list, kwargs_dict = inputs_raw
    else:
        # Assume flat list of args
        args_list = inputs_raw
        kwargs_dict = {}

    # Move inputs to the target device
    args = to_device(list(args_list), device) # Ensure args is a list
    kwargs = to_device(kwargs_dict, device)

    return args, kwargs


def eval_kernel_against_ref(
    original_model_src: str,
    custom_model_src: str,
    original_entry_point: str,
    custom_entry_point: str,
    seed_num: int = 42,
    num_correct_trials: int = 1,
    num_perf_trials: int = 10,
    verbose: bool = False,
    measure_performance: bool = False,
    build_dir: os.PathLike = None,
    device: torch.device | int | None = (
        torch.cuda.current_device() if torch.cuda.is_available() else None
    ),  # have to run on GPU
    backend: str = "cuda",  # can be 'cuda' or 'triton', determines which backend implementation to use
) -> KernelExecResult:
    """
    Evaluate the custom kernel against the original model

    num_correct_trials: number of trials to initialize different random inputs; correctness pass only if all trials pass
    num_perf_trials: run the evalutation many times to take the average
    device: GPU (cuda) device to run the evalutation on
    backend: str, either 'cuda' or 'triton', determines which backend implementation to use
    """
    # TODO: check device is busy
    assert torch.cuda.is_available(), "CUDA is not available, cannot run Eval"
    torch.set_printoptions(
        precision=4,  # Decimal places
        threshold=10,  # Total number of elements before truncating
        edgeitems=3,  # Number of elements at beginning and end of dimensions
        linewidth=80,  # Maximum width before wrapping
    )

    # set CUDA device
    torch.cuda.set_device(device)
    is_triton = backend == "triton"
    metadata = {}  # for storing result metadata
    metadata["hardware"] = torch.cuda.get_device_name(device=device)
    metadata["device"] = str(device)  # for debugging

    if is_triton:
        # need to set env var for triton code to guarentee no wrong device shennanignas
        if isinstance(device, int):
            device_num = device
        elif isinstance(device, torch.device):
            assert (
                device.type == "cuda"
            ), "CUDA is not availible on device, cannot run Eval"
            device_num = device.index
        else:
            raise ValueError(
                f"device must be an int or torch.device, got {type(device)}"
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_num)
    context = {}

    if verbose:
        print(f"[Eval] Start Evalulation! on device: {device}")
        print("[Eval] Loading Original Model")
    load_result = load_reference_model_and_inputs(
        original_model_src, original_entry_point, context
    )
    if load_result is None:
        print(f"[Eval Error] Failed to load reference model or inputs using entry point '{original_entry_point}'. Check source code and entry point name.")
        metadata["load_error"] = f"Reference model entry point '{original_entry_point}' not found or source failed execution."
        graceful_eval_cleanup(context, device) # Clean up before returning
        return KernelExecResult(compiled=False, correctness=False, metadata=metadata)

    Model, get_init_inputs, get_inputs = load_result

    set_seed(seed_num)  # set seed for reproducible input
    init_inputs_raw = get_init_inputs()
    init_args, init_kwargs = _parse_inputs(init_inputs_raw, device)

    with torch.no_grad():
        set_seed(seed_num)  # set seed for reproducible weights
        original_model = Model(*init_args, **init_kwargs)
        assert hasattr(original_model, "forward")
        if verbose:
            print("[Eval] Original Model Loaded")
    if verbose:
        print("[Eval] Loading and Compiling New Model with Custom CUDA Kernel")

    # this is where compilation happens
    try:
        os.environ["TORCH_USE_CUDA_DSA"] = "1"  # compile with device side assertion
        tempfile = None
        # add hash for later to distinguish between multi-turn kernels
        if is_triton:
            ModelNew, tempfile = load_custom_triton_model(
                custom_model_src, custom_entry_point
            )
            if ModelNew is None: # Check if loading failed
                 print("Failed to load custom model from tempfile.")
                 graceful_eval_cleanup(context, device, tempfile) # Cleanup even if tempfile might be None
                 return KernelExecResult(compiled=False, metadata=metadata) # Indicate failure
        else:
            ModelNew = load_custom_cuda_model(custom_model_src, custom_entry_point, context, build_dir)
        torch.cuda.synchronize(device=device)  # not sure if this is too much
    except Exception as e:
        print(
            f"Failed to compile custom CUDA kernel: Record as compilation failure. \nError: {e}"
        )
        # TODO: add metadata for compilation error (how to we get the compilation error message?)

        if "lock" in str(e) or "No such file or directory" in str(e):
            # this is a lock file error, likely due to concurrent compilation
            # this does not necessarily mean the compilation failed, but we should retry
            print(
                f"[Eval] Lock file error during compilation, Please retry. Error: {e}"
            )
            graceful_eval_cleanup(context, device, tempfile)
            return None
        else:
            metadata["compilation_error_name"] = get_error_name(e)
            metadata["compilation_error"] = e
            graceful_eval_cleanup(context, device, tempfile)
            return KernelExecResult(
                compiled=False, metadata=metadata
            )  # skip further steps

    # at this point we passed compilation
    try:
        with torch.no_grad():
            set_seed(seed_num)  # set seed for reproducible weights
            # Use the processed init args and kwargs
            custom_model = ModelNew(*init_args, **init_kwargs)
            assert hasattr(custom_model, "forward")
            torch.cuda.synchronize(device=device)
        if verbose:
            print("[Eval] New Model with Custom CUDA Kernel Loaded")
    except RuntimeError as e:
        print(
            f"Failed to load custom CUDA kernel; Compiled but not able to run, count as runtime error. \nError: {e}"
        )
        # TODO: add metadata for runtime error e.g. error in launching kernel, illegal memory access, ...
        graceful_eval_cleanup(context, device, tempfile)
        metadata["runtime_error"] = e
        metadata["runtime_error_name"] = get_error_name(e)
        return KernelExecResult(
            compiled=True, correctness=False, metadata=metadata
        )  # skip further steps

    kernel_exec_result = None

    # Check Correctness
    if verbose:
        print("[Eval] Checking Correctness")
    try:
        kernel_exec_result = run_and_check_correctness(
            original_model,
            custom_model,
            get_inputs,
            metadata=metadata,
            num_correct_trials=num_correct_trials,
            verbose=verbose,
            seed=seed_num,
            device=device,
        )
    except Exception as e:
        # TODO: add metadata for runtime error e.g. error in launching kernel, illegal memory access, ...
        metadata["runtime_error"] = e
        metadata["runtime_error_name"] = get_error_name(e)
        kernel_exec_result = KernelExecResult(
            compiled=True, correctness=False, metadata=metadata
        )

    # Measure Performance [Optional] | conditioned on compilation + correctness + no exception so far
    if measure_performance:
        try:
            if kernel_exec_result and kernel_exec_result.correctness:
                if verbose:
                    print("[Eval] Measuring Performance as Sample is Correct")

                torch.cuda.synchronize(device=device)
                set_seed(seed_num)
                inputs_raw = get_inputs()
                fwd_args, fwd_kwargs = _parse_inputs(inputs_raw, device)

                model_new = custom_model.cuda(device=device)
                torch.cuda.synchronize(device=device)

                # Pass processed args and kwargs to timing function
                elapsed_times = time_execution_with_cuda_event(
                    model_new,
                    fwd_args, # Pass args list
                    fwd_kwargs, # Pass kwargs dict
                    num_trials=num_perf_trials,
                    verbose=verbose,
                    device=device,
                )
                runtime_stats = get_timing_stats(elapsed_times, device=device)

                if verbose:
                    print(f"[Eval] Performance Stats: {runtime_stats}")
                kernel_exec_result.runtime = runtime_stats["mean"]
                kernel_exec_result.runtime_stats = runtime_stats
        except Exception as e:
            if verbose:
                print(f"[Eval] Error in Measuring Performance: {e}")
            kernel_exec_result.metadata["error_during_performance"] = e

    graceful_eval_cleanup(context, device, tempfile)
    return kernel_exec_result


def register_and_format_exception(
    exception_type: str,
    exception_msg: Exception | str,
    metadata: dict,
    verbose: bool = False,
    truncate=False,
    max_length=200,
):
    """
    max_length characters

    NOTE: I can't get torch truncate to work during exception handling so I have this for now
    """
    # Truncate exception message if too long
    exception_str = str(exception_msg)
    if truncate and len(exception_str) > max_length:
        exception_str = exception_str[: max_length - 3] + "..."

    if verbose:
        print(f"[Exception {exception_type}] {exception_str} ")
    metadata[exception_type] = exception_str

    return metadata


def time_execution_with_cuda_event(
    kernel_fn: callable,
    fwd_args: list, # Explicitly expect args list
    fwd_kwargs: dict, # Explicitly expect kwargs dict
    num_warmup: int = 3,
    num_trials: int = 10,
    verbose: bool = True,
    device: torch.device = None,
) -> list[float]:
    """
    Time a CUDA kernel function over multiple trials using torch.cuda.Event

    Args:
        kernel_fn: Function to time (typically model.forward)
        fwd_args: Positional arguments for kernel_fn
        fwd_kwargs: Keyword arguments for kernel_fn
        num_warmup: Number of warmup runs
        num_trials: Number of timing trials to run
        verbose: Whether to print per-trial timing info
        device: CUDA device to use, if None, use current device

    Returns:
        List of elapsed times in milliseconds
    """
    if device is None:
        device = torch.cuda.current_device()

    # Warm ups
    for _ in range(num_warmup):
        # Call with args and kwargs
        kernel_fn(*fwd_args, **fwd_kwargs)
        torch.cuda.synchronize(device=device)

    print(
        f"[Profiling] Using device: {device} {torch.cuda.get_device_name(device)}, warm up {num_warmup}, trials {num_trials}"
    )
    elapsed_times = []

    # Actual trials
    for trial in range(num_trials):
        # create event marker default is not interprocess
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        # Call with args and kwargs
        kernel_fn(*fwd_args, **fwd_kwargs)
        end_event.record()

        # Synchronize to ensure the events have completed
        torch.cuda.synchronize(device=device)

        # Calculate the elapsed time in milliseconds
        elapsed_time_ms = start_event.elapsed_time(end_event)
        if verbose:
            print(f"Trial {trial + 1}: {elapsed_time_ms:.3g} ms")
        elapsed_times.append(elapsed_time_ms)

    return elapsed_times


def run_and_check_correctness(
    original_model_instance: nn.Module,
    new_model_instance: nn.Module,
    get_inputs_fn: callable,
    metadata: dict,
    num_correct_trials: int,
    verbose=False,
    seed=42,
    device=None,
) -> KernelExecResult:
    """
    run the model and check correctness,
    assume model already loaded and compiled (loaded and compiled in the caller)
    this is all on GPU, requiring cuda device and transfer .cuda()

    num_correct_trials: run the evalutation multiple times with (ideally) different random inputs to ensure correctness
    """
    pass_count = 0

    # Generate num_correct_trials seeds deterministically from the initial seed
    torch.manual_seed(seed)
    correctness_trial_seeds = [
        torch.randint(0, 2**32 - 1, (1,)).item() for _ in range(num_correct_trials)
    ]

    with torch.no_grad():

        for trial in range(num_correct_trials):

            trial_seed = correctness_trial_seeds[trial]
            if verbose:
                print(f"[Eval] Generating Random Input with seed {trial_seed}")

            set_seed(trial_seed)
            inputs_raw = get_inputs_fn()
            fwd_args, fwd_kwargs = _parse_inputs(inputs_raw, device)

            set_seed(trial_seed)
            model = original_model_instance.cuda(device=device)

            set_seed(trial_seed)
            model_new = new_model_instance.cuda(device=device)

            # Call model with processed args and kwargs
            output = model(*fwd_args, **fwd_kwargs)
            torch.cuda.synchronize(device=device)
            # ensure all GPU operations are completed before checking results

            try:
                # Call new model with processed args and kwargs
                output_new = model_new(*fwd_args, **fwd_kwargs)
                torch.cuda.synchronize(device=device)
                if output.shape != output_new.shape:
                    metadata = register_and_format_exception(
                        "correctness_issue",
                        f"Output shape mismatch: Expected {output.shape}, got {output_new.shape}",
                        metadata,
                    )
                    if verbose:
                        print(
                            f"[FAIL] trial {trial}: Output shape mismatch: Expected {output.shape}, got {output_new.shape}"
                        )
                    return KernelExecResult(
                        compiled=True, correctness=False, metadata=metadata
                    )

                # check output value difference
                if not torch.allclose(
                    output, output_new, atol=1e-02, rtol=1e-02
                ):  # fail
                    max_diff = torch.max(torch.abs(output - output_new)).item()
                    avg_diff = torch.mean(torch.abs(output - output_new)).item()
                    metadata.setdefault("max_difference", []).append(f"{max_diff:.6f}")
                    metadata.setdefault("avg_difference", []).append(f"{avg_diff:.6f}")
                    metadata["correctness_issue"] = "Output mismatch"
                    if verbose:
                        print(f"[FAIL] trial {trial}: Output mismatch")
                else:  # pass
                    pass_count += 1
                    if verbose:
                        print(f"[PASS] trial {trial}: New Model matches Model")

            except Exception as e:
                print("[Error] Exception happens during correctness check")
                print(f"Error in launching kernel for ModelNew: {e}")

                metadata = register_and_format_exception(
                    "runtime_error", e, metadata, truncate=True
                )
                return KernelExecResult(
                    compiled=True, correctness=False, metadata=metadata
                )
                # break

    if verbose:
        print(
            f"[Eval] Pass count: {pass_count}, num_correct_trials: {num_correct_trials}"
        )

    # put all the useful info here!
    metadata["correctness_trials"] = f"({pass_count} / {num_correct_trials})"

    if pass_count == num_correct_trials:
        return KernelExecResult(compiled=True, correctness=True, metadata=metadata)
    else:
        return KernelExecResult(compiled=True, correctness=False, metadata=metadata)



################################################################################
# Performance Eval
################################################################################


def get_timing_stats(elapsed_times: list[float], device: torch.device = None) -> dict:
    """Get timing statistics from a list of elapsed times.

    Args:
        elapsed_times: List of elapsed times in milliseconds
        device: CUDA device, record device info
    Returns:
        Dict containing mean, std, min, max and num_trials
        all timing are in ms
    """

    stats = {
        "mean": float(f"{np.mean(elapsed_times):.3g}"),
        "std": float(f"{np.std(elapsed_times):.3g}"),
        "min": float(f"{np.min(elapsed_times):.3g}"),
        "max": float(f"{np.max(elapsed_times):.3g}"),
        "num_trials": len(elapsed_times),
    }

    if device:
        stats["hardware"] = torch.cuda.get_device_name(device=device)
        stats["device"] = str(device)  # for debugging

    return stats


def detect_backend(custom_model_src: str) -> str:
    """
    Detects whether the custom model source uses Triton or standard CUDA/PyTorch extensions.

    Args:
        custom_model_src: The source code string of the custom model.

    Returns:
        'triton' if Triton usage is detected, 'cuda' otherwise.
    """
    if "import triton" in custom_model_src or "@triton.jit" in custom_model_src:
        return "triton"
    else:
        return "cuda"
