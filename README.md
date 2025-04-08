# Evaluating Custom Kernels (Cuda and Triton)

This script evaluates a custom kernel implementation (e.g., written in Triton or CUDA via Python) against a reference PyTorch implementation for correctness and optionally performance.

## Usage

```bash
python scripts/run_and_check.py --ref_src <path_to_ref.py> --custom_src <path_to_custom.py> [OPTIONS]
```

## Arguments

The script uses `simple_parsing` to define its arguments. Here are the available options:

*   `--ref_src` (str, required): Path to the reference Python source file containing the reference `nn.Module`.
*   `--custom_src` (str, required): Path to the custom Python source file containing the `nn.Module` with the kernel to be evaluated.
*   `--ref_entry_point` (str, default: `"Model"`): Class name of the reference `nn.Module` within the `ref_src` file.
*   `--custom_entry_point` (str, default: `"ModelNew"`): Class name of the custom `nn.Module` within the `custom_src` file.
*   `--device` (str, default: `"cuda:0"`): The CUDA device to run the evaluation on (e.g., 'cuda:0', 'cuda:1').
*   `--verbose` (bool, flag, default: `False`): Enable verbose output during evaluation, showing detailed comparison results.
*   `--measure_performance` (bool, flag, default: `False`): Enable performance measurement alongside correctness checks.
*   `--num_correct_trials` (int, default: `1`): Number of independent trials to run for correctness checking. Each trial uses different random inputs.
*   `--num_perf_trials` (int, default: `10`): Number of independent trials to run for performance measurement.
*   `--build_dir_prefix` (str, default: `"/tmp/triton_eval_builds"`): Prefix for the directory where compiled kernels and temporary build files will be stored. A unique subdirectory based on the kernel source hash will be created under this prefix.
*   `--clear_cache` (bool, flag, default: `False`): If set, the specific build cache directory for the current `custom_src` will be removed before running the evaluation.

## Examples

### CUDA Kernel Evaluation

```bash
python scripts/run_and_check.py \
    --ref_src tests/test_data/cuda/model_ex_add.py \
    --custom_src tests/test_data/cuda/model_new_ex_add.py
```
*(Assumes default entry point names "Model" and "ModelNew")*

### Triton Kernel Evaluation

```bash
python scripts/run_and_check.py \
    --ref_src tests/test_data/triton/embed_code.py \
    --custom_src tests/test_data/triton/embed_triton.py \
    --ref_entry_point LinearEmbedding \
    --custom_entry_point LinearEmbeddingNew \
    --measure_performance
```
![](assets/triton.png)

This command evaluates the `LinearEmbeddingNew` module from `embed_triton.py` against the `LinearEmbedding` module from `embed_code.py`, also measuring performance.
