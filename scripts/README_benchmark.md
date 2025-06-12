# PyTorch Dataset Benchmarking

This updated `run_ds.py` script processes datasets containing PyTorch code and runs them through the server with benchmarking and `torch.compile` support.

## Features

- **Execution**: Runs PyTorch code on GPU workers via the server
- **Benchmarking**: Measures execution time and memory usage
- **torch.compile**: Optional compilation and speedup measurement
- **Comprehensive Metrics**: Collects system metrics, benchmark results, and compilation data

## Usage

```bash
python scripts/run_ds.py \
    --input_dataset "your/dataset" \
    --output_dataset "your/benchmarked_dataset" \
    --benchmark true \
    --torch_compile true \
    --benchmark_runs 10
```

## Configuration Options

```python
@dataclass
class Args:
    debug: bool = False                    # Process only 10 samples for testing
    input_dataset: str = "..."             # HuggingFace dataset or local path
    output_dataset: str = "..."            # Output dataset name
    weave_project: str = "..."             # Weave project for tracking
    push: bool = True                      # Push to HuggingFace Hub
    num_proc: int = 10                     # Number of parallel processes
    timeout: int = 60                      # Timeout per request
    server_url: str = "http://127.0.0.1:9347"  # Server URL
    
    # Benchmarking options
    benchmark: bool = True                 # Enable benchmarking
    benchmark_runs: int = 10               # Number of benchmark runs
    torch_compile: bool = True             # Enable torch.compile benchmarking
    torch_compile_mode: str = "default"    # Compilation mode
```

## Dataset Format

Your input dataset should have the following structure:

```python
{
    "pt_code": "...",    # PyTorch code to execute
    "tests": "...",      # Test code with benchmark_function
    # ... other fields preserved in output
}
```

### Benchmarking Approach

The benchmarking system measures the execution time of your **entire** `pt_code + tests` combined. This means:

- The server will re-execute your complete code+tests multiple times to get timing statistics
- Your `tests` field should contain the actual workload you want to benchmark
- No special `benchmark_function()` is needed - the tests themselves are the benchmark

```python
# Example tests field
tests = '''
import torch

def test_functionality():
    # Your test code here - this will be benchmarked
    x = torch.randn(10, 10, device='cuda')
    result = your_function(x)
    print(f"Test result: {result.shape}")
    return result

# Run your tests - this entire execution is benchmarked
test_functionality()

# Add more compute if you want more substantial benchmarking
for i in range(100):
    large_x = torch.randn(1000, 1000, device='cuda')
    large_result = your_function(large_x)

print("Benchmarking workload completed")
'''
```

## Output Dataset

The processed dataset will include all original fields plus:

### Execution Results
- `status_code`: Execution status (0 = success)
- `stdout`: Standard output from execution
- `stderr`: Error output (if any)
- `execution_success`: Boolean flag for easy filtering

### System Metrics
- `gpu_mem_used_gb`: GPU memory usage in GB
- `cpu_percent`: CPU usage percentage
- `ram_percent`: RAM usage percentage

### Benchmark Results
- `benchmark_mean_time_ms`: Average execution time in milliseconds
- `benchmark_std_time_ms`: Standard deviation of execution times
- `benchmark_memory_peak_mb`: Peak memory usage during benchmarking
- `benchmark_successful_runs`: Number of successful benchmark runs
- `has_benchmark_data`: Boolean flag indicating if benchmark data is available

### torch.compile Results (if enabled)
- `torch_compile_benchmark_mean_time_ms`: Compiled version execution time
- `torch_compile_benchmark_std_time_ms`: Compiled version time std dev
- `torch_compile_speedup`: Speedup ratio (regular_time / compiled_time)
- `has_torch_compile_data`: Boolean flag for torch.compile data availability

## Example Complete Sample

```python
sample = {
    "pt_code": '''
import torch
import torch.nn.functional as F

def efficient_attention(query, key, value):
    """Efficient attention implementation"""
    scores = torch.matmul(query, key.transpose(-2, -1))
    scores = scores / (query.size(-1) ** 0.5)
    attn_weights = F.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, value)
''',
    
    "tests": '''
import torch
torch.manual_seed(42)

def test_attention():
    batch_size, seq_len, d_model = 2, 10, 64
    
    query = torch.randn(batch_size, seq_len, d_model, device='cuda')
    key = torch.randn(batch_size, seq_len, d_model, device='cuda')  
    value = torch.randn(batch_size, seq_len, d_model, device='cuda')
    
    result = efficient_attention(query, key, value)
    print(f"Attention output shape: {result.shape}")
    
    assert result.shape == (batch_size, seq_len, d_model)
    print("✅ Attention test passed")
    return result

# Run tests
test_attention()

# Run additional benchmarking workload with larger inputs
print("Running benchmark workload...")
for i in range(10):
    batch_size, seq_len, d_model = 8, 512, 768  # Larger for benchmarking
    
    query = torch.randn(batch_size, seq_len, d_model, device='cuda')
    key = torch.randn(batch_size, seq_len, d_model, device='cuda')
    value = torch.randn(batch_size, seq_len, d_model, device='cuda')
    
    result = efficient_attention(query, key, value)

print("Benchmark workload completed")
'''
}
```

## Testing

Before processing your full dataset, test with the provided test script:

```bash
# Make sure your server is running
python axolotl_dev/sandbox/server.py

# In another terminal, run the test
python scripts/test_run_ds.py
```

## Tips

1. **Server Setup**: Make sure your server is running at the specified URL
2. **GPU Memory**: Ensure your tests don't exceed GPU memory
3. **Workload Design**: Include meaningful compute work in your tests for better benchmarking
4. **Error Handling**: Check `execution_success` field to filter successful runs
5. **Memory Management**: Use `torch.cuda.empty_cache()` in your code if needed

## Troubleshooting

- **No benchmark data**: Ensure your tests have sufficient compute work to measure
- **Server connection errors**: Verify server is running and URL is correct
- **GPU OOM**: Reduce tensor sizes in your tests
- **Import errors**: Ensure all required imports are in your code sections
- **Low benchmark times**: Add more compute work to your tests for meaningful measurements 