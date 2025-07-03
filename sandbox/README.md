# Triton/PyTorch Code Execution Server

A high-performance FastAPI-based server for secure execution of Triton kernels and PyTorch code with advanced benchmarking capabilities, GPU resource management, and structured result handling.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.68+-green.svg)](https://fastapi.tiangolo.com/)
[![CUDA](https://img.shields.io/badge/CUDA-11.0+-orange.svg)](https://developer.nvidia.com/cuda-toolkit)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Client Usage](#client-usage)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Architecture](#architecture)
- [Performance Considerations](#performance-considerations)
- [Troubleshooting](#troubleshooting)
- [Examples](#examples)

## Overview

This server enables remote execution of Triton kernels and PyTorch code in isolated worker processes with GPU pinning, resource monitoring, and performance benchmarking. It's designed for evaluating and testing GPU-accelerated code in production environments, training pipelines, and research workflows.

## Features

### 🚀 **Execution Capabilities**
- **Triton Kernel Execution**: Full support for Triton kernels with GPU compilation
- **PyTorch Code Execution**: Standard PyTorch operations with CUDA support
- **torch.compile Integration**: Automatic torch.compile benchmarking and comparison
- **Code Isolation**: Each execution runs in isolated worker processes

### 🔧 **Infrastructure**
- **Multi-GPU Support**: Automatically detects and utilizes all available GPUs
- **Worker Pool Management**: Self-healing worker pool with automatic crash recovery
- **Resource Monitoring**: Real-time GPU memory, CPU, and RAM usage tracking
- **Fault Tolerance**: Automatic worker replacement on CUDA OOM or compilation errors

### 📊 **Performance & Benchmarking**
- **Statistical Benchmarking**: Multiple runs with mean, std deviation, and confidence intervals
- **Memory Profiling**: Peak memory usage tracking during execution
- **Performance Comparison**: Side-by-side torch.compile vs regular PyTorch benchmarks
- **Resource Metrics**: Detailed system resource utilization

### 🎯 **New: Structured Results & Monitoring**
- **TaskResult Class**: Clean, type-safe result handling with helper methods
- **Rich Summaries**: Beautiful logging with emojis and key metrics
- **Configurable Verbosity**: Control logging levels via `WORKER_VERBOSE`
- **Status Monitoring**: Built-in `/status` endpoint and periodic status tables

## Installation

### Prerequisites

```bash
# CUDA Toolkit (11.0+ recommended)
# Python 3.8+
# PyTorch with CUDA support
# Triton (for kernel execution)
```

### Dependencies

```bash
pip install fastapi uvicorn httpx rich
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install triton
pip install pynvml psutil  # For resource monitoring
```

### Optional Dependencies

```bash
pip install weave-python  # For experiment tracking
pip install wandb        # Alternative experiment tracking
```

## Quick Start

### 1. Start the Server

```bash
# Navigate to the sandbox directory
cd sandbox

# Start the server (default: http://0.0.0.0:9347)
python server.py
```

### 2. Test with the Client

```bash
# Run the demo client
python client.py
```

### 3. Health Check

```bash
curl http://localhost:9347/
curl http://localhost:9347/status  # New: Get detailed status
```

## Client Usage

### Basic Client Example

```python
from client import TritonClient

# Initialize client
client = TritonClient("http://localhost:9347")

# Health check
if client.health_check():
    
    # Execute Triton kernel
    triton_result = client.run_triton(
        code=triton_kernel_code,
        tests=test_code,
        benchmark=True,
        benchmark_runs=10
    )
    
    # Execute PyTorch code
    pytorch_result = client.run_pytorch(
        code=pytorch_code,
        tests=test_code,
        benchmark=True,
        torch_compile=True,
        entrypoint="my_function"
    )
    
    # Display results
    client.display_results(triton_result, "Triton Results")
    client.display_results(pytorch_result, "PyTorch Results")

client.close()
```

### Advanced Usage with TaskResult

```python
# Custom configuration
client = TritonClient("http://remote-server:9347")

# High-precision benchmarking
result = client.run_triton(
    code=kernel_code,
    tests=test_code,
    benchmark=True,
    benchmark_runs=100  # More runs for better statistics
)

# Performance comparison
regular_result = client.run_pytorch(
    code=code, tests=tests, 
    benchmark=True, torch_compile=False
)

compiled_result = client.run_pytorch(
    code=code, tests=tests, 
    benchmark=True, torch_compile=True,
    torch_compile_mode="max-autotune"  # Aggressive optimization
)

# Calculate speedup
speedup = regular_result["benchmark_mean_time_ms"] / compiled_result["torch_compile_benchmark_mean_time_ms"]
print(f"torch.compile speedup: {speedup:.2f}x")
```

### Using TaskResult in Client Code

```python
from client import TritonClient
from sandbox.task_types import TaskResult

client = TritonClient("http://localhost:9347")

# Execute and get structured result
response = client.run_triton(code, tests, benchmark=True)

# Convert API response (dict) to TaskResult object using Pydantic
result = TaskResult.model_validate(response)

print(f"Task successful: {result.is_successful}")
print(f"Has benchmarks: {result.has_benchmarks}")
print(f"Summary: {result.get_summary()}")

if result.has_benchmarks:
    print(f"Average time: {result.benchmark_mean_time_ms:.2f}ms")
    print(f"Successful runs: {result.benchmark_successful_runs}")
```

## API Reference

### `POST /run_triton`

Execute Triton kernel code with optional benchmarking.

**Request:**
```json
{
  "code": "string",           // Triton kernel code
  "tests": "string",          // Test/execution code  
  "benchmark": false,         // Enable benchmarking
  "benchmark_runs": 10        // Number of benchmark iterations
}
```

**Response:**
```json
{
  "status_code": 0,                           // 0 = success, -1 = error
  "stdout": "execution output",               // Program output
  "stderr": "error output",                   // Error messages
  "gpu_mem_used_gb": 1.5,                    // GPU memory usage
  "cpu_percent": 25.0,                       // CPU utilization
  "ram_percent": 30.0,                       // RAM utilization
  "benchmark_mean_time_ms": 10.5,            // Average execution time
  "benchmark_std_time_ms": 0.3,              // Standard deviation
  "benchmark_memory_peak_mb": 512.0,         // Peak memory usage
  "benchmark_successful_runs": 10            // Successful benchmark runs
}
```

### `POST /run_pytorch`

Execute PyTorch code with optional torch.compile benchmarking.

**Request:**
```json
{
  "code": "string",                    // PyTorch code
  "tests": "string",                   // Test/execution code
  "benchmark": false,                  // Enable benchmarking
  "benchmark_runs": 10,                // Number of benchmark iterations
  "torch_compile": false,              // Enable torch.compile
  "torch_compile_mode": "default",     // Compilation mode
  "entrypoint": "function_name"        // Function to benchmark
}
```

**Additional Response Fields:**
```json
{
  "torch_compile_benchmark_mean_time_ms": 8.2,  // Compiled execution time
  "torch_compile_benchmark_std_time_ms": 0.2,   // Compiled std deviation
  "torch_compile_speedup": 1.28                 // Speedup ratio
}
```

### `GET /status` (New)

Get detailed server status and metrics.

**Response:**
```json
{
  "total_workers": 8,
  "active_workers": 8,
  "dead_workers": 0,
  "worker_pool_running": true,
  "pending_requests": 2,
  "queue_size": 1,
  "stats": {
    "total_requests": 150,
    "successful_requests": 147,
    "failed_requests": 2,
    "timeout_requests": 1,
    "start_time": 1635724800.0
  },
  "uptime_seconds": 3600.5,
  "gpus_configured": 8,
  "concurrency_per_gpu": 1
}
```

### `POST /reset_workers`

Restart all worker processes (useful for clearing GPU memory).

**Response:**
```json
{
  "message": "Workers reset successfully."
}
```

### `GET /`

Health check endpoint.

**Response:**
```json
{
  "message": "Triton Worker Pool Server is ready!"
}
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CONCURRENCY_PER_GPU` | `1` | Number of worker processes per GPU |
| `TASK_TIMEOUT_SECONDS` | `30` | Maximum execution time per task |
| `WORKER_JOIN_TIMEOUT` | `20` | Timeout for graceful worker shutdown |
| `TRITON_SERVER_URL` | `http://127.0.0.1:9347` | Server URL for client connections |
| `WORKER_VERBOSE` | `false` | Enable verbose worker logging |
| `STATUS_LOG_INTERVAL` | `30` | Seconds between status table logs |

### Example Configuration

```bash
# High-throughput setup
export CONCURRENCY_PER_GPU=2
export TASK_TIMEOUT_SECONDS=60

# Enable verbose worker logging
export WORKER_VERBOSE=true

# Start server
python server.py
```

### Verbosity Control

```bash
# Quiet mode (default) - clean, minimal output
python server.py

# Verbose mode - detailed worker logging
export WORKER_VERBOSE=true
python server.py
```

## Architecture

### Code Structure

**Core Components:**
- `server.py` - FastAPI server with worker pool management
- `worker.py` - Worker process execution logic
- `task_types.py` - Shared data structures and types (TaskResult class)
- `client.py` - Client interface for server communication

**TaskResult Class:**
The `TaskResult` class is a Pydantic BaseModel that provides structured result handling with:
- **Type Safety**: Proper typing for all fields with automatic validation
- **Field Validation**: Built-in constraints (e.g., `ge=0` for non-negative values)
- **JSON Serialization**: Native `.model_dump()` and `.model_validate()` methods
- **Helper Methods**: `is_successful`, `has_benchmarks`, `has_torch_compile_results`
- **Rich Summaries**: `get_summary()` for beautiful logging
- **FastAPI Integration**: Seamless integration with OpenAPI schema generation

```python
from sandbox.task_types import TaskResult

# Example usage with automatic validation
result = TaskResult(
    task_id="abc123",
    status_code=0,
    stdout="Hello World",
    benchmark_mean_time_ms=42.5,
    torch_compile_speedup=1.5,
    gpu_mem_used_gb=2.3  # Automatically validated to be >= 0
)

print(result.is_successful)  # True
print(result.get_summary())  # Task abc123...: ✅ SUCCESS | ⏱️ 42.50ms | 🚀 1.50x speedup | 🖥️ 2.3GB GPU

# Pydantic validation in action
json_data = result.model_dump_json()  # Native JSON serialization
restored = TaskResult.model_validate_json(json_data)  # Native validation
```

### System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   FastAPI       │    │   WorkerPool     │    │   GPU Workers   │
│   Server        │◄──►│   Manager        │◄──►│   (Per GPU)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         ▲                       ▲                       ▲
         │                       │                       │
    HTTP Clients           Task Queue              Code Execution
         │                       │                       │
    ┌─────────────┐    ┌─────────────────┐    ┌─────────────────┐
    │ TaskResult  │    │ Status Monitor  │    │ Resource Monitor│
    │ Objects     │    │ & Logging       │    │ & Benchmarking  │
    └─────────────┘    └─────────────────┘    └─────────────────┘
```

### Process Flow

1. **Request Receipt**: FastAPI server receives execution request
2. **Task Queuing**: Request added to multiprocessing queue
3. **Worker Assignment**: Available worker picks up task
4. **GPU Execution**: Code executed on assigned GPU with monitoring
5. **Result Collection**: Metrics and output collected in TaskResult object
6. **Result Processing**: Structured result with rich summaries
7. **Cleanup**: Temporary files cleaned, resources released

## Performance Considerations

### Optimization Tips

1. **Batch Processing**: Group multiple small tasks for better throughput
2. **Worker Configuration**: Tune `CONCURRENCY_PER_GPU` based on GPU memory
3. **Benchmark Runs**: Use 10-100 runs for statistical significance
4. **torch.compile**: Enable for repetitive operations (warmup overhead applies)
5. **Memory Management**: Use `/reset_workers` to clear GPU memory between large tasks

### Scaling Guidelines

| GPU Memory | Recommended Workers | Max Concurrent Tasks |
|------------|--------------------|--------------------|
| 8GB        | 1                  | 2-4                |
| 16GB       | 2                  | 4-8                |
| 24GB+      | 3-4                | 8-16               |

### Monitoring Benefits

With the new structured monitoring:
- ✅ **Clean Status Tables**: Regular system health overview
- ✅ **Rich Task Summaries**: Key metrics with emojis (⏱️ 🚀 🖥️)
- ✅ **Real-time Metrics**: `/status` endpoint for programmatic monitoring
- ✅ **Configurable Verbosity**: Control noise with `WORKER_VERBOSE`

## Troubleshooting

### Common Issues

**Server won't start:**
```bash
# Check GPU availability
nvidia-smi

# Check port availability  
lsof -i :9347

# Check dependencies
pip list | grep -E "(torch|triton|fastapi)"
```

**Connection timeouts:**
```python
# Increase client timeout
client = TritonClient()
client.client.timeout = httpx.Timeout(connect=60.0, read=600.0)
```

**CUDA Out of Memory:**
```python
# Reset workers to clear GPU memory
client.reset_workers()

# Or reduce batch sizes in your code
```

**Worker crashes:**
- Workers automatically restart on crashes
- Check server logs for detailed error information
- Reduce `CONCURRENCY_PER_GPU` if instability occurs

**Import errors (circular imports):**
- The `task_types.py` file avoids conflicts with Python's built-in `types` module
- Use proper imports: `from sandbox.task_types import TaskResult`

### Debug Mode

```bash
# Enable detailed logging
export PYTHONPATH=/app
export LOG_LEVEL=DEBUG
export WORKER_VERBOSE=true
python server.py
```

### Status Monitoring

```bash
# Check server status
curl http://localhost:9347/status

# Monitor server logs for status tables (every 30s by default)
# Set STATUS_LOG_INTERVAL=10 for more frequent updates
```

## Examples

### Triton Kernel Example

```python
triton_code = """
import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    # ... kernel implementation
"""

test_code = """
A = torch.randn((512, 512), device='cuda', dtype=torch.float16)
B = torch.randn((512, 512), device='cuda', dtype=torch.float16) 
C = triton_matmul(A, B)
torch_result = torch.matmul(A, B)
print(f"Max difference: {torch.max(torch.abs(C - torch_result))}")
"""
```

### PyTorch with torch.compile

```python
pytorch_code = """
import torch
import torch.nn.functional as F

@torch.compile(mode="max-autotune")
def optimized_attention(q, k, v):
    scores = torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
    attn_weights = F.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, v)

def benchmark_attention():
    q = torch.randn(32, 8, 128, 64, device='cuda')
    k = torch.randn(32, 8, 128, 64, device='cuda') 
    v = torch.randn(32, 8, 128, 64, device='cuda')
    return optimized_attention(q, k, v)
"""
```

---

**📝 For more examples and advanced usage, see `client.py` and `test_benchmark.py`.**

**🐛 Found a bug? Please open an issue with reproduction steps.**

**🚀 Performance improvements and feature requests are welcome!**