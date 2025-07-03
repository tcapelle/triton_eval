# Triton/PyTorch Code Execution Server

A high-performance FastAPI-based server for secure execution of Triton kernels and PyTorch code with advanced benchmarking capabilities and GPU resource management.

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

### Advanced Usage

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

### Example Configuration

```bash
# High-throughput setup
export CONCURRENCY_PER_GPU=2
export TASK_TIMEOUT_SECONDS=60

# Start server
python server.py
```

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   FastAPI       │    │   WorkerPool     │    │   GPU Workers   │
│   Server        │◄──►│   Manager        │◄──►│   (Per GPU)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         ▲                       ▲                       ▲
         │                       │                       │
    HTTP Clients           Task Queue              Code Execution
```

### Components

- **server.py**: Main FastAPI application with request handling
- **worker.py**: Worker process implementation for isolated code execution
- **WorkerPool**: Manages worker lifecycle with automatic recovery
- **client.py**: Reference client implementation with examples

### Process Flow

1. **Request Receipt**: FastAPI server receives execution request
2. **Task Queuing**: Request added to multiprocessing queue
3. **Worker Assignment**: Available worker picks up task
4. **GPU Execution**: Code executed on assigned GPU with monitoring
5. **Result Collection**: Metrics and output collected and returned
6. **Cleanup**: Temporary files cleaned, resources released

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

### Debug Mode

```bash
# Enable detailed logging
export PYTHONPATH=/app
export LOG_LEVEL=DEBUG
python server.py
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