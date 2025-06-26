# Triton/PyTorch Code Execution Server

A FastAPI-based server that provides secure execution of Triton and PyTorch code with benchmarking capabilities and GPU resource management.

## Overview

This server enables remote execution of Triton kernels and PyTorch code in isolated worker processes with GPU pinning, resource monitoring, and performance benchmarking. It's designed for evaluating and testing GPU-accelerated code in a controlled environment.

## Key Features

- **Multi-GPU Support**: Automatically detects available GPUs and distributes workload
- **Worker Pool Management**: Maintains a pool of worker processes with automatic recovery
- **Code Execution**: Supports both Triton kernel and PyTorch code execution
- **Benchmarking**: Built-in performance benchmarking with statistical analysis
- **torch.compile Support**: Optional torch.compile benchmarking for PyTorch code
- **Resource Monitoring**: GPU memory, CPU, and RAM usage tracking
- **Fault Tolerance**: Automatic worker replacement on crashes

## API Endpoints

### `POST /run_triton`
Execute Triton kernel code with optional benchmarking.

**Request Body:**
```json
{
  "code": "# Your Triton kernel code",
  "tests": "# Test code to execute",
  "benchmark": false,
  "benchmark_runs": 10
}
```

### `POST /run_pytorch`
Execute PyTorch code with optional torch.compile benchmarking.

**Request Body:**
```json
{
  "code": "# Your PyTorch code",
  "tests": "# Test code to execute", 
  "benchmark": false,
  "benchmark_runs": 10,
  "torch_compile": false,
  "torch_compile_mode": "default",
  "entrypoint": "function_name"
}
```

### `POST /reset_workers`
Restart all worker processes (useful for clearing GPU memory).

### `GET /`
Health check endpoint.

## Configuration

Environment variables:
- `CONCURRENCY_PER_GPU`: Workers per GPU (default: 1)
- `TASK_TIMEOUT_SECONDS`: Task execution timeout (default: 30)
- `WORKER_JOIN_TIMEOUT`: Worker shutdown timeout (default: 20)

## Running the Server

```bash
python server.py
```

The server will start on `http://0.0.0.0:9347` by default.

## Architecture

- **server.py**: Main FastAPI application with worker pool management
- **worker.py**: Worker process implementation for code execution
- **WorkerPool**: Manages worker lifecycle with automatic recovery
- **Monitoring**: GPU/CPU/RAM metrics collection via pynvml and psutil

## Response Format

All execution endpoints return:
```json
{
  "status_code": 0,
  "stdout": "execution output",
  "stderr": "error output",
  "gpu_mem_used_gb": 1.5,
  "cpu_percent": 25.0,
  "ram_percent": 30.0,
  "benchmark_mean_time_ms": 10.5,
  "benchmark_std_time_ms": 0.3,
  "benchmark_memory_peak_mb": 512.0,
  "benchmark_successful_runs": 10,
  "torch_compile_benchmark_mean_time_ms": 8.2,
  "torch_compile_benchmark_std_time_ms": 0.2,
  "torch_compile_speedup": 1.28
}
```

## Safety Features

- Process isolation prevents crashes from affecting other tasks
- Automatic worker replacement on fatal errors (CUDA OOM, compilation errors)
- Temporary file cleanup after execution
- Resource monitoring and limits
- Graceful shutdown handling