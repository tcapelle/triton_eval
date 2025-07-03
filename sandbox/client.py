#!/usr/bin/env python3
"""
Minimal client for the Triton/PyTorch Code Execution Server

This script demonstrates how to interact with the server for executing
Triton kernels and PyTorch code with benchmarking capabilities.
"""

import httpx
import json
import time
from typing import Dict, Any, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

class TritonClient:
    """Simple client for the Triton/PyTorch execution server."""
    
    def __init__(self, server_url: str = "http://127.0.0.1:9347"):
        self.server_url = server_url
        self.client = httpx.Client(
            timeout=httpx.Timeout(connect=30.0, read=300.0, write=30.0, pool=30.0)
        )
    
    def health_check(self) -> bool:
        """Check if the server is running."""
        try:
            response = self.client.get(f"{self.server_url}/")
            response.raise_for_status()
            console.print("✅ Server is running!", style="green")
            return True
        except Exception as e:
            console.print(f"❌ Server not accessible: {e}", style="red")
            return False
    
    def run_triton(
        self, 
        code: str, 
        tests: str, 
        benchmark: bool = False,
        benchmark_runs: int = 10
    ) -> Optional[Dict[str, Any]]:
        """Execute Triton kernel code."""
        try:
            response = self.client.post(
                f"{self.server_url}/run_triton",
                json={
                    "code": code,
                    "tests": tests,
                    "benchmark": benchmark,
                    "benchmark_runs": benchmark_runs
                }
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            console.print(f"❌ Triton execution failed: {e}", style="red")
            return None
    
    def run_pytorch(
        self,
        code: str,
        tests: str,
        benchmark: bool = False,
        benchmark_runs: int = 10,
        torch_compile: bool = False,
        torch_compile_mode: str = "default",
        entrypoint: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Execute PyTorch code."""
        payload = {
            "code": code,
            "tests": tests,
            "benchmark": benchmark,
            "benchmark_runs": benchmark_runs,
            "torch_compile": torch_compile,
            "torch_compile_mode": torch_compile_mode
        }
        if entrypoint:
            payload["entrypoint"] = entrypoint
            
        try:
            response = self.client.post(
                f"{self.server_url}/run_pytorch",
                json=payload
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            console.print(f"❌ PyTorch execution failed: {e}", style="red")
            return None
    
    def reset_workers(self) -> bool:
        """Reset worker processes."""
        try:
            response = self.client.post(f"{self.server_url}/reset_workers")
            response.raise_for_status()
            console.print("✅ Workers reset successfully!", style="green")
            return True
        except Exception as e:
            console.print(f"❌ Failed to reset workers: {e}", style="red")
            return False
    
    def display_results(self, result: Dict[str, Any], title: str = "Execution Results"):
        """Display execution results in a formatted table."""
        if not result:
            return
            
        # Create results table
        table = Table(title=title)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        # Basic execution info
        table.add_row("Status Code", str(result.get("status_code", "N/A")))
        
        if result.get("stdout"):
            table.add_row("Output", result["stdout"][:100] + "..." if len(result["stdout"]) > 100 else result["stdout"])
        
        if result.get("stderr"):
            table.add_row("Error", result["stderr"][:100] + "..." if len(result["stderr"]) > 100 else result["stderr"])
        
        # Resource usage
        if result.get("gpu_mem_used_gb"):
            table.add_row("GPU Memory (GB)", f"{result['gpu_mem_used_gb']:.2f}")
        if result.get("cpu_percent"):
            table.add_row("CPU Usage (%)", f"{result['cpu_percent']:.1f}")
        if result.get("ram_percent"):
            table.add_row("RAM Usage (%)", f"{result['ram_percent']:.1f}")
        
        # Benchmark results
        if result.get("benchmark_mean_time_ms"):
            table.add_row("Mean Time (ms)", f"{result['benchmark_mean_time_ms']:.2f}")
            table.add_row("Std Time (ms)", f"{result['benchmark_std_time_ms']:.2f}")
            table.add_row("Peak Memory (MB)", f"{result['benchmark_memory_peak_mb']:.1f}")
            table.add_row("Successful Runs", str(result['benchmark_successful_runs']))
        
        # PyTorch compile results
        if result.get("torch_compile_benchmark_mean_time_ms"):
            table.add_row("Compiled Time (ms)", f"{result['torch_compile_benchmark_mean_time_ms']:.2f}")
            table.add_row("Speedup", f"{result['torch_compile_speedup']:.2f}x")
        
        console.print(table)
    
    def close(self):
        """Close the HTTP client."""
        self.client.close()


def example_triton_kernel():
    """Example Triton kernel execution."""
    console.print(Panel("[bold blue]Triton Kernel Example[/bold blue]", expand=False))
    
    triton_code = """
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

def triton_add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output
"""

    test_code = """
torch.manual_seed(42)
x = torch.randn(10000, device='cuda')
y = torch.randn(10000, device='cuda')

# Test the Triton kernel
result = triton_add(x, y)
expected = x + y

# Verify correctness
if torch.allclose(result, expected, rtol=1e-5):
    print("✅ Triton kernel works correctly!")
    print(f"Result shape: {result.shape}")
    print(f"Max difference: {torch.max(torch.abs(result - expected)).item():.2e}")
else:
    print("❌ Triton kernel failed!")
"""
    
    return triton_code, test_code


def example_pytorch_code():
    """Example PyTorch code execution."""
    console.print(Panel("[bold blue]PyTorch Example[/bold blue]", expand=False))
    
    pytorch_code = """
import torch
import torch.nn.functional as F

def pytorch_add(x: torch.Tensor, y: torch.Tensor):
    return x + y

def benchmark_function():
    x = torch.randn(10000, device='cuda')
    y = torch.randn(10000, device='cuda')
    return pytorch_add(x, y)
"""

    test_code = """
torch.manual_seed(42)
x = torch.randn(10000, device='cuda')
y = torch.randn(10000, device='cuda')

# Test the function
result = pytorch_add(x, y)
expected = x + y

# Verify correctness
if torch.allclose(result, expected):
    print("✅ PyTorch function works correctly!")
    print(f"Result shape: {result.shape}")
else:
    print("❌ PyTorch function failed!")

# Benchmark
result_bench = benchmark_function()
print(f"Benchmark result shape: {result_bench.shape}")
"""
    
    return pytorch_code, test_code


def main():
    """Main demonstration function."""
    console.print(Panel("[bold green]Triton/PyTorch Server Client Demo[/bold green]", expand=False))
    
    # Initialize client
    client = TritonClient()
    
    # Health check
    if not client.health_check():
        console.print("❌ Server is not running. Please start the server first.", style="red")
        return
    
    try:
        # Triton example
        triton_code, triton_tests = example_triton_kernel()
        console.print("\n🚀 Running Triton kernel...")
        triton_result = client.run_triton(triton_code, triton_tests, benchmark=True, benchmark_runs=5)
        if triton_result:
            client.display_results(triton_result, "Triton Kernel Results")
        
        # PyTorch example
        pytorch_code, pytorch_tests = example_pytorch_code()
        console.print("\n🚀 Running PyTorch code...")
        pytorch_result = client.run_pytorch(
            pytorch_code, 
            pytorch_tests, 
            benchmark=True, 
            benchmark_runs=5,
            torch_compile=False,
            entrypoint="benchmark_function"
        )
        if pytorch_result:
            client.display_results(pytorch_result, "PyTorch Results")
        
        # PyTorch with torch.compile
        console.print("\n🚀 Running PyTorch code with torch.compile...")
        compiled_result = client.run_pytorch(
            pytorch_code,
            pytorch_tests,
            benchmark=True,
            benchmark_runs=5,
            torch_compile=True,
            torch_compile_mode="default",
            entrypoint="benchmark_function"
        )
        if compiled_result:
            client.display_results(compiled_result, "PyTorch + torch.compile Results")
            
    finally:
        client.close()


if __name__ == "__main__":
    main() 