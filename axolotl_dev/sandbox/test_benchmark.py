#!/usr/bin/env python3
"""
Simple test script to verify benchmark functionality in the server.
"""

import asyncio
import httpx
import json
from rich.console import Console
from rich.table import Table

console = Console()
SERVER_URL = "http://127.0.0.1:9347"

async def test_basic_execution():
    """Test basic code execution without benchmarking"""
    console.print("[bold blue]Testing basic execution (no benchmark)[/bold blue]")
    
    code = """
import torch

def simple_add(a, b):
    return a + b
"""
    
    tests = """
import torch
torch.manual_seed(42)

# Test the function
result = simple_add(5, 3)
print(f"Result: {result}")

# Basic tensor test
x = torch.tensor([1, 2, 3], device='cuda')
y = torch.tensor([4, 5, 6], device='cuda')
z = x + y
print(f"Tensor addition result: {z}")
"""
    
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(f"{SERVER_URL}/run_triton", 
                                   json={
                                       "code": code,
                                       "tests": tests,
                                       "benchmark": False
                                   },
                                   timeout=30.0)
            resp.raise_for_status()
            result = resp.json()
            
            console.print(f"✅ Status: {result['status_code']}")
            if result['stdout']:
                console.print(f"📄 Output:\n{result['stdout']}")
            if result['stderr']:
                console.print(f"⚠️  Error:\n{result['stderr']}")
                
            return result['status_code'] == 0
            
        except Exception as e:
            console.print(f"❌ Request failed: {e}")
            return False

async def test_triton_benchmark():
    """Test benchmarking with a simple Triton kernel"""
    console.print("[bold blue]Testing Triton kernel with benchmarking[/bold blue]")
    
    code = """
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

def triton_add(x, y):
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output
"""
    
    tests = """
import torch
torch.manual_seed(42)

def test_triton_add():
    results = {}
    
    # Test case 1: Basic functionality test
    size = 1000
    x = torch.randn(size, device='cuda', dtype=torch.float32)
    y = torch.randn(size, device='cuda', dtype=torch.float32)
    
    result_triton = triton_add(x, y)
    result_torch = x + y
    
    # Verify correctness
    max_diff = torch.max(torch.abs(result_triton - result_torch)).item()
    results["max_difference"] = max_diff
    results["results_match"] = max_diff < 1e-5
    results["result_shape"] = result_triton.shape
    
    print(f"Max difference between Triton and PyTorch: {max_diff}")
    print(f"Results match: {max_diff < 1e-5}")
    print(f"Result shape: {result_triton.shape}")
    
    return results

def benchmark_function():
    '''Function for benchmarking'''
    size = 1024 * 256  # 256K elements
    x = torch.randn(size, device='cuda', dtype=torch.float32)
    y = torch.randn(size, device='cuda', dtype=torch.float32)
    return triton_add(x, y)

# Run tests
test_results = test_triton_add()
print(f"Test results: {test_results}")
"""
    
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(f"{SERVER_URL}/run_triton",
                                   json={
                                       "code": code,
                                       "tests": tests, 
                                       "benchmark": True,
                                       "benchmark_runs": 3
                                   },
                                   timeout=120.0)
            resp.raise_for_status()
            result = resp.json()
            
            console.print(f"✅ Status: {result['status_code']}")
            if result['stdout']:
                console.print(f"📄 Output:\n{result['stdout']}")
            if result['stderr']:
                console.print(f"⚠️  Error:\n{result['stderr']}")
            
            # Show benchmark results
            if result.get('benchmark_mean_time_ms'):
                table = Table(title="Triton Kernel Benchmark Results")
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="green")
                
                table.add_row("Mean Time (ms)", f"{result['benchmark_mean_time_ms']:.2f}")
                table.add_row("Std Time (ms)", f"{result['benchmark_std_time_ms']:.2f}")
                table.add_row("Peak Memory (MB)", f"{result['benchmark_memory_peak_mb']:.1f}")
                table.add_row("Successful Runs", str(result['benchmark_successful_runs']))
                
                console.print(table)
            
            return result['status_code'] == 0
            
        except Exception as e:
            console.print(f"❌ Request failed: {e}")
            return False

async def test_pytorch_benchmark():
    """Test PyTorch benchmarking without torch.compile"""
    console.print("[bold blue]Testing PyTorch benchmarking (regular)[/bold blue]")
    
    code = """
import torch
from typing import Optional

def add(input: torch.Tensor, other: torch.Tensor, alpha: float=1, out: Optional[torch.Tensor]=None):
    \"\"\"
    Adds the tensor or number 'other', scaled by 'alpha', to the 'input' tensor.
    
    Args:
        input (Tensor): The input tensor.
        other (Tensor or Number): The tensor or number to add to input.
        alpha (Number, optional): The multiplier for 'other'. Default is 1.
        out (Tensor, optional): The output tensor. If provided, the result will be stored in this tensor.
        
    Returns:
        Tensor: The result of adding 'other' scaled by 'alpha' to 'input'.
    \"\"\"
    return torch.add(input, other, alpha=alpha, out=out)
"""
    
    tests = """
import torch
torch.manual_seed(42)

def test_add():
    results = {}

    # Test case 1: Adding two tensors with default alpha
    input1 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    other1 = torch.tensor([4.0, 5.0, 6.0], device='cuda')
    results["test_case_1"] = add(input1, other1)

    # Test case 2: Adding a tensor and a scalar with default alpha
    input2 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    other2 = 2.0
    results["test_case_2"] = add(input2, other2)

    # Test case 3: Adding two tensors with a specified alpha
    input3 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    other3 = torch.tensor([4.0, 5.0, 6.0], device='cuda')
    results["test_case_3"] = add(input3, other3, alpha=0.5)

    # Test case 4: Larger inputs
    input4 = torch.randn(30, 20, device='cuda')
    other4 = torch.randn(30, 20, device='cuda')
    alpha = 0.5
    results["test_case_4"] = add(input4, other4, alpha=alpha)

    return results

def benchmark_function():
    '''Function that will be called for benchmarking'''
    input_tensor = torch.randn(256, 256, device='cuda')
    other_tensor = torch.randn(256, 256, device='cuda')
    return add(input_tensor, other_tensor, alpha=0.5)

# Run tests
test_results = test_add()
print(f"Test results keys: {list(test_results.keys())}")
for key, value in test_results.items():
    if hasattr(value, 'shape'):
        print(f"{key}: shape={value.shape}, dtype={value.dtype}")
    else:
        print(f"{key}: {value}")
"""
    
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(f"{SERVER_URL}/run_pytorch",
                                   json={
                                       "code": code, 
                                       "tests": tests,
                                       "benchmark": True,
                                       "benchmark_runs": 5,
                                       "torch_compile": False
                                   },
                                   timeout=60.0)
            resp.raise_for_status()
            result = resp.json()
            
            console.print(f"✅ Status: {result['status_code']}")
            if result['stdout']:
                console.print(f"📄 Output:\n{result['stdout']}")
            if result['stderr']:
                console.print(f"⚠️  Error:\n{result['stderr']}")
            
            # Show benchmark results
            if result.get('benchmark_mean_time_ms'):
                table = Table(title="PyTorch Benchmark Results")
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="green")
                
                table.add_row("Mean Time (ms)", f"{result['benchmark_mean_time_ms']:.2f}")
                table.add_row("Std Time (ms)", f"{result['benchmark_std_time_ms']:.2f}")
                table.add_row("Peak Memory (MB)", f"{result['benchmark_memory_peak_mb']:.1f}")
                table.add_row("Successful Runs", str(result['benchmark_successful_runs']))
                
                console.print(table)
            
            return result['status_code'] == 0
            
        except Exception as e:
            console.print(f"❌ Request failed: {e}")
            return False

async def test_pytorch_torch_compile():
    """Test PyTorch benchmarking with torch.compile"""
    console.print("[bold blue]Testing PyTorch with torch.compile benchmarking[/bold blue]")
    
    code = """
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(512, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 64)
    
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.linear3(x)

def create_model_and_data():
    model = SimpleModel().cuda()
    input_data = torch.randn(32, 512, device='cuda')
    return model, input_data
"""
    
    tests = """
import torch
torch.manual_seed(42)

# Create model and data
model, input_data = create_model_and_data()

def test_model():
    results = {}
    
    # Test the model functionality
    with torch.no_grad():
        output = model(input_data)
        results["output_shape"] = output.shape
        results["output_range"] = [output.min().item(), output.max().item()]
        results["model_parameters"] = sum(p.numel() for p in model.parameters())
    
    print(f"Model output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    
    return results

def benchmark_function():
    '''Function that will be called for benchmarking'''
    with torch.no_grad():
        return model(input_data)

# Run tests
test_results = test_model()
print(f"Test results: {test_results}")
"""
    
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(f"{SERVER_URL}/run_pytorch",
                                   json={
                                       "code": code, 
                                       "tests": tests,
                                       "benchmark": True,
                                       "benchmark_runs": 5,
                                       "torch_compile": True,
                                       "torch_compile_mode": "default"
                                   },
                                   timeout=120.0)  # Longer timeout for compilation
            resp.raise_for_status()
            result = resp.json()
            
            console.print(f"✅ Status: {result['status_code']}")
            if result['stdout']:
                console.print(f"📄 Output:\n{result['stdout']}")
            if result['stderr']:
                console.print(f"⚠️  Error:\n{result['stderr']}")
            
            # Show benchmark results
            if result.get('benchmark_mean_time_ms'):
                table = Table(title="PyTorch + torch.compile Benchmark Results")
                table.add_column("Version", style="cyan")
                table.add_column("Mean Time (ms)", style="green")
                table.add_column("Std Time (ms)", style="yellow")
                
                table.add_row("Regular PyTorch", f"{result['benchmark_mean_time_ms']:.2f}", f"{result['benchmark_std_time_ms']:.2f}")
                
                if result.get('torch_compile_benchmark_mean_time_ms'):
                    table.add_row("torch.compile", f"{result['torch_compile_benchmark_mean_time_ms']:.2f}", f"{result['torch_compile_benchmark_std_time_ms']:.2f}")
                    
                    if result.get('torch_compile_speedup'):
                        table.add_row("Speedup", f"{result['torch_compile_speedup']:.2f}x", "")
                
                console.print(table)
            
            return result['status_code'] == 0
            
        except Exception as e:
            console.print(f"❌ Request failed: {e}")
            return False

async def test_triton_pytorch_comparison():
    """Test the same functionality with both Triton and PyTorch endpoints"""
    console.print("[bold blue]Testing Triton vs PyTorch comparison[/bold blue]")
    
    # Same add functionality for both
    triton_code = """
import torch
import triton
import triton.language as tl

@triton.jit
def triton_add_kernel(x_ptr, y_ptr, output_ptr, n_elements, alpha: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + alpha * y
    tl.store(output_ptr + offsets, output, mask=mask)

def triton_add(input: torch.Tensor, other: torch.Tensor, alpha: float = 1.0):
    output = torch.empty_like(input)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    triton_add_kernel[grid](input, other, output, n_elements, alpha, BLOCK_SIZE=1024)
    return output
"""
    
    pytorch_code = """
import torch
from typing import Optional

def add(input: torch.Tensor, other: torch.Tensor, alpha: float=1, out: Optional[torch.Tensor]=None):
    return torch.add(input, other, alpha=alpha, out=out)
"""
    
    # Same tests for both
    shared_tests = """
import torch
torch.manual_seed(42)

def test_add_functionality():
    results = {}
    
    # Test case: Adding two tensors with alpha
    input1 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    other1 = torch.tensor([4.0, 5.0, 6.0], device='cuda')
    alpha = 0.5
    
    # Use the appropriate function based on the implementation
    if 'triton_add' in globals():
        result = triton_add(input1, other1, alpha=alpha)
        impl_name = "triton"
    else:
        result = add(input1, other1, alpha=alpha)
        impl_name = "pytorch"
    
    results["implementation"] = impl_name
    results["result"] = result
    results["expected"] = input1 + alpha * other1
    results["difference"] = torch.max(torch.abs(result - (input1 + alpha * other1))).item()
    
    print(f"Implementation: {impl_name}")
    print(f"Result: {result}")
    print(f"Expected: {input1 + alpha * other1}")
    print(f"Max difference: {results['difference']}")
    
    return results

def benchmark_function():
    '''Function for benchmarking'''
    input_tensor = torch.randn(1024, 512, device='cuda')
    other_tensor = torch.randn(1024, 512, device='cuda')
    alpha = 0.5
    
    if 'triton_add' in globals():
        return triton_add(input_tensor, other_tensor, alpha=alpha)
    else:
        return add(input_tensor, other_tensor, alpha=alpha)

# Run tests
test_results = test_add_functionality()
print(f"Test completed for {test_results['implementation']} implementation")
"""
    
    # Test both implementations
    results = {}
    
    async with httpx.AsyncClient() as client:
        # Test Triton implementation
        try:
            resp = await client.post(f"{SERVER_URL}/run_triton",
                                   json={
                                       "code": triton_code,
                                       "tests": shared_tests,
                                       "benchmark": True,
                                       "benchmark_runs": 3
                                   },
                                   timeout=120.0)
            resp.raise_for_status()
            triton_result = resp.json()
            results["triton"] = triton_result
            console.print("✅ Triton implementation tested")
        except Exception as e:
            console.print(f"❌ Triton test failed: {e}")
            results["triton"] = None
        
        # Test PyTorch implementation
        try:
            resp = await client.post(f"{SERVER_URL}/run_pytorch",
                                   json={
                                       "code": pytorch_code,
                                       "tests": shared_tests,
                                       "benchmark": True,
                                       "benchmark_runs": 3,
                                       "torch_compile": False
                                   },
                                   timeout=120.0)
            resp.raise_for_status()
            pytorch_result = resp.json()
            results["pytorch"] = pytorch_result
            console.print("✅ PyTorch implementation tested")
        except Exception as e:
            console.print(f"❌ PyTorch test failed: {e}")
            results["pytorch"] = None
    
    # Compare results
    if results["triton"] and results["pytorch"]:
        table = Table(title="Triton vs PyTorch Comparison")
        table.add_column("Implementation", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Mean Time (ms)", style="yellow")
        table.add_column("Speedup", style="magenta")
        
        triton_time = results["triton"].get('benchmark_mean_time_ms', 0)
        pytorch_time = results["pytorch"].get('benchmark_mean_time_ms', 0)
        
        table.add_row("Triton", "✅ PASSED" if results["triton"]["status_code"] == 0 else "❌ FAILED", 
                     f"{triton_time:.2f}" if triton_time else "N/A", "1.00x")
        
        speedup = pytorch_time / triton_time if triton_time > 0 and pytorch_time > 0 else 0
        table.add_row("PyTorch", "✅ PASSED" if results["pytorch"]["status_code"] == 0 else "❌ FAILED",
                     f"{pytorch_time:.2f}" if pytorch_time else "N/A", f"{speedup:.2f}x" if speedup > 0 else "N/A")
        
        console.print(table)
        
        return all(r and r["status_code"] == 0 for r in results.values() if r)
    
    return False

async def test_server_health():
    """Test if the server is running"""
    console.print("[bold blue]Testing server health[/bold blue]")
    
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{SERVER_URL}/", timeout=5.0)
            resp.raise_for_status()
            result = resp.json()
            console.print(f"✅ Server is running: {result}")
            return True
        except Exception as e:
            console.print(f"❌ Server not accessible: {e}")
            return False

async def main():
    """Run all tests"""
    console.rule("[bold green]Benchmark Functionality Test Suite[/bold green]")
    
    # Test server health first
    server_ok = await test_server_health()
    if not server_ok:
        console.print("❌ Server is not running. Please start the server first with:")
        console.print("   python axolotl_dev/sandbox/server.py")
        return
    
    console.print()
    
    # Run tests
    tests = [
        ("Basic Execution", test_basic_execution),
        ("Triton Benchmark", test_triton_benchmark),
        ("PyTorch Benchmark", test_pytorch_benchmark), 
        ("PyTorch + torch.compile", test_pytorch_torch_compile),
        ("Triton vs PyTorch Comparison", test_triton_pytorch_comparison),
    ]
    
    results = []
    for test_name, test_func in tests:
        console.rule(f"[bold yellow]{test_name}[/bold yellow]")
        try:
            success = await test_func()
            results.append((test_name, success))
            console.print(f"{'✅' if success else '❌'} {test_name}: {'PASSED' if success else 'FAILED'}")
        except Exception as e:
            console.print(f"❌ {test_name}: FAILED with exception: {e}")
            results.append((test_name, False))
        console.print()
    
    # Summary
    console.rule("[bold green]Test Summary[/bold green]")
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    summary_table = Table(title=f"Results: {passed}/{total} tests passed")
    summary_table.add_column("Test", style="cyan")
    summary_table.add_column("Result", style="bold")
    
    for test_name, success in results:
        summary_table.add_row(
            test_name,
            f"{'✅ PASSED' if success else '❌ FAILED'}"
        )
    
    console.print(summary_table)
    
    if passed == total:
        console.print("[bold green]🎉 All tests passed! Benchmark functionality is working correctly.[/bold green]")
    else:
        console.print(f"[bold red]⚠️  {total - passed} tests failed. Please check the errors above.[/bold red]")

if __name__ == "__main__":
    asyncio.run(main()) 