import time
import statistics
import torch
import asyncio
import httpx
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import numpy as np

@dataclass
class BenchmarkResult:
    """Results from a single benchmark run"""
    mean_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    median_time_ms: float
    gpu_memory_peak_mb: float
    successful_runs: int
    total_runs: int
    speedup_vs_pytorch: Optional[float] = None
    memory_efficiency_vs_pytorch: Optional[float] = None

@dataclass
class ComparisonResult:
    """Results comparing PyTorch vs Triton implementations"""
    pytorch_result: BenchmarkResult
    triton_result: BenchmarkResult
    speedup: float  # triton speedup over pytorch
    memory_efficiency: float  # triton memory efficiency vs pytorch
    performance_score: float  # overall performance score (0-1)
    
class PerformanceBenchmarker:
    """Handles benchmarking of PyTorch and Triton code implementations"""
    
    def __init__(self, server_url: str = "http://127.0.0.1:9347", warmup_runs: int = 5, benchmark_runs: int = 20):
        self.server_url = server_url
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs
        self.run_code_endpoint = f"{server_url}/run_code"
    
    async def benchmark_pytorch_code(self, pytorch_code: str, test_setup: str, 
                                   input_tensors: Dict[str, Any]) -> BenchmarkResult:
        """Benchmark PyTorch reference implementation"""
        benchmark_code = f"""
import torch
import time
import gc
import torch.cuda

{pytorch_code}

{test_setup}

# Warmup runs
for _ in range({self.warmup_runs}):
    try:
        result = benchmark_function()
        torch.cuda.synchronize()
    except:
        pass

# Benchmark runs
times = []
gpu_memory_peak = 0

for run_idx in range({self.benchmark_runs}):
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    
    start_time = time.perf_counter()
    try:
        result = benchmark_function()
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        run_time = (end_time - start_time) * 1000  # Convert to ms
        times.append(run_time)
        
        peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
        gpu_memory_peak = max(gpu_memory_peak, peak_memory)
        
    except Exception as e:
        print(f"PyTorch benchmark run {{run_idx}} failed: {{e}}")
        continue

successful_runs = len(times)
if successful_runs > 0:
    mean_time = sum(times) / len(times)
    std_time = (sum((t - mean_time) ** 2 for t in times) / len(times)) ** 0.5 if len(times) > 1 else 0
    min_time = min(times)
    max_time = max(times)
    median_time = sorted(times)[len(times) // 2]
    
    print(f"BENCHMARK_RESULT:{{mean_time}},{{std_time}},{{min_time}},{{max_time}},{{median_time}},{{gpu_memory_peak}},{{successful_runs}}")
else:
    print("BENCHMARK_FAILED: No successful runs")
"""
        
        async with httpx.AsyncClient() as client:
            try:
                resp = await client.post(self.run_code_endpoint,
                                       json={"code": benchmark_code, "tests": ""},
                                       timeout=300.0)  # 5 minute timeout for benchmarks
                resp.raise_for_status()
                data = resp.json()
                
                if data["status_code"] == 0:
                    # Parse benchmark results from stdout
                    return self._parse_benchmark_result(data["stdout"])
                else:
                    return BenchmarkResult(
                        mean_time_ms=float('inf'), std_time_ms=0, min_time_ms=float('inf'),
                        max_time_ms=float('inf'), median_time_ms=float('inf'),
                        gpu_memory_peak_mb=float('inf'), successful_runs=0, total_runs=self.benchmark_runs
                    )
            except Exception as e:
                print(f"PyTorch benchmark failed: {e}")
                return BenchmarkResult(
                    mean_time_ms=float('inf'), std_time_ms=0, min_time_ms=float('inf'),
                    max_time_ms=float('inf'), median_time_ms=float('inf'),
                    gpu_memory_peak_mb=float('inf'), successful_runs=0, total_runs=self.benchmark_runs
                )
    
    async def benchmark_triton_code(self, triton_code: str, test_setup: str,
                                  input_tensors: Dict[str, Any]) -> BenchmarkResult:
        """Benchmark Triton implementation"""
        benchmark_code = f"""
import torch
import triton
import triton.language as tl
import time
import gc

{triton_code}

{test_setup}

# Warmup runs
for _ in range({self.warmup_runs}):
    try:
        result = benchmark_function()
        torch.cuda.synchronize()
    except:
        pass

# Benchmark runs
times = []
gpu_memory_peak = 0

for run_idx in range({self.benchmark_runs}):
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    
    start_time = time.perf_counter()
    try:
        result = benchmark_function()
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        run_time = (end_time - start_time) * 1000  # Convert to ms
        times.append(run_time)
        
        peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
        gpu_memory_peak = max(gpu_memory_peak, peak_memory)
        
    except Exception as e:
        print(f"Triton benchmark run {{run_idx}} failed: {{e}}")
        continue

successful_runs = len(times)
if successful_runs > 0:
    mean_time = sum(times) / len(times)
    std_time = (sum((t - mean_time) ** 2 for t in times) / len(times)) ** 0.5 if len(times) > 1 else 0
    min_time = min(times)
    max_time = max(times)
    median_time = sorted(times)[len(times) // 2]
    
    print(f"BENCHMARK_RESULT:{{mean_time}},{{std_time}},{{min_time}},{{max_time}},{{median_time}},{{gpu_memory_peak}},{{successful_runs}}")
else:
    print("BENCHMARK_FAILED: No successful runs")
"""
        
        async with httpx.AsyncClient() as client:
            try:
                resp = await client.post(self.run_code_endpoint,
                                       json={"code": benchmark_code, "tests": ""},
                                       timeout=300.0)
                resp.raise_for_status()
                data = resp.json()
                
                if data["status_code"] == 0:
                    return self._parse_benchmark_result(data["stdout"])
                else:
                    return BenchmarkResult(
                        mean_time_ms=float('inf'), std_time_ms=0, min_time_ms=float('inf'),
                        max_time_ms=float('inf'), median_time_ms=float('inf'),
                        gpu_memory_peak_mb=float('inf'), successful_runs=0, total_runs=self.benchmark_runs
                    )
            except Exception as e:
                print(f"Triton benchmark failed: {e}")
                return BenchmarkResult(
                    mean_time_ms=float('inf'), std_time_ms=0, min_time_ms=float('inf'),
                    max_time_ms=float('inf'), median_time_ms=float('inf'),
                    gpu_memory_peak_mb=float('inf'), successful_runs=0, total_runs=self.benchmark_runs
                )
    
    def _parse_benchmark_result(self, stdout: str) -> BenchmarkResult:
        """Parse benchmark results from stdout"""
        for line in stdout.split('\n'):
            if line.startswith("BENCHMARK_RESULT:"):
                try:
                    values = line.split(":", 1)[1].split(",")
                    return BenchmarkResult(
                        mean_time_ms=float(values[0]),
                        std_time_ms=float(values[1]),
                        min_time_ms=float(values[2]),
                        max_time_ms=float(values[3]),
                        median_time_ms=float(values[4]),
                        gpu_memory_peak_mb=float(values[5]),
                        successful_runs=int(values[6]),
                        total_runs=self.benchmark_runs
                    )
                except (ValueError, IndexError) as e:
                    print(f"Failed to parse benchmark result: {e}")
                    break
        
        return BenchmarkResult(
            mean_time_ms=float('inf'), std_time_ms=0, min_time_ms=float('inf'),
            max_time_ms=float('inf'), median_time_ms=float('inf'),
            gpu_memory_peak_mb=float('inf'), successful_runs=0, total_runs=self.benchmark_runs
        )
    
    async def compare_implementations(self, pytorch_code: str, triton_code: str, 
                                    test_setup: str, input_tensors: Dict[str, Any]) -> ComparisonResult:
        """Compare PyTorch and Triton implementations"""
        
        # Run benchmarks in parallel
        pytorch_task = self.benchmark_pytorch_code(pytorch_code, test_setup, input_tensors)
        triton_task = self.benchmark_triton_code(triton_code, test_setup, input_tensors)
        
        pytorch_result, triton_result = await asyncio.gather(pytorch_task, triton_task)
        
        # Calculate comparison metrics
        if pytorch_result.successful_runs == 0 or triton_result.successful_runs == 0:
            speedup = 0.0
            memory_efficiency = 0.0
            performance_score = 0.0
        else:
            # Speedup: pytorch_time / triton_time (higher is better)
            speedup = pytorch_result.mean_time_ms / triton_result.mean_time_ms
            
            # Memory efficiency: pytorch_memory / triton_memory (higher is better)
            if triton_result.gpu_memory_peak_mb > 0:
                memory_efficiency = pytorch_result.gpu_memory_peak_mb / triton_result.gpu_memory_peak_mb
            else:
                memory_efficiency = 1.0
            
            # Performance score (0-1): combined metric
            # Weight speedup more heavily than memory efficiency
            performance_score = self._calculate_performance_score(speedup, memory_efficiency)
        
        triton_result.speedup_vs_pytorch = speedup
        triton_result.memory_efficiency_vs_pytorch = memory_efficiency
        
        return ComparisonResult(
            pytorch_result=pytorch_result,
            triton_result=triton_result,
            speedup=speedup,
            memory_efficiency=memory_efficiency,
            performance_score=performance_score
        )
    
    def _calculate_performance_score(self, speedup: float, memory_efficiency: float) -> float:
        """Calculate overall performance score (0-1)"""
        # Normalize speedup (anything above 2x is excellent)
        normalized_speedup = min(speedup / 2.0, 1.0)
        
        # Normalize memory efficiency (anything above 1x is good)
        normalized_memory = min(memory_efficiency, 1.0)
        
        # Weighted combination (80% speedup, 20% memory)
        score = 0.8 * normalized_speedup + 0.2 * normalized_memory
        return max(0.0, min(1.0, score))

# Global benchmarker instance
benchmarker = PerformanceBenchmarker() 