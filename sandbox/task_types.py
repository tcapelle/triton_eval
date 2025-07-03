"""Shared data structures for the Triton Worker Pool Server."""

from pydantic import BaseModel, Field
from typing import Optional


class TaskResult(BaseModel):
    """Encapsulates all task execution results and metrics."""
    
    task_id: str = Field(..., description="Unique identifier for the task")
    status_code: int = Field(..., description="Exit code: 0 = success, -1 = error")
    stdout: str = Field(default="", description="Standard output from task execution")
    stderr: str = Field(default="", description="Standard error from task execution")
    
    # System metrics
    gpu_mem_used_gb: Optional[float] = Field(default=None, description="GPU memory usage in GB", ge=0)
    cpu_percent: Optional[float] = Field(default=None, description="CPU utilization percentage", ge=0, le=100)
    ram_percent: Optional[float] = Field(default=None, description="RAM utilization percentage", ge=0, le=100)
    
    # Benchmarking metrics
    benchmark_mean_time_ms: Optional[float] = Field(default=None, description="Average execution time in milliseconds", ge=0)
    benchmark_std_time_ms: Optional[float] = Field(default=None, description="Standard deviation of execution time", ge=0)
    benchmark_memory_peak_mb: Optional[float] = Field(default=None, description="Peak memory usage during benchmarking in MB", ge=0)
    benchmark_successful_runs: Optional[int] = Field(default=None, description="Number of successful benchmark runs", ge=0)
    
    # PyTorch-specific metrics
    torch_compile_benchmark_mean_time_ms: Optional[float] = Field(default=None, description="Average torch.compile execution time in milliseconds", ge=0)
    torch_compile_benchmark_std_time_ms: Optional[float] = Field(default=None, description="Standard deviation of torch.compile execution time", ge=0)
    torch_compile_speedup: Optional[float] = Field(default=None, description="Speedup ratio of compiled vs regular execution", ge=0)
    
    @property
    def is_successful(self) -> bool:
        """Check if the task completed successfully."""
        return self.status_code == 0
    
    @property
    def has_benchmarks(self) -> bool:
        """Check if benchmarking results are available."""
        return self.benchmark_mean_time_ms is not None
    
    @property
    def has_torch_compile_results(self) -> bool:
        """Check if torch.compile benchmarking results are available."""
        return self.torch_compile_benchmark_mean_time_ms is not None
    
    def get_summary(self) -> str:
        """Get a brief summary of the task result."""
        status = "✅ SUCCESS" if self.is_successful else "❌ FAILED"
        parts = [f"Task {self.task_id[:8]}...: {status}"]
        
        if self.has_benchmarks:
            parts.append(f"⏱️  {self.benchmark_mean_time_ms:.2f}ms")
            if self.benchmark_successful_runs:
                parts.append(f"({self.benchmark_successful_runs} runs)")
        
        if self.has_torch_compile_results and self.torch_compile_speedup:
            parts.append(f"🚀 {self.torch_compile_speedup:.2f}x speedup")
        
        if self.gpu_mem_used_gb:
            parts.append(f"🖥️  {self.gpu_mem_used_gb:.1f}GB GPU")
        
        return " | ".join(parts) 