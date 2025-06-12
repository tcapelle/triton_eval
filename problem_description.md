# GRPO Training System for PyTorch-to-Triton Kernel Generation

## Overview

This project implements a **GRPO (Group Relative Policy Optimization)** training system that teaches large language models to convert PyTorch code into high-performance Triton GPU kernels. The system combines reinforcement learning with comprehensive reward functions to optimize both correctness and performance of generated kernels.

## System Architecture

### 1. Core Training Loop (GRPO)

The system uses GRPO, a reinforcement learning approach that:
- Takes PyTorch code as input
- Generates Triton kernel implementations 
- Evaluates outputs using multiple reward functions
- Updates model weights based on reward signals
- Iteratively improves kernel generation quality

**Key Configuration:**
- Base models: `facebook/KernelLLM`, `Qwen/Qwen2.5-Coder-14B-Instruct`, `predibase/Predibase-T2T-32B-RFT`
- Uses vLLM for efficient inference during training
- Multiple reward functions with configurable weights
- Training on synthetic PyTorch→Triton datasets

### 2. Multi-Stage Data Pipeline

The system employs several specialized prompts for different stages:

#### Stage 1: PyTorch Analysis & Conversion Planning
- **System Prompt**: `eval_system_prompt` - Expert GPU kernel reasoner
- **Process**: Analyzes PyTorch code, identifies optimization opportunities, creates conversion plan
- **Output**: Detailed reasoning + Triton implementation

#### Stage 2: Code Generation & Refinement  
- **System Prompt**: `predibase_system_prompt` - Converts PyTorch to Triton
- **Process**: Generates working Triton kernels with entrypoint functions
- **Quality Control**: Static analysis, syntax validation, execution testing

#### Stage 3: Test Generation & Validation
- **System Prompt**: `generate_pytorch_prompt` - Creates PyTorch equivalents with tests
- **Process**: Generates comprehensive test suites for validation
- **Cleanup**: `fixer_pytorch_prompt` fixes test issues, `formatting_prompt` standardizes code

## 3. Comprehensive Reward System

The reward system evaluates generated code across multiple dimensions:

### Structural Rewards
- **`think_reward`**: Rewards structured reasoning (≥100 char thinking blocks)
- **`one_code_blob_reward`**: Ensures single, clean code output
- **`language_reward`**: Promotes English responses, penalizes non-English

### Static Code Analysis Rewards
- **`imports_decorator_reward`**: Validates required imports (`import triton`, `@triton.jit`)
- **`constexpr_reward`**: Encourages use of `tl.constexpr` for compile-time constants
- **`valid_tl_methods_reward`**: Ensures only valid triton.language methods are used
- **`masks_load_store_reward`**: Promotes proper masking in memory operations
- **`torch_empty_penalty`**: Discourages inefficient tensor allocation patterns
- **`torch_zeros_reward`**: Rewards efficient zero tensor creation

### Dynamic Execution Rewards
- **`reward_code_runs`**: Primary reward based on correctness and execution
  - `-0.2`: Code fails to run
  - `0.0`: Code runs but produces incorrect results  
  - `+1.0`: Code runs and produces correct results

### Performance Rewards (In Development)
The system is being extended to include performance-based rewards using benchmarking:
- Execution time comparisons
- Memory usage efficiency
- GPU utilization metrics
- Speedup over PyTorch baselines

## 4. Safe Execution Environment

### Server Architecture (`server.py`)
- **FastAPI server** with worker pool management
- **Multi-GPU support** with automatic worker assignment
- **Process isolation** prevents crashes from affecting other tasks
- **Automatic recovery** respawns dead worker processes
- **Timeout handling** prevents hanging executions

### Worker Pool System
- Dedicated worker processes per GPU
- Async task execution with result collection
- Benchmarking capabilities for performance measurement
- Support for both Triton and PyTorch code execution

### Safety Features
- Sandboxed execution environment
- Resource limits and timeouts
- Crash isolation and recovery
- Comprehensive error logging and metrics

## 5. Training Data & Prompts

### Synthetic Dataset Generation
- **Base samples**: `scripts/data/simple_samples.jsonl` - Basic operations (add, mul, relu, etc.)
- **Complex operations**: softmax, layer_norm, attention, RoPE embeddings
- **Kernel patterns**: Element-wise, reductions, convolutions, normalization

### Prompt Engineering
Multiple specialized prompts handle different aspects:
- **Conversion prompts**: PyTorch → Triton translation
- **Analysis prompts**: Performance bottleneck identification  
- **Testing prompts**: Comprehensive test case generation
- **Reasoning prompts**: Step-by-step conversion planning
- **Quality prompts**: Code formatting and standardization

## 6. Key Performance Optimizations

### Triton Best Practices (Cookbook)
The system enforces Triton coding standards:
- Proper tiling and masking patterns
- Memory coalescing for 128-byte alignment
- Use of `tl.constexpr` for compile-time constants
- Efficient reduction patterns
- Warp specialization for small reductions
- Async pipeline for memory overlap

### Code Quality Standards
- Type hints and docstrings required
- Standardized device handling (`DEVICE = torch.device(...)`)
- Consistent test patterns with dictionary results
- Memory-efficient tensor operations
- Proper error handling and cleanup

## 7. Current Status & Next Steps

### Implemented Features ✅
- Complete GRPO training pipeline
- Multi-dimensional reward system  
- Safe execution environment
- Comprehensive prompt engineering
- Static code analysis rewards
- Basic correctness validation

### In Development 🚧
- **Performance-based rewards** using benchmarking metrics
- Integration of timing/memory usage into reward calculations
- Speedup measurement against PyTorch baselines
- Advanced performance profiling and optimization

### Future Enhancements 🔮
- Multi-kernel fusion optimization
- Auto-tuning parameter optimization
- Advanced memory hierarchy utilization
- Cross-platform kernel generation
- Integration with production ML pipelines

## 8. Usage Example

```python
# Training Configuration
rl: grpo
reward_funcs:
  - rewards.think_reward
  - rewards.one_code_blob_reward  
  - rewards.reward_code_runs
  - rewards.imports_decorator_reward
  # ... other rewards

# Input: PyTorch Code
def relu(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x)

# Expected Output: Optimized Triton Kernel
@triton.jit
def triton_relu_kernel(X, Y, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask, other=0.0)
    y = tl.maximum(x, 0.0)
    tl.store(Y + offs, y, mask=mask)
```

This system represents a significant advancement in automated GPU kernel optimization, combining the power of large language models with reinforcement learning to generate high-performance compute kernels automatically. 