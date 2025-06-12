# Performance-Based Rewards Usage Guide

## Overview

The GRPO training system now includes performance-based rewards that evaluate Triton kernels based on their execution speed and memory efficiency compared to PyTorch baselines. This enables the model to learn not just correctness, but also performance optimization.

## Key Features

### 🚀 **Automatic Performance Evaluation**
- **Single server call** per completion with benchmarking enabled
- Benchmarks Triton kernels with multiple runs for statistical accuracy
- Compares against pre-computed PyTorch baseline performance
- Rewards speedups using logarithmic scaling
- Penalizes slowdowns proportionally

### 📊 **Comprehensive Metrics**
- **Execution Time**: Primary performance metric
- **Memory Efficiency**: Bonus for low memory usage
- **Statistical Reliability**: Multiple benchmark runs with std deviation
- **Speedup Ratios**: Clear performance comparisons

### 💡 **Smart Reward Scaling**
- **1.5x speedup** → +0.41 performance reward
- **2.0x speedup** → +0.69 performance reward  
- **3.0x speedup** → +1.10 performance reward
- **0.5x speed (50% slower)** → -0.25 penalty

## Integration in Training

### 1. Dataset Preparation
Ensure your dataset includes PyTorch benchmark times:

```python
dataset_sample = {
    "triton_code": "...",
    "tests": "...", 
    "pytorch_benchmark_times": [12.5, 8.3, 15.2],  # Pre-computed baseline times in ms
    "stdout": ["...", "...", "..."],
    "entrypoint": ["kernel_name", ...]
}
```

### 2. Training Configuration
No changes needed! The existing configuration automatically includes performance rewards:

```yaml
rl: grpo
reward_funcs:
  - rewards.think_reward
  - rewards.one_code_blob_reward  
  - rewards.reward_code_runs        # Now includes performance rewards automatically!
  - rewards.imports_decorator_reward
  - rewards.constexpr_reward
  # ... other rewards
```

The `reward_code_runs` function automatically:
- ✅ **Executes Triton code once** with benchmarking enabled
- ✅ **Evaluates correctness** (existing behavior)
- ✅ **Calculates performance rewards** when PyTorch baselines are available
- ✅ **Combines both into a single reward** (no duplicate server calls)

## Reward Calculation Details

### Base Correctness (unchanged)
- **Code fails to run**: -0.2
- **Code runs but incorrect**: 0.0  
- **Code runs and correct**: +1.0

### Performance Bonus (new)
- **Speedup > 1.0**: `+1.0 * log(speedup)`
- **Slowdown < 1.0**: `-0.5 * (1 - speed_ratio)`
- **Memory efficient (< 50MB)**: +0.2 bonus
- **Benchmark failure**: -0.1 penalty

### Example Total Rewards
```python
# 2x faster kernel with low memory usage
reward = 1.0 (correct) + 0.69 (2x speedup) + 0.2 (memory) = 1.89

# 50% slower kernel  
reward = 1.0 (correct) - 0.25 (slowdown) + 0.2 (memory) = 0.95

# Broken kernel
reward = -0.2 (failed) + 0 (no performance) = -0.2
```

## Server Configuration

The system automatically uses the `/run_triton` endpoint with benchmarking enabled:

```python
# Automatic benchmarking configuration
SERVER_URL = "http://127.0.0.1:9347"
benchmark_runs = 10  # Multiple runs for accuracy
timeout = 300.0      # Longer timeout for benchmarking
```

## Testing the System

Run the test script to verify everything works:

```bash
python axolotl_dev/test_performance_rewards.py
```

Expected output:
```
🧪 Testing Performance Reward System
==================================================
Testing performance_scorer...
2x speedup score: {'has_benchmark_data': True, 'speedup': 2.0, 'is_faster': True, ...}
✅ performance_scorer tests passed!

Testing _compute_code_runs_reward...
Correct + 2x speedup + memory bonus: 1.893 (expected: 1.893)
✅ _compute_code_runs_reward tests passed!

🎉 All tests passed! Performance reward system is working correctly.
```

## Monitoring Performance

### Weights & Biases Integration
Performance metrics are automatically logged to W&B:

```python
# Logged metrics include:
{
    "speedup": 2.1,
    "triton_time_ms": 8.5,
    "pytorch_time_ms": 17.8,
    "memory_mb": 42.3,
    "performance_reward": 0.74
}
```

### Weave Tracking
All performance scoring is tracked with Weave for debugging:

```python
@weave.op
def performance_scorer(...)  # Automatically tracked
```

## Best Practices

### 1. Dataset Quality
- Ensure PyTorch baselines are accurate and representative
- Use consistent benchmarking conditions
- Include diverse kernel types and sizes

### 2. Reward Balancing
- Monitor total reward distributions
- Adjust `REWARD_MAGNITUDES` if needed
- Balance correctness vs performance emphasis

### 3. Training Monitoring
- Track performance improvements over training
- Monitor for reward gaming/exploitation
- Validate on held-out performance benchmarks

## Troubleshooting

### Common Issues

**No performance rewards appearing:**
- Check `pytorch_benchmark_times` in dataset
- Verify server is running with `/run_triton` endpoint
- Ensure benchmarking is enabled

**Inconsistent performance measurements:**
- Increase `benchmark_runs` for more stable measurements  
- Check for GPU memory pressure
- Verify consistent execution environment

**Unexpected reward values:**
- Run `test_performance_rewards.py` to verify calculations
- Check W&B logs for detailed performance metrics
- Review `REWARD_MAGNITUDES` configuration

This system represents a significant advancement in automated GPU kernel optimization, now optimizing for both correctness AND performance! 🚀 