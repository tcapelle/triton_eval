#!/usr/bin/env python3

import torch
from datasets import load_dataset
from triton_rewards_modular import TritonEnvWithExecution
from openai import OpenAI
import httpx

def test_triton_scoring():
    """Test that the TritonEnvWithExecution correctly calls both static and API scoring."""
    
    print("Loading dataset...")
    train_dataset = load_dataset("tcapelle/boostrap_oai_pt_think", split="train")
    train_dataset = train_dataset.map(lambda row: {"prompt": row["prompt"][:-1]})
    train_dataset = train_dataset.map(lambda row: {"answer": row.get("triton_code", "")})
    
    # Take just a few samples for testing
    test_dataset = train_dataset.select(range(2))
    
    print("Creating TritonEnvWithExecution...")
    triton_env = TritonEnvWithExecution(dataset=test_dataset)
    
    print("Testing rubric scoring directly...")
    
    # Create a mock OpenAI client
    client = OpenAI(
        base_url="http://127.0.0.1:9347/v1",
        api_key="EMPTY",
        http_client=httpx.Client(timeout=30.0)
    )
    
    # Test with a simple completion
    test_prompt = test_dataset[0]['prompt']
    test_answer = test_dataset[0]['answer']
    test_info = {key: test_dataset[0].get(key, "") for key in 
                ['tests', 'stdout', 'entrypoint', 'benchmark_mean_time_ms', 'benchmark_memory_peak_mb']}
    
    print(f"Test prompt type: {type(test_prompt)}")
    print(f"Test answer length: {len(test_answer) if test_answer else 0}")
    print(f"Test info keys: {list(test_info.keys())}")
    
    # Mock completion with Triton code
    mock_completion = [{'role': 'assistant', 'content': '''
<think>
I need to create a Triton kernel for this function.
</think>

<triton>
import torch
import triton
import triton.language as tl

@triton.jit
def test_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = x * 2
    tl.store(y_ptr + offsets, output, mask=mask)

def triton_function(x):
    output = torch.zeros_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    test_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    return output
</triton>
'''}]
    
    print("\n" + "="*50)
    print("Testing combined rubric scoring...")
    print("="*50)
    
    # Test the rubric directly
    try:
        result = triton_env.rubric.score_rollouts(
            prompts=[test_prompt],
            completions=[mock_completion],
            answers=[test_answer],
            states=[{}],
            tasks=['default'],
            infos=[test_info],
            max_concurrent=1
        )
        
        print(f"\nScoring completed successfully!")
        print(f"Result keys: {list(result.keys())}")
        for key, values in result.items():
            if isinstance(values, list) and len(values) > 0:
                print(f"{key}: {values[0]}")
            else:
                print(f"{key}: {values}")
                
        # Check if we got the triton execution reward
        if 'triton_execution_reward' in result:
            print(f"\n✅ SUCCESS: API scoring (triton_execution_reward) was called!")
            print(f"triton_execution_reward: {result['triton_execution_reward']}")
        else:
            print(f"\n❌ FAILURE: API scoring was not called")
            
    except Exception as e:
        print(f"\n❌ ERROR during scoring: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_triton_scoring() 