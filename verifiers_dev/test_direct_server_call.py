"""Test direct server call to check if triton server is working."""

import httpx
import asyncio

SERVER_URL = "http://127.0.0.1:9347"
RUN_TRITON_ENDPOINT = f"{SERVER_URL}/run_triton"

# Valid triton code
triton_code = """
import torch
import triton
import triton.language as tl

@triton.jit
def relu_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    output = tl.maximum(x, 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)

def relu(x):
    output = torch.zeros_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    relu_kernel[grid](x, output, n_elements, BLOCK_SIZE)
    return output
"""

test_code = """
import torch

def test_relu():
    x = torch.randn(100, device='cuda')
    # Test with the triton kernel
    output = relu(x)
    expected = torch.relu(x)
    assert torch.allclose(output, expected)
    print("Test passed!")

test_relu()
"""

async def test_server_call():
    print(f"Testing server at {RUN_TRITON_ENDPOINT}")
    print("=" * 60)
    
    async with httpx.AsyncClient() as client:
        try:
            print("Sending request...")
            resp = await client.post(
                RUN_TRITON_ENDPOINT,
                json={
                    "code": triton_code, 
                    "tests": test_code,
                    "benchmark": True,
                    "benchmark_runs": 10
                },
                timeout=30.0
            )
            print(f"Status code: {resp.status_code}")
            print(f"Response: {resp.text}")
            
            if resp.status_code == 200:
                data = resp.json()
                print(f"Parsed response: {data}")
            
        except httpx.ConnectError as e:
            print(f"Connection error: {e}")
            print("The triton server might not be running at http://127.0.0.1:9347")
        except Exception as e:
            print(f"Error: {type(e).__name__}: {e}")

asyncio.run(test_server_call())