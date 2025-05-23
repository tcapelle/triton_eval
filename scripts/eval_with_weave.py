import os
import re
import asyncio
import random
from dataclasses import dataclass
from pathlib import Path

import weave
import openai
import simple_parsing as sp
from rich.console import Console

from triton_eval.agents.tools import remove_tests, extract_code, run_python_code
from triton_eval.kernel_checks import is_valid_kernel

script_dir = os.path.dirname(os.path.abspath(__file__))

console = Console()


# MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
# MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct-ft"
# MODEL_NAME = "Qwen/Qwen2.5-Coder-14B-Instruct"
# MODEL_NAME = "Qwen/Qwen2.5-Coder-32B-Instruct"
# MODEL_NAME = "Qwen/Qwen2.5-Coder-14B-Instruct-ft-206"
# MODEL_NAME = "Qwen/Qwen2.5-Coder-14B-Instruct-ft-206-v1"
# MODEL_NAME = "Qwen/Qwen2.5-Coder-14B-Instruct-ft-309-v2"
# MODEL_NAME = "qwen-coder-14b-ft-v3"
# MODEL_NAME = "qwen-coder-32b-ft-v1"
# MODEL_NAME = "qwen3-14b-ft"
# MODEL_NAME = "qwen3-14b-sft-grpo"
# MODEL_NAME = "qwen3-14b-sft4"
# MODEL_NAME = "predibase-32b"
# MODEL_NAME = "kernelllm"
# MODEL_NAME = "qwen-14b-sft"
# MODEL_NAME = "devstral"
# MODEL_NAME = "qwen3-14b"
MODEL_NAME = "qwen2.5-coder-14b-instruct"




TEMPERATURE = 0.0
TIMEOUT = 60
AVAILABLE_GPUS = [4, 5, 6, 7]

@dataclass
class ScriptArgs:
    trials: int = 1
    model_name: str = MODEL_NAME
    temperature: float = TEMPERATURE
    max_tokens: int = 6000
    weave_project: str = "grpo-cuda/triton-bench"
    weave_dataset: str = "Tritonbench_T_v2:latest"
    debug: bool = False
    use_openai: bool = False

console.rule("[bold green]Running Weave Eval[/bold green]")

args = sp.parse(ScriptArgs)
print(args)


if not args.use_openai:
    CUSTOM_BASE_URL = "http://0.0.0.0:8000/v1"
else:
    CUSTOM_BASE_URL = None

client = openai.OpenAI(
    base_url=CUSTOM_BASE_URL,
)
weave.init(args.weave_project)

ds = weave.ref(args.weave_dataset).get()
if args.debug:
    ds = ds.rows[:10]


## TRAINING PROMPT ###########################
system_prompt = """
# GPU‐Kernel Reasoner Prompt

You are an expert GPU‐kernel reasoner and Triton evangelist. You will be given a PyTorch code snippet. Your goal is to:

1. **Analyze the PyTorch implementation**  
   - Break down its algorithmic steps, memory access patterns, and computational characteristics.  
   - Identify potential performance bottlenecks or numerical‐stability issues in the PyTorch version.  
   - List the **Pros** of the existing PyTorch code (e.g., readability, use of optimized libraries) and the **Cons** (e.g., extra memory traffic, kernel launch overhead).

2. **Build a detailed Conversion Plan**  
   - Walk through *every* design decision and transformation needed to convert the PyTorch code into a high‐performance Triton kernel plus a Python wrapper.  

3. **Deliver the Final Implementation**  
   - The Triton kernel annotated with key parameters.  
   - A drop‐in Python wrapper matching the original PyTorch function’s signature.

---

## Output Format

### 1. PyTorch Analysis  
- **Algorithmic Steps:** numbered list of what the PyTorch code does.  
- **Memory & Compute Characteristics:** brief notes on data layout, reductions, fusions, etc.  
- **Pros:** bullet list of strengths in the current implementation.  
- **Cons:** bullet list of limitations or inefficiencies that Triton could address.

### 2. Conversion Plan  
A numbered list of **8–12 steps**. Each step must:  
- Describe one concept or decision in detail (index calculations, grid dimensions, block/tile mapping, masking, memory layout, fusion, numerical‐stability tricks, vectorization strategy, etc.).  
- Reference the specific PyTorch lines or operations and their Triton equivalents (`tl.load`, `tl.store`, `tl.arange`, `program_id`, masks, etc.).  
- Explain *why* each constant or strategy is chosen (block size, tile shape, use of shared memory vs. registers, data types).  
- Include notes on performance considerations (kernel launch overhead, memory bandwidth, GPU occupancy) and any numerical‐stability hacks (e.g., subtracting the max before `exp`).

### 3. Final Implementation  
Two fenced Python code blocks:

1. **Triton Kernel**  
   - Annotated with parameter comments (`stride`, `BLOCK_SIZE`, etc.).  
   - Inline markers showing where each step of the Conversion Plan is realized.

2. **Python Wrapper**  
   - Exact same function name and signature as the original PyTorch version.  
   - Allocates the output tensor, computes grid dimensions, launches the Triton kernel, and returns the result.  
   - Contains concise comments linking back to key Conversion Plan steps.


To make our life easier, enclose all the reasoning and conversion plan with <think> ... </think> tags. For the final implementation, reply with a single blob of code enclosed with <triton> ... </triton> tags.
"""

user_prompt = """Convert the following PyTorch code to a Triton kernel.
Pytorch code:

```python
{pt_code}
```

The entrypoint function must be named: {entrypoint}
The Triton kernel implementation (called by the entrypoint) must be named: {entrypoint}_kernel

No computation logic should be done within the entrypoint function. All computation logic should be done within the Triton kernel implementation.
Enclose the conversion reasoning with <think> ... </think> and the implementation with <triton> ... </triton> tags.
"""

##############################################

predibase_system_prompt = """You are a helpful assistant that converts PyTorch code into Triton kernels."""


predibase_user_prompt = """Convert this PyTorch module implementation into an equivalent Triton kernel:

<torch_code>
{pt_code}
</torch_code>

The Triton kernel should:
1. Import torch, triton, and triton.language as tl and other necessary modules
2. Use @triton.jit decorator on the kernel implementation (not the entrypoint function)
3. Have proper grid and block sizes
4. Use a mask in the load/store operations
5. Use typed constants (tl.constexpr)
6. Handle tensor dimensions correctly
7. Return output matching PyTorch's implementation
8. Do not include any test code in your response, only the Triton kernel implementation and entrypoint function

The entrypoint function must be named: {entrypoint}
The Triton kernel implementation (called by the entrypoint) must be named: {entrypoint}_kernel

No computation logic should be done within the entrypoint function. All computation logic should be done within the Triton kernel implementation.

The final generated code in the response must start with <triton_code> and end with </triton_code> tags.
"""

@weave.op
def predibase_extract_code(code: str) -> str:
    pattern = r"<triton_code>(.*?)</triton_code>"
    matches = re.findall(pattern, code, re.DOTALL)
    if matches:
        return matches[-1].strip()  # Get the last match
    else:
        return ""

    
# extract_code = predibase_extract_code
# system_prompt = predibase_system_prompt
# user_prompt = predibase_user_prompt

##############################################



def call_model(system_prompt: str, user_prompt: str, model_name: str, **model_kwargs):
    "Use reponse API for o3/o4 models, otherwise use chat completion"
    if model_name.startswith("o") or model_name.startswith("codex"):
        out = client.responses.create(
            model=model_name,
            input=[{"role": "system", "content": system_prompt}, 
                    {"role": "user", "content": user_prompt}],
            reasoning={"effort": "medium"},
            ).output_text.strip()
    else:
        out = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "system", "content": system_prompt}, 
                        {"role": "user", "content": user_prompt}],
            **model_kwargs
        ).choices[0].message.content.strip()
    return out




class OpenAICompatibleModel(weave.Model):
    "this is just a pydantic BaseModel subclass"
    model_name: str
    temperature: float
    max_tokens: int = 3000
    system_prompt: str
    user_prompt: str


    @weave.op
    def predict(self, pt_code: str, entrypoint: str):
        code = remove_tests(pt_code)
        "Takes a code string and returns a response from the model"
        out = call_model(
            self.system_prompt, 
            self.user_prompt.format(pt_code=code, entrypoint=entrypoint), 
            self.model_name, 
            temperature=self.temperature, 
            max_tokens=self.max_tokens)
        return out


@weave.op
def run_scorer(output, tests, stdout, runs, entrypoint):
    "Runs the code and returns the output"
    gpu_id = random.choice(AVAILABLE_GPUS)
    
    # get triton from model output
    triton_code = extract_code(output)

    # check valid kernel
    analysis = is_valid_kernel(triton_code, entrypoint)

    triton_and_test = f'import torch\nfrom typing import *\n{triton_code}\n\n{"#"*146}\n\n{tests}'

    # Run the triton code
    if analysis["is_valid"]:
        triton_output = run_python_code(triton_and_test, env={"CUDA_VISIBLE_DEVICES": str(gpu_id)}, timeout=TIMEOUT)
        triton_runs = triton_output["status_code"] == 0
        triton_stdout = triton_output["stdout"]
        triton_stderr = triton_output["stderr"]
        is_correct = (stdout == triton_stdout and runs and triton_runs)
    else:
        is_correct = False
        triton_runs = False
        triton_stdout = ""
        triton_stderr = ""
    result = {
        "is_valid": analysis["is_valid"],
        "triton_runs": triton_runs,
        "triton_stdout": triton_stdout,
        "triton_stderr": triton_stderr,
        "is_correct": is_correct,
        "validity": analysis["reason"]
    }

    return result

@weave.op
def think_scorer(output):
    "Check if the output has exactly one <think> block with content >= 10 chars"
    thinking_content = re.findall(r"<think>(.*?)</think>", output, re.DOTALL)

    num_matches = len(thinking_content)
    thinking_ok = False
    thinking_length = 0

    if num_matches == 1:
        content_length = len(thinking_content[0].strip())
        thinking_length = content_length
        if content_length >= 10:
            thinking_ok = True

    return {"thinking_ok": thinking_ok, "thinking_length": thinking_length}


scorers = [run_scorer, think_scorer]

weave_model = OpenAICompatibleModel(
    model_name=args.model_name,
    temperature=args.temperature,
    max_tokens=args.max_tokens,
    system_prompt=system_prompt,
    user_prompt=user_prompt,
)

evaluation = weave.Evaluation(dataset=ds, scorers=scorers, trials=args.trials, evaluation_name=args.model_name)

asyncio.run(evaluation.evaluate(model=weave_model))