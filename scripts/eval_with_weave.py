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



use_openai = False

if not use_openai:
    CUSTOM_BASE_URL = "http://0.0.0.0:8000/v1"
else:
    CUSTOM_BASE_URL = None

if not use_openai:
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
    MODEL_NAME = "qwen-14b-sft"
else:
    MODEL_NAME = "codex-mini-latest"



TEMPERATURE = 0.0
TIMEOUT = 60
AVAILABLE_GPUS = [4, 5, 6, 7]

@dataclass
class ScriptArgs:
    trials: int = 1
    base_url: str = CUSTOM_BASE_URL
    model_name: str = MODEL_NAME
    temperature: float = TEMPERATURE
    max_tokens: int = 6000
    weave_project: str = "grpo-cuda/triton-bench"
    weave_dataset: str = "Tritonbench_T_v2:latest"
    debug: bool = False

console.rule("[bold green]Running Weave Eval[/bold green]")

args = sp.parse(ScriptArgs)
print(args)

client = openai.OpenAI(
    base_url=args.base_url,
)
weave.init(args.weave_project)

ds = weave.ref(args.weave_dataset).get()
if args.debug:
    ds = ds.rows[:10]


## TRAINING PROMPT ###########################
system_prompt = """
You are an expert in Triton programming, capable of writing corresponding Triton kernels and wrapper functions based on functional descriptions and function parameters.
# Instructions
- Ensure that the Triton wrapper function matches the signature of the provided PyTorch function and calls the Triton implementation.
- Generate a detailed plan on how to convert and optimize the PyTorch code to a Triton kernel before writing the code.
- The reasoning process MUST BE enclosed within <think> and </think> tags.
- Reply with the reasoning process and the Triton kernel within a single code block enclosed in "```python" and "```".
"""

user_prompt = """Convert the following PyTorch code to a Triton kernel.
Pytorch code:

```python
{pt_code}
```

The entrypoint function must be named: {entrypoint}
The Triton kernel implementation (called by the entrypoint) must be named: {entrypoint}_kernel

No computation logic should be done within the entrypoint function. All computation logic should be done within the Triton kernel implementation.
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

    triton_and_test = f'import torch\n{triton_code}\n\n{"#"*146}\n\n{tests}'

    # Run the triton code
    if analysis["is_valid"]:
        triton_output = run_python_code(triton_and_test, env={"CUDA_VISIBLE_DEVICES": str(gpu_id)}, timeout=TIMEOUT)
        triton_runs = triton_output["status_code"] == 0
        triton_stdout = triton_output["stdout"]
        match = (stdout == triton_stdout and runs and triton_runs)
    else:
        match = False
        triton_runs = False
        triton_stdout = ""

    result = {
        "triton_runs": triton_runs,
        "triton_stdout": triton_stdout,
        "pt_runs": runs,
        "match": match,
        "analysis": analysis
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

@weave.op
def one_code_blob(output):
    "Check if the output has exactly one Python code blob"
    code_blobs = re.findall(r"```python(.*?)```", output, re.DOTALL)

    num_matches = len(code_blobs)
    one_code_blob_ok = False
    code_length = 0

    if num_matches == 1:
        content = code_blobs[0].strip()
        code_length = len(content)
        if code_length > 0:
            one_code_blob_ok = True

    return {"one_code_blob_ok": one_code_blob_ok, "code_length": code_length}

scorers = [run_scorer, think_scorer, one_code_blob]

weave_model = OpenAICompatibleModel(
    model_name=args.model_name,
    temperature=args.temperature,
    max_tokens=args.max_tokens,
    system_prompt=system_prompt,
    user_prompt=user_prompt,
)

evaluation = weave.Evaluation(dataset=ds, scorers=scorers, trials=args.trials, evaluation_name=args.model_name)

asyncio.run(evaluation.evaluate(model=weave_model))