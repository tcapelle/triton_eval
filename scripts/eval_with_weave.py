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

from my_smol_agent.tools import remove_tests, extract_code, run_python_code

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
    MODEL_NAME = "Qwen/Qwen2.5-Coder-14B-Instruct"
    # MODEL_NAME = "Qwen/Qwen2.5-Coder-32B-Instruct"
    # MODEL_NAME = "Qwen/Qwen2.5-Coder-14B-Instruct-ft-206"
    # MODEL_NAME = "Qwen/Qwen2.5-Coder-14B-Instruct-ft-206-v1"
    # MODEL_NAME = "Qwen/Qwen2.5-Coder-14B-Instruct-ft-309-v2"
else:
    MODEL_NAME = "o4-mini-2025-04-16"



TEMPERATURE = 0.0
TIMEOUT = 60
AVAILABLE_GPUS = [4, 5, 6, 7]

@dataclass
class ScriptArgs:
    trials: int = 1
    base_url: str = CUSTOM_BASE_URL
    model_name: str = MODEL_NAME
    temperature: float = TEMPERATURE
    max_tokens: int = 3000
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
- Ensure that the wrapper function fully corresponds to the provided function information.
- Generate a detailed plan on how to convert and optimize the Pytorch code to a Triton kernel before writing the code.
- The reasoning process MUST BE enclosed within <think> and </think> tags."
- Reply with the thinking process and a single blob of code surrounded with ```python and ```.
"""

user_prompt = """Convert the following PyTorch code to a Triton kernel.
Pytorch code:
```python
{pt_code}```

The function should have the same name as the PyTorch function. 

Don't forget to format your answer as:
<think>
thinking process
</think>
```python
code
```"""

##############################################


def call_model(system_prompt: str, user_prompt: str, model_name: str, **model_kwargs):
    "Use reponse API for o3/o4 models, otherwise use chat completion"
    if model_name.startswith("o"):
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
    def predict(self, pt_code: str):
        code = remove_tests(pt_code)
        "Takes a code string and returns a response from the model"
        out = call_model(
            self.system_prompt, 
            self.user_prompt.format(pt_code=code), 
            self.model_name, 
            temperature=self.temperature, 
            max_tokens=self.max_tokens)
        return out
    

@weave.op
def run_scorer(output, tests, stdout, runs):
    "Runs the code and returns the output"
    gpu_id = random.choice(AVAILABLE_GPUS)
    
    # get triton from model output
    triton_code = extract_code(output)

    triton_and_test = f'import torch\n{triton_code}\n\n{"#"*146}\n\n{tests}'

    # Run the triton code
    triton_output = run_python_code(triton_and_test, env={"CUDA_VISIBLE_DEVICES": str(gpu_id)}, timeout=TIMEOUT)
    triton_runs = triton_output["status_code"] == 0
    triton_stdout = triton_output["stdout"]
    match = (stdout == triton_stdout and runs and triton_runs)

    return {"triton_runs": triton_runs,
            "triton_stdout": triton_stdout,
            "pt_runs": runs,
            "match": match}

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