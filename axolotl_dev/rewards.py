import random  
import weave
import math
import re
import torch
from concurrent.futures import ProcessPoolExecutor

from tools import extract_code, extract_tests, run_python_code

weave.init("grpo-cuda/axolotl-grpo")

AVAILABLE_GPUS = list(range(torch.cuda.device_count()))

# ===== Reward Functions =====
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

def think_reward(completions, **kwargs):
    "Reward the model for having a thinking process"
    responses = [completion[0]['content'] for completion in completions]
    rewards = []
    for response in responses:
        think_score = think_scorer(response)
        ok = think_score["thinking_ok"]
        reward = 1 * math.exp(-think_score["thinking_length"]/1000) if ok else -1
        rewards.append(reward)
    return rewards

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

def one_code_blob_reward(completions, **kwargs):
    "Reward the model for having a single code blob"
    responses = [completion[0]['content'] for completion in completions]
    rewards = []
    for response in responses:
        code_blob_score = one_code_blob(response)
        ok = code_blob_score["one_code_blob_ok"]
        reward = 1 * math.exp(-code_blob_score["code_length"]/1000) if ok else -1
        rewards.append(reward)
    return rewards

@weave.op
def run_scorer(output: str, tests: str, pytorch_code_output: str):
    "Runs the code and returns the output"
    assert isinstance(tests, str), f"tests is not a string: {tests}"
    # run pt code
    gpu_id = random.choice(AVAILABLE_GPUS)

    triton_code = extract_code(output)
    triton_and_test = f'import torch\n{triton_code}\n\n{"#"*146}\n\n{tests}'

    # Run the triton code
    triton_output = run_python_code(triton_and_test, env={"CUDA_VISIBLE_DEVICES": str(gpu_id)})

    match = (pytorch_code_output == triton_output["stdout"] 
             and triton_output["status_code"] == 0)

    return {"triton_runs": triton_output["status_code"] == 0,
            "pt_runs": True,
            "match": match}

def _compute_code_runs_reward(run_output):
    "If the code doesn't run, renturn -1, it if runs but doesn't match, return 0, otherwise return 1"
    triton_runs = run_output["triton_runs"]
    pt_runs = run_output["pt_runs"]
    match = run_output["match"]
    if not triton_runs or not pt_runs:
        return -1
    elif not match:
        return 0
    else:
        return 1

@weave.op
def reward_code_runs(completions, tests, pytorch_code_output, **kwargs):
    "Reward the model for the code running - runs checks in parallel"
    responses = [completion[0]['content'] for completion in completions]
    rewards = []

    num_workers = len(AVAILABLE_GPUS) if AVAILABLE_GPUS else 1
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        futures = [executor.submit(run_scorer, response, tests[0], pytorch_code_output[0]) for response in responses]

        # Collect results as they complete (maintaining order)
        run_scores = [future.result() for future in futures]

    # Extract the 'match' status from each result
    rewards = [_compute_code_runs_reward(score) for score in run_scores]
    return rewards

