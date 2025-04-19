import random  
import weave
import math
import re
import torch

if torch.distributed.get_rank() == 0:
    weave.init("grpo-cuda/axolotl-grpo")

RUN_SAFE = True

from tools import extract_code, run_python_in_process, run_python_code

# generated deadlocks with tokenizers
# weave.init("grpo-cuda/axolotl-grpo")

AVAILABLE_GPUS = list(range(torch.cuda.device_count()))

# List of valid triton.language methods
# Sourced from triton.__dir__() and filtering for relevant functions
tl_methods = [
    'PropagateNan', 'TRITON_MAX_TENSOR_NUMEL', 'abs', 'advance', 'arange',
    'argmax', 'argmin', 'associative_scan', 'atomic_add', 'atomic_and',
    'atomic_cas', 'atomic_max', 'atomic_min', 'atomic_or', 'atomic_xchg',
    'atomic_xor', 'bfloat16', 'block_type', 'broadcast', 'broadcast_to',
    'cast', 'cat', 'cdiv', 'ceil', 'clamp', 'const', 'const_pointer_type',
    'constexpr', 'cos', 'cumprod', 'cumsum', 'debug_barrier', 'device_assert',
    'device_print', 'div_rn', 'dot', 'dtype', 'erf', 'exp', 'exp2',
    'expand_dims', 'fdiv', 'flip', 'float16', 'float32', 'float64',
    'float8e4b15', 'float8e4b8', 'float8e4nv', 'float8e5', 'float8e5b16',
    'floor', 'fma', 'full', 'function_type', 'histogram',
    'inline_asm_elementwise', 'int1', 'int16', 'int32', 'int64', 'int8',
    'interleave', 'join', 'load', 'log', 'log2', 'make_block_ptr', 'max',
    'max_constancy', 'max_contiguous', 'maximum', 'min', 'minimum',
    'multiple_of', 'num_programs', 'pair_uniform_to_normal', 'permute',
    'philox', 'pi32_t', 'pointer_type', 'program_id', 'rand', 'rand4x',
    'randint', 'randint4x', 'randn', 'randn4x', 'range', 'ravel', 'reduce',
    'reshape', 'rsqrt', 'sigmoid', 'sin', 'softmax', 'sort', 'split', 'sqrt',
    'sqrt_rn', 'static_assert', 'static_print', 'static_range', 'store',
    'str_to_ty', 'sum', 'swizzle2d', 'tensor', 'trans', 'uint16', 'uint32',
    'uint64', 'uint8', 'uint_to_uniform_float', 'umulhi', 'view', 'void',
    'where', 'xor_sum', 'zeros', 'zeros_like'
]
VALID_TL_METHODS = set(tl_methods)

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
        reward = 0.2 * math.exp(-think_score["thinking_length"]/1000) if ok else -0.1
        rewards.append(reward)
    return rewards

@weave.op
def one_code_blob(output):
    "Check if the output has exactly one Python code blob after removing the think block"
    # Remove the think block first
    output_without_think = re.sub(r"<think>.*?</think>", "", output, flags=re.DOTALL).strip()

    code_blobs = re.findall(r"```python(.*?)```", output_without_think, re.DOTALL)

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
    "Reward the model for having a single code blob after the think block"
    responses = [completion[0]['content'] for completion in completions]
    rewards = []
    for response in responses:
        # The scorer now handles removing the think block internally
        code_blob_score = one_code_blob(response)
        ok = code_blob_score["one_code_blob_ok"]
        # Penalize more heavily if no code blob found after removing think
        reward = 0.2 * math.exp(-code_blob_score["code_length"]/1000) if ok else -0.2
        rewards.append(reward)
    return rewards

@weave.op
def run_scorer(output: str, tests: str, pytorch_code_output: str):
    "Runs the code and returns the output"
    assert isinstance(tests, str), f"tests is not a string: {tests}"
    # run pt code
    gpu_id = random.choice(AVAILABLE_GPUS)

    triton_code = extract_code(output)
    if len(triton_code) < 10:
        return {"triton_runs": False, "pt_runs": True, "match": False}
    triton_and_test = f'import torch\n{triton_code}\n\n{"#"*146}\n\n{tests}'

    # Run the triton code
    if RUN_SAFE:
        triton_output = run_python_code(triton_and_test)
    else:
        triton_output = run_python_in_process(triton_and_test)

    match = (pytorch_code_output == triton_output["stdout"] 
             and triton_output["status_code"] == 0)

    return {"triton_runs": triton_output["status_code"] == 0,
            "pt_runs": True,
            "match": match}

def _compute_code_runs_reward(run_output):
    "If the code doesn't run, renturn -1, it if runs but doesn't match, return 0, otherwise return 1"
    triton_runs = run_output["triton_runs"]
    match = run_output["match"]
    if not triton_runs:
        return -0.25
    elif not match:
        return 0.2
    else:
        return 0.5

@weave.op
def reward_code_runs(completions, tests, pytorch_code_output, **kwargs):
    "Reward the model for the code running - runs checks in parallel"
    responses = [completion[0]['content'] for completion in completions]
    rewards = []

    # dummy seq run
    run_scores = [run_scorer(response, tests[0], pytorch_code_output[0]) for response in responses]


    # Extract the 'match' status from each result
    rewards = [_compute_code_runs_reward(score) for score in run_scores]
    return rewards

# ===== Static Code Analysis Rewards =====

@weave.op
def imports_decorator_scorer(code: str) -> dict:
    """Checks for required imports and @triton.jit decorator."""
    has_triton_import = "import triton" in code
    has_tl_import = "import triton.language as tl" in code
    has_jit_decorator = "@triton.jit" in code or "@tl.jit" in code # Allow both forms
    all_present = has_triton_import and has_tl_import and has_jit_decorator
    return {
        "has_triton_import": has_triton_import,
        "has_tl_import": has_tl_import,
        "has_jit_decorator": has_jit_decorator,
        "imports_decorator_ok": all_present,
    }

def imports_decorator_reward(completions, **kwargs):
    """Rewards 0.2 if required imports and decorator are present, 0 otherwise."""
    responses = [completion[0]['content'] for completion in completions]
    rewards = []
    for response in responses:
        triton_code = extract_code(response)
        if not triton_code:
            rewards.append(0) # Penalize lack of code elsewhere
            continue
        score = imports_decorator_scorer(triton_code)
        rewards.append(0.2 if score["imports_decorator_ok"] else 0)
    return rewards

@weave.op
def constexpr_scorer(code: str) -> dict:
    """Checks if tl.constexpr is used."""
    uses_constexpr = re.search(r"tl\.constexpr\s*\(", code) is not None
    return {"uses_constexpr": uses_constexpr}

def constexpr_reward(completions, **kwargs):
    """Rewards 0.2 if tl.constexpr is used, 0 otherwise."""
    responses = [completion[0]['content'] for completion in completions]
    rewards = []
    for response in responses:
        triton_code = extract_code(response)
        if not triton_code:
            rewards.append(0)
            continue
        score = constexpr_scorer(triton_code)
        rewards.append(0.2 if score["uses_constexpr"] else 0)
    return rewards

@weave.op
def valid_tl_methods_scorer(code: str) -> dict:
    """Checks if only valid triton.language methods are used."""
    # Find all occurrences of tl.<method_name>
    # This regex looks for 'tl.' followed by a valid Python identifier
    # We capture the identifier name
    used_methods = set(re.findall(r"tl\.([a-zA-Z_]\w*)", code))

    invalid_methods_used = used_methods - VALID_TL_METHODS
    all_valid = len(invalid_methods_used) == 0

    return {
        "all_tl_methods_valid": all_valid,
        "invalid_methods_found": list(invalid_methods_used) # For debugging
    }

def valid_tl_methods_reward(completions, **kwargs):
    """Rewards 0.2 if only valid tl methods are used, 0 otherwise."""
    responses = [completion[0]['content'] for completion in completions]
    rewards = []
    for response in responses:
        triton_code = extract_code(response)
        if not triton_code:
            rewards.append(0)
            continue
        score = valid_tl_methods_scorer(triton_code)
        rewards.append(0.2 if score["all_tl_methods_valid"] else 0)
    return rewards

@weave.op
def masks_load_store_scorer(code: str) -> dict:
    """Checks if masks are used during load and store operations."""
    # Check for tl.load(..., mask=...) or tl.store(..., mask=...)
    # This regex is simplified and might need refinement for complex cases
    uses_mask_load = re.search(r"tl\.load\s*\(.*mask\s*=", code, re.DOTALL) is not None
    uses_mask_store = re.search(r"tl\.store\s*\(.*mask\s*=", code, re.DOTALL) is not None

    # Require mask for at least one load or store if they exist
    has_load = "tl.load" in code
    has_store = "tl.store" in code

    uses_mask = False
    if has_load and uses_mask_load:
        uses_mask = True
    if has_store and uses_mask_store:
         uses_mask = True
    # If there are no loads or stores, this check doesn't apply negatively
    if not has_load and not has_store:
        uses_mask = True # Or maybe 0 reward? Let's say True for now.

    return {"uses_mask_load_store": uses_mask}

def masks_load_store_reward(completions, **kwargs):
    """Rewards 0.1 if masks are used with tl.load/tl.store, 0 otherwise."""
    responses = [completion[0]['content'] for completion in completions]
    rewards = []
    for response in responses:
        triton_code = extract_code(response)
        if not triton_code:
            rewards.append(0)
            continue
        score = masks_load_store_scorer(triton_code)
        rewards.append(0.1 if score["uses_mask_load_store"] else 0)
    return rewards

@weave.op
def torch_empty_scorer(code: str) -> dict:
    """Checks if torch.empty is used."""
    uses_torch_empty = re.search(r"torch\.empty\s*\(", code) is not None
    return {"uses_torch_empty": uses_torch_empty}

def torch_empty_penalty(completions, **kwargs):
    """Penalizes -0.1 if torch.empty is used, 0 otherwise."""
    responses = [completion[0]['content'] for completion in completions]
    rewards = []
    for response in responses:
        triton_code = extract_code(response)
        if not triton_code:
            rewards.append(0)
            continue
        # Check only within the kernel function definition if possible?
        # For now, check the whole extracted code blob.
        score = torch_empty_scorer(triton_code)
        rewards.append(-0.1 if score["uses_torch_empty"] else 0)
    return rewards

@weave.op
def torch_zeros_scorer(code: str) -> dict:
    """Checks if torch.zeros is used."""
    uses_torch_zeros = re.search(r"torch\.zeros\s*\(", code) is not None
    return {"uses_torch_zeros": uses_torch_zeros}

def torch_zeros_reward(completions, **kwargs):
    """Rewards 0.1 if torch.zeros is used, 0 otherwise."""
    responses = [completion[0]['content'] for completion in completions]
    rewards = []
    for response in responses:
        triton_code = extract_code(response)
        if not triton_code:
            rewards.append(0)
            continue
        # Similar to torch.empty, ideally check only in entrypoint.
        score = torch_zeros_scorer(triton_code)
        rewards.append(0.1 if score["uses_torch_zeros"] else 0)
    return rewards
