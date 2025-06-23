import re
import math
import asyncio
import httpx
import os
import weave
from typing import Dict, Any, List
from verifiers.parsers import XMLParser
from verifiers.envs import SingleTurnEnv
from verifiers.rubrics import Rubric, RubricGroup

# Import kernel validation tools
from triton_eval.utils import compare_outputs
from triton_eval.agents.tools import extract_code
from triton_eval.kernel_checks import is_valid_kernel


# Server configuration
SERVER_URL = os.environ.get("TRITON_SERVER_URL", "http://127.0.0.1:9347")
RUN_TRITON_ENDPOINT = f"/run_triton"
BENCHMARK_RUNS = 10

# Valid triton.language methods
VALID_TL_METHODS = set([
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
])


# Static reward functions (cheap to compute)
@weave.op
def think_reward(completion, answer=None, state=None, task=None, info=None, **kwargs) -> float:
    """Reward for having a thinking process."""
    # Handle both string and list of messages format
    if isinstance(completion, list):
        completion = completion[-1]['content'] if completion else ""
    thinking_content = re.findall(r"<think>(.*?)</think>", completion, re.DOTALL)
    
    if len(thinking_content) == 1:
        content_length = len(thinking_content[0].strip())
        if content_length >= 100:
            thinking_length = max(content_length - 5000, 0)
            return 0.1 * math.exp(-0.5*thinking_length/1000)
    
    return -0.1


@weave.op
def one_code_blob_reward(completion, answer=None, state=None, task=None, info=None, **kwargs) -> float:
    """Reward for having a single code blob after the think block."""
    # Handle both string and list of messages format
    if isinstance(completion, list):
        completion = completion[-1]['content'] if completion else ""
    output_without_think = completion.split("</think>")[-1].strip() if "</think>" in completion else completion
    code_blobs = re.findall(r"<triton>(.*?)</triton>", output_without_think, re.DOTALL)
    
    if len(code_blobs) == 1:
        code_length = len(code_blobs[0].strip())
        if code_length > 0:
            code_length = max(code_length - 5000, 0)
            return 0.1 * math.exp(-0.5*code_length/1000)
    
    return -0.1


@weave.op
def imports_decorator_reward(completion, answer=None, state=None, task=None, info=None, **kwargs) -> float:
    """Rewards if required imports and decorator are present."""
    # Handle both string and list of messages format
    if isinstance(completion, list):
        completion = completion[-1]['content'] if completion else ""
    code = extract_code(completion)
    
    if not code:
        return 0
    
    has_triton_import = "import triton" in code
    has_tl_import = "import triton.language as tl" in code
    has_jit_decorator = "@triton.jit" in code or "@tl.jit" in code
    
    return 0.1 if has_triton_import and has_tl_import and has_jit_decorator else 0


@weave.op
def constexpr_reward(completion, answer=None, state=None, task=None, info=None, **kwargs) -> float:
    """Rewards if tl.constexpr is used."""
    # Handle both string and list of messages format
    if isinstance(completion, list):
        completion = completion[-1]['content'] if completion else ""
    code = extract_code(completion)
    
    if not code:
        return 0
    
    uses_constexpr = re.search(r"tl\.constexpr[,\s]+", code) is not None
    return 0.1 if uses_constexpr else 0


@weave.op
def valid_tl_methods_reward(completion, answer=None, state=None, task=None, info=None, **kwargs) -> float:
    """Rewards if only valid tl methods are used."""
    # Handle both string and list of messages format
    if isinstance(completion, list):
        completion = completion[-1]['content'] if completion else ""
    code = extract_code(completion)
    
    if not code:
        return 0
    
    used_methods = set(re.findall(r"tl\.([a-zA-Z_]\w*)", code))
    invalid_methods = used_methods - VALID_TL_METHODS
    
    return 0.1 if len(invalid_methods) == 0 else 0


@weave.op
def masks_load_store_reward(completion, answer=None, state=None, task=None, info=None, **kwargs) -> float:
    """Rewards if masks are used with tl.load/tl.store."""
    # Handle both string and list of messages format
    if isinstance(completion, list):
        completion = completion[-1]['content'] if completion else ""
    code = extract_code(completion)
    
    if not code:
        return 0
    
    uses_mask_load = re.search(r"tl\.load\s*\(.*mask\s*=", code, re.DOTALL) is not None
    uses_mask_store = re.search(r"tl\.store\s*\(.*mask\s*=", code, re.DOTALL) is not None
    
    has_load = "tl.load" in code
    has_store = "tl.store" in code
    
    uses_mask = False
    if has_load and uses_mask_load:
        uses_mask = True
    if has_store and uses_mask_store:
        uses_mask = True
    if not has_load and not has_store:
        uses_mask = True
    
    return 0.1 if uses_mask else 0


@weave.op
def torch_empty_penalty(completion, answer=None, state=None, task=None, info=None, **kwargs) -> float:
    """Penalizes if torch.empty is used."""
    # Handle both string and list of messages format
    if isinstance(completion, list):
        completion = completion[-1]['content'] if completion else ""
    code = extract_code(completion)
    
    if not code:
        return 0
    
    uses_torch_empty = re.search(r"torch\.empty\s*\(", code) is not None
    return -0.1 if uses_torch_empty else 0


@weave.op
def torch_zeros_reward(completion, answer=None, state=None, task=None, info=None, **kwargs) -> float:
    """Rewards if torch.zeros is used."""
    # Handle both string and list of messages format
    if isinstance(completion, list):
        completion = completion[-1]['content'] if completion else ""
    code = extract_code(completion)
    
    if not code:
        return 0
    
    uses_torch_zeros = re.search(r"torch\.zeros\s*\(", code) is not None
    return 0.1 if uses_torch_zeros else 0


@weave.op
def language_reward(completion, answer=None, state=None, task=None, info=None, **kwargs) -> float:
    """Reward English responses."""
    return 0.1  # Simplified - assumes English


# Create static rubric with all the cheap rewards
def create_static_rubric(parser: XMLParser) -> Rubric:
    """Create rubric with static (cheap) reward functions."""
    return Rubric(
        funcs=[
            think_reward,
            one_code_blob_reward,
            imports_decorator_reward,
            constexpr_reward,
            valid_tl_methods_reward,
            masks_load_store_reward,
            torch_empty_penalty,
            torch_zeros_reward,
            language_reward,
            parser.get_format_reward_func(),
        ],
        weights=[
            0.1,   # think
            0.1,   # one_code_blob
            0.1,   # imports_decorator
            0.1,   # constexpr
            0.1,   # valid_tl_methods
            0.1,   # masks_load_store
            0.1,   # torch_empty_penalty
            0.1,   # torch_zeros
            0.1,   # language
            0.2,   # format
        ],
        parser=parser
    )
@weave.op
def call_triton_server(code, tests, client, url=SERVER_URL) -> Dict[str, Any]:
    triton_endpoint = f"{url}{RUN_TRITON_ENDPOINT}"
    resp = client.post(triton_endpoint,
                      json={
                          "code": code, 
                          "tests": tests,
                          "benchmark": True,
                          "benchmark_runs": BENCHMARK_RUNS
                      },
                      timeout=300.0)
    resp.raise_for_status()
    data = resp.json()
    
    # Convert to triton_-prefixed format
    result = {}
    for key, value in data.items():
        result[f"triton_{key}"] = value
    return result

# Async API rubric for expensive Triton server calls
class TritonAPIRubric(Rubric):
    """Rubric that makes async calls to Triton server for execution scoring."""
    
    def __init__(self, parser: XMLParser, triton_server_url: str, **kwargs):
        super().__init__(parser=parser, **kwargs)
        self.add_reward_func(self.triton_execution_reward)
        self.triton_server_url = triton_server_url

    @weave.op
    def triton_execution_reward(self, completion, answer=None, info=None, **kwargs) -> float:
        """Async reward function for code execution and performance."""
        # Handle both string and list of messages format
        if isinstance(completion, list):
            completion = completion[-1]['content'] if completion else ""
        
        # Use extract_code to properly extract code from output
        code = extract_code(completion)
        
        # Check if code is valid
        if not code or len(code) < 10:
            return -0.2
        
        # Get test info from dataset
        if info is None:
            info = {}
        
        # Access values directly from info (mapped in train.py)
        tests = info.get("tests", "")
        expected_stdout = info.get("pt_stdout", "")
        entrypoint = info.get("entrypoint", "")
        pytorch_baseline_time_ms = info.get("benchmark_mean_time_ms")
        pytorch_baseline_memory_mb = info.get("benchmark_memory_peak_mb")
        
        # Validate kernel if entrypoint is provided
        if entrypoint:
            analysis = is_valid_kernel(code, entrypoint)
            if not analysis.get("is_valid", False):
                return -0.2
        
        # Execute code on server synchronously
        with httpx.Client() as client:
            try:
                result = call_triton_server(code, tests, client, self.triton_server_url)
            except Exception:
                # Server error
                return -0.2
        
        # Check execution results
        runs = result.get("triton_status_code", -1) == 0
        
        # Base reward
        if not runs:
            return -0.2

        correct = compare_outputs(expected_stdout, result.get("triton_stdout", ""))["match"] and runs

        if not correct:
            # runs but not correct
            return 0.0
        else:
            # it runs and it's correct!
            base_reward = 1.0
        
        # Performance rewards (only if correct and runs)
        performance_reward = 0.0
        memory_reward = 0.0
        
        if correct and runs and pytorch_baseline_time_ms is not None:
            triton_time_ms = result.get("triton_benchmark_mean_time_ms")
            triton_memory_mb = result.get("triton_benchmark_memory_peak_mb")
            triton_successful_runs = result.get("triton_benchmark_successful_runs", 0)
            
            # Performance scoring
            if (triton_time_ms is not None and 
                triton_time_ms > 0 and 
                pytorch_baseline_time_ms > 0 and
                triton_successful_runs > 0):
                
                speedup = pytorch_baseline_time_ms / triton_time_ms
                if speedup > 1.0:
                    performance_reward = math.log(speedup)
            else:
                performance_reward = -0.1
            
            # Memory efficiency reward
            if (triton_memory_mb is not None and 
                pytorch_baseline_memory_mb is not None and 
                triton_memory_mb > 0 and 
                pytorch_baseline_memory_mb > 0):
                
                memory_improvement_ratio = pytorch_baseline_memory_mb / triton_memory_mb
                if memory_improvement_ratio > 1.0:
                    memory_reward = 0.2
        
        return base_reward + performance_reward + memory_reward

def get_triton_env(dataset, triton_server_url) -> SingleTurnEnv:
    if triton_server_url is None:
        triton_server_url = SERVER_URL
    parser = XMLParser(['think', 'triton'], answer_field='triton')
    static_rubric = create_static_rubric(parser)
    api_rubric = TritonAPIRubric(parser, triton_server_url)
    group = RubricGroup(rubrics=[api_rubric, static_rubric])
    return SingleTurnEnv(dataset=dataset, rubric=group)