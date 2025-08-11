import re
import math
import time
import random
import httpx
import asyncio
import weave
from typing import Dict, Any, List
from textwrap import dedent
from verifiers.parsers.xml_parser import XMLParser
from verifiers.envs.singleturn_env import SingleTurnEnv
from verifiers.envs.multiturn_env import MultiTurnEnv
from verifiers.rubrics.rubric_group import Rubric, RubricGroup
from verifiers.types import Messages, State, Info, RolloutScores

# Import kernel validation tools
from triton_eval.utils import compare_outputs
from triton_eval.agents.tools import extract_code
from triton_eval.kernel_checks import is_valid_kernel

# Import the base RubricGroup
from verifiers.rubrics.rubric_group import RubricGroup

class FixedRubricGroup(RubricGroup):
    """Fixed RubricGroup that properly aggregates reward scores."""
    
    async def score_rollouts(
        self,
        prompts: List[Messages],
        completions: List[Messages],
        answers: List[str],
        states: List[State],
        tasks: List[str],
        infos: List[Info] = [],
        **kwargs,
    ) -> RolloutScores:
        """
        Run all rubrics sequentially and return the aggregated scores.
        
        Reward functions with the same name are summed up.
        Rewards are also properly aggregated.
        """
        all_scores = RolloutScores(
            reward=[0.0] * len(prompts),  # Initialize with zeros for each prompt
            metrics={},
        )
        
        for rubric in self.rubrics:
            rubric_scores = await rubric.score_rollouts(
                prompts, completions, answers, states, tasks, infos, **kwargs
            )
            
            # Aggregate rewards
            if rubric_scores.reward:
                all_scores.reward = [
                    a + b for a, b in zip(all_scores.reward, rubric_scores.reward)
                ]
            
            # Aggregate metrics
            for key, value in rubric_scores.metrics.items():
                if key in all_scores.metrics:
                    # element-wise sum
                    all_scores.metrics[key] = [
                        a + b for a, b in zip(all_scores.metrics[key], value)
                    ]
                else:
                    all_scores.metrics[key] = value
        print(f"    #### all_scores:\n {all_scores}\n==================\n")
        return all_scores


# Server configuration
TRITON_SERVER_URL = "http://127.0.0.1:9347"
TRITON_RUN_ENDPOINT = f"/run_triton"
TRITON_BENCHMARK = True
TRITON_BENCHMARK_RUNS = 10


class TritonClient:
    """HTTP client for the Triton execution server with error handling and retry logic."""
    
    def __init__(self, server_url: str = TRITON_SERVER_URL, run_triton_endpoint: str = TRITON_RUN_ENDPOINT, max_concurrent: int = 8):
        self.server_url = server_url
        self.run_triton_endpoint = run_triton_endpoint
        self.client: httpx.Client = self._create_client()
        self.async_client: httpx.AsyncClient = self._create_async_client()
        # Limit concurrent requests to avoid overloading the server / client pool
        self._semaphore = asyncio.Semaphore(max_concurrent)
    
    def _create_client(self) -> httpx.Client:
        """Create a new HTTP client with proper configuration."""
        timeout_config = httpx.Timeout(connect=30.0, read=300.0, write=30.0, pool=30.0)
        limits = httpx.Limits(max_keepalive_connections=10, max_connections=20)
        return httpx.Client(timeout=timeout_config, limits=limits)

    def _create_async_client(self) -> httpx.AsyncClient:
        """Create a new Async HTTP client with proper configuration."""
        timeout_config = httpx.Timeout(connect=30.0, read=300.0, write=30.0, pool=30.0)
        limits = httpx.Limits(max_keepalive_connections=10, max_connections=20)
        return httpx.AsyncClient(timeout=timeout_config, limits=limits)
    
    async def run_code_async(self, code: str, tests: str, benchmark: bool = TRITON_BENCHMARK, benchmark_runs: int = TRITON_BENCHMARK_RUNS) -> Dict[str, Any]:
        """Execute code on the Triton server asynchronously with retry logic and error handling."""
        
        triton_endpoint = f"{self.server_url}{self.run_triton_endpoint}"
        
        # Default error response
        error_response = {
            "triton_status_code": -1,
            "triton_stdout": "",
            "triton_stderr": "Connection failed",
            "triton_gpu_mem_used_gb": None,
            "triton_cpu_percent": None,
            "triton_ram_percent": None,
            "triton_benchmark_mean_time_ms": None,
            "triton_benchmark_std_time_ms": None,
            "triton_benchmark_memory_peak_mb": None,
            "triton_benchmark_successful_runs": 0,
        }
        
        # Use semaphore to limit concurrent requests
        async with self._semaphore:
            # Retry logic with exponential backoff
            max_retries = 3
            base_delay = 1.0
            last_error = None
            
            for attempt in range(max_retries):
                try:
                    resp = await self.async_client.post(triton_endpoint,
                                          json={
                                              "code": code, 
                                              "tests": tests,
                                              "benchmark": benchmark,
                                              "benchmark_runs": benchmark_runs
                                          })
                    resp.raise_for_status()
                    data = resp.json()
                    
                    # Convert to triton_-prefixed format
                    result = {}
                    for key, value in data.items():
                        result[f"triton_{key}"] = value
                    return result
                    
                except (httpx.ConnectTimeout, httpx.ConnectError) as e:
                    last_error = e
                    if attempt == max_retries - 1:  # Last attempt
                        error_response["triton_stderr"] = f"Connection failed after {max_retries} attempts: {last_error}"
                        return error_response
                    
                    # Exponential backoff with jitter
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    print(f"Connection failed (attempt {attempt + 1}/{max_retries}), retrying in {delay:.2f}s: {e}")
                    await asyncio.sleep(delay)
                    
                except httpx.HTTPStatusError as e:
                    # HTTP error (4xx, 5xx) - don't retry, return error info
                    error_response["triton_stderr"] = f"HTTP {e.response.status_code}: {e.response.text}"
                    return error_response
                    
                except Exception as e:
                    # Other unexpected errors - don't retry
                    error_response["triton_stderr"] = f"Unexpected error: {e}"
                    return error_response
            
            # This should never be reached due to the exception handling above
            error_response["triton_stderr"] = "Unexpected code path reached"
            return error_response
    
    def run_code(self, code: str, tests: str, benchmark: bool = TRITON_BENCHMARK, benchmark_runs: int = TRITON_BENCHMARK_RUNS) -> Dict[str, Any]:
        """Execute code on the Triton server with retry logic and error handling."""
        
        triton_endpoint = f"{self.server_url}{self.run_triton_endpoint}"
        
        # Default error response
        error_response = {
            "triton_status_code": -1,
            "triton_stdout": "",
            "triton_stderr": "Connection failed",
            "triton_gpu_mem_used_gb": None,
            "triton_cpu_percent": None,
            "triton_ram_percent": None,
            "triton_benchmark_mean_time_ms": None,
            "triton_benchmark_std_time_ms": None,
            "triton_benchmark_memory_peak_mb": None,
            "triton_benchmark_successful_runs": 0,
        }
        
        # Use the initialized client
        
        # Retry logic with exponential backoff
        max_retries = 3
        base_delay = 1.0
        last_error = None
        
        for attempt in range(max_retries):
            try:
                resp = self.client.post(triton_endpoint,
                                      json={
                                          "code": code, 
                                          "tests": tests,
                                          "benchmark": benchmark,
                                          "benchmark_runs": benchmark_runs
                                      })
                resp.raise_for_status()
                data = resp.json()
                
                # Convert to triton_-prefixed format
                result = {}
                for key, value in data.items():
                    result[f"triton_{key}"] = value
                return result
                
            except (httpx.ConnectTimeout, httpx.ConnectError) as e:
                last_error = e
                if attempt == max_retries - 1:  # Last attempt
                    error_response["triton_stderr"] = f"Connection failed after {max_retries} attempts: {last_error}"
                    return error_response
                
                # Exponential backoff with jitter
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"Connection failed (attempt {attempt + 1}/{max_retries}), retrying in {delay:.2f}s: {e}")
                time.sleep(delay)
                
            except httpx.HTTPStatusError as e:
                # HTTP error (4xx, 5xx) - don't retry, return error info
                error_response["triton_stderr"] = f"HTTP {e.response.status_code}: {e.response.text}"
                return error_response
                
            except Exception as e:
                # Other unexpected errors - don't retry
                error_response["triton_stderr"] = f"Unexpected error: {e}"
                return error_response
        
        # This should never be reached due to the exception handling above
        error_response["triton_stderr"] = "Unexpected code path reached"
        return error_response

    
    async def close_async(self):
        """Close the async HTTP client."""
        if hasattr(self, 'async_client') and self.async_client is not None:
            await self.async_client.aclose()
    
    def close(self):
        """Close the HTTP client."""
        if hasattr(self, 'client') and self.client is not None:
            self.client.close()
        # For async client, we need to run it in an event loop
        if hasattr(self, 'async_client') and self.async_client is not None:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If we're in an async context, schedule the close
                    asyncio.create_task(self.async_client.aclose())
                else:
                    # If no loop is running, create one to close the client
                    asyncio.run(self.async_client.aclose())
            except RuntimeError:
                # If we can't get a loop, just leave it (will be cleaned up on exit)
                pass

    def __del__(self):
        """Cleanup on destruction."""
        self.close()


# Module-level client instance
triton_client = TritonClient()

# Register cleanup function
import atexit
atexit.register(triton_client.close)

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
            thinking_length = max(content_length - 10000, 0)
            return 0.4 * math.exp(-0.5*thinking_length/1000) - 0.2
    
    return -0.2


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
            code_length = max(code_length - 6000, 0)
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


# Create static rubric with all the cheap rewards
def create_static_rubric(parser: XMLParser) -> Rubric:
    """Create rubric with static (cheap) reward functions."""
    
    # Create a wrapper for format_reward_func that handles both string and list inputs
    original_format_reward_func = parser.get_format_reward_func()
    
    @weave.op
    def format_reward_func_wrapper(completion, **kwargs):
        """Wrapper that handles both string and list completion formats."""
        # Convert string to list format if needed
        if isinstance(completion, str):
            # Create a minimal message list with the completion as content
            completion = [{"role": "assistant", "content": completion}]
        elif isinstance(completion, list) and completion and isinstance(completion[-1], dict) and 'content' in completion[-1]:
            # Already in the correct format
            pass
        else:
            # Handle edge cases
            return 0.0
        
        # Filter out kwargs that the original function doesn't accept
        # The format_reward_func only accepts completion as a positional argument
        return original_format_reward_func(completion)
    
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
            format_reward_func_wrapper,  # Use the wrapper instead
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
            0.2,   # format
        ],
        parser=parser,
        parallelize_scoring=False  # Disable parallel scoring to avoid async issues
    )

# Async API rubric for expensive Triton server calls
class TritonAPIRubric(Rubric):
    """Rubric that makes async calls to Triton server for execution scoring."""
    
    def __init__(
        self, 
        parser: XMLParser, 
        triton_client: TritonClient, 
        triton_benchmark: bool=TRITON_BENCHMARK,
        triton_benchmark_runs: int=TRITON_BENCHMARK_RUNS,
        **kwargs):
        super().__init__(parser=parser, **kwargs)
        self.add_reward_func(self.triton_execution_reward)
        self.triton_client = triton_client
        self.triton_benchmark = triton_benchmark
        self.triton_benchmark_runs = triton_benchmark_runs
        # Add a counter for debugging
        self._call_counter = 0
    
    @weave.op
    def triton_execution_reward(self, completion, answer=None, info=None, **kwargs) -> float:
        """Reward function for code execution and performance."""
        # Handle both string and list of messages format
        if isinstance(completion, list):
            completion = completion[-1]['content'] if completion else ""
        
        # Use extract_code to properly extract code from output
        code = extract_code(completion)
        print(f"    triton_execution_reward -> code:\n {code:100}\n==================\n")
        
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
        # The sync client handles retries and thread safety
        result = self.triton_client.run_code(
            code,
            tests,
            benchmark=self.triton_benchmark,
            benchmark_runs=self.triton_benchmark_runs,
        )
        
        # Check if there was an error
        if result.get("triton_status_code", -1) != 0:
            error_msg = result.get("triton_stderr", "Unknown error")
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

def load_environment(
    train_dataset,
    eval_dataset=None,
    triton_server_url: str = TRITON_SERVER_URL,
    triton_run_endpoint: str = TRITON_RUN_ENDPOINT,
    triton_benchmark: bool = TRITON_BENCHMARK,
    triton_benchmark_runs: int = TRITON_BENCHMARK_RUNS,
    ):
    triton_client = TritonClient(server_url=triton_server_url, run_triton_endpoint=triton_run_endpoint)
    parser = XMLParser(['think', 'triton'], answer_field='triton')
    static_rubric = create_static_rubric(parser)
    api_rubric = TritonAPIRubric(
        parser,
        triton_client,
        triton_benchmark=triton_benchmark,
        triton_benchmark_runs=triton_benchmark_runs,
        parallelize_scoring=False,
    )
    group = FixedRubricGroup(rubrics=[api_rubric, static_rubric])
    return SingleTurnEnv(dataset=train_dataset, rubric=group, eval_dataset=eval_dataset)

@weave.op
def remove_thinking_block(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove the first thinking block from the messages."""
    for message in messages:
        if message['role'] == 'assistant':
            if "</think>" in message['content']:
                content = message['content'].split("</think>")[1].strip()
            else:
                content = message['content']
            
            # empty thinking block
            message['content'] = "<think>\n\n</think>\n\n" + content
    return messages


class MultiTurnTritonEnv(MultiTurnEnv):
    def __init__(self, triton_client: TritonClient, max_turns: int = 10, debug_dummy_env: bool = False, **kwargs):
        super().__init__(message_type="chat", max_turns=max_turns, **kwargs)
        self.triton_client = triton_client
        self.debug_dummy_env = debug_dummy_env
    
    @weave.op
    def run_code(self, code: str, info: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        """Run the code and return the error message and the comparison results."""

        tests = info.get("tests", "")
        expected_stdout = info.get("pt_stdout", "")
        entrypoint = info.get("entrypoint", "")

        # Execute code on server asynchronously to avoid blocking the event loop
        result = self.triton_client.run_code(code, tests, benchmark=False)
        
        # Check if there was an error
        if result.get("triton_status_code", -1) != 0:
            error_msg = result.get("triton_stderr", "Unknown error")
            return {"runs": False, "error": error_msg, "comparison": None}
        
        # Check execution results
        runs = result.get("triton_status_code", -1) == 0
        
        if runs:
            comparison = compare_outputs(expected_stdout, result.get("triton_stdout", ""))
            return {"runs": True, "error": None, "comparison": comparison}
        
        else: # it does not run
            triton_execution_error = result.get("triton_stderr", None)
            if triton_execution_error is None:
                # Provide a generic error message to avoid downstream None handling
                triton_execution_error = "Unknown Triton execution error"
            return {"runs": False, "error": triton_execution_error, "comparison": None}

    @weave.op
    def setup_state(self, state: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        state.setdefault("runs", False)
        state.setdefault("error", None)
        state.setdefault("comparison", None)
        return state

    @weave.op
    def is_completed(self, messages: List[Dict[str, Any]], state: Dict[str, Any], **kwargs: Any) -> bool:
        try:
            return bool(state.get("runs")) and bool(state.get("comparison", {}).get("match"))
        except Exception:
            return False

    @weave.op
    def env_response(self,
                     messages: List[Dict[str, Any]],
                     state: Dict[str, Any],
                     **kwargs: Any) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        info: Dict[str, Any] = state.get("info", {})

        fail_prompt = dedent("""\
        The code you wrote doesn't run. The error when running the code is: 
        
        <error_message>
        {error}
        </error_message>
        Consider the code you wrote and the error message and write a working triton code.
        """)

        triton_prompt = dedent("""The exeuction server errored out, try re-running the code again.""")

        incorrect_prompt = dedent("""\
        The code runs but it is not correct. You are failing the tests:
        <tests>
        {tests}
        </tests>
        And here are the results of the tests execution:
        <comparison_results>
        {comparison_results}
        </comparison_results>
        Consider the code you wrote, the tests and the comparison results. Fix the code so it passes the tests.
        """)

        completion = messages[-1]['content']
        
        # Use extract_code to properly extract code from output
        code = extract_code(completion)
        
        # Debug mode: simulate env behavior without calling Triton server
        if getattr(self, "debug_dummy_env", False):
            env_messages: List[Dict[str, Any]] = []
            if not code or len(code.strip()) < 10:
                # Simulate a run failure with an explanatory error
                state.update({"runs": False, "error": "Dummy: no or too-short code", "comparison": None})
                env_messages.append({"role": "user", "content": fail_prompt.format(error="Dummy: no or too-short code")})
                return env_messages, state
            # Simulate success (runs and correct)
            state.update({"runs": True, "error": None, "comparison": {"match": True, "results": "dummy"}})
            return [], state
        run_result = self.run_code(code, info=info, **kwargs)
        state.update(run_result)

        # Decide next user message(s) based on run result
        env_messages: List[Dict[str, Any]] = []
        error = state.get("error")
        if error is not None:
            if "Triton Server Error" in str(error):
                env_messages.append({"role": "user", "content": triton_prompt})
            elif not state.get("runs"):
                env_messages.append({"role": "user", "content": fail_prompt.format(error=error)})
        elif state.get("runs"):
            tests = info.get("tests", "")
            comparison_results = str(state.get("comparison", {}).get("results"))
            if not state.get("comparison", {}).get("match"):
                env_messages.append({
                    "role": "user",
                    "content": incorrect_prompt.format(tests=tests, comparison_results=comparison_results),
                })

        # If code ran and is correct, return no additional env messages
        return env_messages, state

    # Use MultiTurnEnv.rollout from the base class


def load_multi_turn_environment(
    train_dataset,
    eval_dataset=None,
    triton_server_url: str = TRITON_SERVER_URL,
    triton_run_endpoint: str = TRITON_RUN_ENDPOINT,
    triton_benchmark: bool = TRITON_BENCHMARK,
    triton_benchmark_runs: int = TRITON_BENCHMARK_RUNS,
    max_turns: int = 3,
    debug_dummy_env: bool = False,
    ) -> MultiTurnTritonEnv:
    """Create a multi-turn Triton environment."""
    triton_client = TritonClient(server_url=triton_server_url, run_triton_endpoint=triton_run_endpoint)
    parser = XMLParser(['think', 'triton'], answer_field='triton')
    static_rubric = create_static_rubric(parser)
    api_rubric = TritonAPIRubric(
        parser,
        triton_client,
        triton_benchmark=triton_benchmark,
        triton_benchmark_runs=triton_benchmark_runs,
    )
    group = FixedRubricGroup(rubrics=[api_rubric, static_rubric], parallelize_scoring=False)
    return MultiTurnTritonEnv(
        dataset=train_dataset, 
        triton_client=triton_client, 
        max_turns=max_turns,
        debug_dummy_env=debug_dummy_env,
        rubric=group, 
        eval_dataset=eval_dataset)