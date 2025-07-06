# Length Parameter Analysis for Multi-Turn Training

## Executive Summary

Your multi-turn training setup has several length-related parameters that interact in complex ways, potentially causing issues as conversations grow longer. This analysis examines how vLLM server parameters, training configuration, and multi-turn environment behavior work together.

## Current Length Parameter Hierarchy

### 1. vLLM Server Parameters (`vllm_server.py`)

#### Primary Server-Wide Limits
- **`max_model_len`**: Default 8192, configurable via `ScriptArguments`
  - Server-wide sequence length limit for the model
  - Used for KV cache allocation and memory planning
  - Cannot be exceeded without server restart

#### Generation Parameters  
- **`token_chunk_size`**: Default 64 tokens
  - Tokens generated per iteration in chunked batching
  - Allows for dynamic batching and interruption
  - Affects memory usage and latency

#### Dynamic Truncation
- **`truncate_prompt_tokens`**: `max_model_len - (chunk_size//2)`
  - Applied in `SamplingParams` for each generation
  - Example: 8192 - (64//2) = 8160 tokens max prompt
  - Truncates from the beginning of prompt if exceeded

#### Pre-Generation Filtering
- **`max_prompt_tokens`**: `max_model_len - chunk_size`
  - Pre-check threshold: 8192 - 64 = 8128 tokens
  - Requests exceeding this are rejected with error
  - Applied before sending to vLLM engine

### 2. Training Configuration Parameters

#### Trainer-Side Limits (`GRPOConfig`)
- **`max_prompt_length`**: 8000 (in your config)
  - Training-side prompt truncation during data processing
  - Applied when tokenizing datasets
  - Independent from vLLM server limits

- **`max_completion_length`**: 12000 (in your config)  
  - Training-side completion truncation
  - Applied to generated completions during training
  - Sum: 8000 + 12000 = 20000 > 8192 (server limit)

#### OpenAI API Parameters
- **`max_tokens`**: Default 1024 (per request)
  - Per-request generation limit from OpenAI API
  - Maps to `effective_max_tokens` in server
  - Can be overridden per request

### 3. Multi-Turn Environment Behavior

#### Message Accumulation (`MutiTurnTritonEnv`)
```python
# Each turn adds multiple messages:
messages.append({"role": "assistant", "content": response})      # Model response
messages.append(env_msg)  # Environment feedback (user role)

# Conversation grows as: [system, user, assistant, user, assistant, user, ...]
```

#### Turn Limits
- **`max_turns`**: Default 10 in `MultiTurnEnv`
- **`max_turns`**: 3 in `get_multi_turn_env()` (your specific config)
- No explicit length checking during turns

## Critical Issues Identified

### 1. **Length Parameter Mismatch** 

The training expects to handle much longer sequences than the server can process.

Training Config: max_prompt_length (8000) + max_completion_length (12000) = 20000 tokens
vLLM Server: max_model_len = 8192 tokens

### 2. **Multi-Turn Conversation Growth**
Each turn adds approximately:
- Assistant response: ~500-2000 tokens (estimated for Triton code)
- Environment feedback: ~100-500 tokens (error messages, test results)
- **Per turn growth**: ~600-2500 tokens

**Projection**:
- Turn 1: ~1000 tokens (initial prompt)
- Turn 2: ~2000 tokens
- Turn 3: ~4000 tokens  
- Turn 4: **~6500 tokens** → Approaching server limit
- Turn 5: **~9000 tokens** → Exceeds server limit

### 3. **Server Rejection Pattern**
```python
# vllm_server.py line 839-844
max_prompt_tokens = script_args.max_model_len - chunk_size
if token_count > max_prompt_tokens:
    logger.info(f"Request {req_state.request_id} prompt too long...")
    req_state.finish_reason = "length"
    req_state.error = ValueError(f"Prompt exceeds maximum length...")
```

**Result**: Later turns in multi-turn training will be systematically rejected.

### 4. **No Conversation Length Management**
The `MutiTurnTritonEnv` has no mechanism to:
- Track conversation length
- Truncate conversation history
- Implement sliding window
- Compress earlier turns

## Recommended Solutions

### 1. **Align Server and Training Limits**
```yaml
# In your config
max_model_len: 16000         # Increase server limit
max_prompt_length: 8000      # Keep current  
max_completion_length: 6000  # Reduce to fit: 8000 + 6000 = 14000 < 16000
```

### 2. **Implement Conversation Length Management**
Add to `MutiTurnTritonEnv`:
```python
def truncate_conversation(self, messages: List[Dict], max_tokens: int) -> List[Dict]:
    """Keep system prompt + recent turns within token limit."""
    # Always keep system message
    system_msg = messages[0] if messages[0]["role"] == "system" else None
    conversation = messages[1:] if system_msg else messages
    
    # Implement sliding window or importance-based truncation
    # Return truncated conversation
```

### 3. **Progressive Length Limits by Turn**
```python
def get_max_tokens_for_turn(self, turn: int, base_limit: int) -> int:
    """Reduce available tokens as conversation grows."""
    turn_penalty = turn * 500  # Reduce by 500 tokens per turn
    return max(1000, base_limit - turn_penalty)
```

### 4. **Server Configuration Updates**
```yaml
# vllm server config
max-model-len: 16000          # Increase from 8192
token-chunk-size: 256         # Increase for longer generations
batch-request-timeout-seconds: 600  # Longer timeout for complex turns
```

### 5. **Conversation Compression Strategy**
Implement conversation summarization:
```python
def compress_early_turns(self, messages: List[Dict], keep_recent: int = 2):
    """Summarize early turns to save space."""
    if len(messages) > keep_recent * 2 + 1:  # +1 for system
        # Summarize messages[1:-keep_recent*2] 
        # Keep system + recent turns
```

## Implementation Priority

### Phase 1: Immediate Fixes
1. ✅ Increase `max_model_len` to 16000
2. ✅ Adjust training length parameters to fit
3. ✅ Add conversation length monitoring

### Phase 2: Conversation Management  
1. ✅ Implement conversation truncation
2. ✅ Add sliding window for long conversations
3. ✅ Test with various turn counts

### Phase 3: Advanced Optimization
1. ✅ Conversation compression/summarization
2. ✅ Turn-specific length budgets
3. ✅ Performance tuning

## Code Changes Required

### 1. Update `MutiTurnTritonEnv.rollout()`
```python
# Add before each generation:
if self.get_conversation_length(messages) > self.max_conversation_tokens:
    messages = self.truncate_conversation(messages)
```

### 2. Update Server Configuration
```python
# verifiers/verifiers/inference/vllm_config.py
max_model_len: int = field(
    default=16000,  # Increase from 8192
    metadata={"help": "..."}
)
```

### 3. Update Training Config  
```yaml
# verifiers_dev/configs/config32b.yaml
max_prompt_length: 8000
max_completion_length: 6000  # Reduce from 12000
max-model-len: 16000         # Match server
```

## Testing Strategy

1. **Length Monitoring**: Add logging for conversation lengths
2. **Multi-Turn Testing**: Test 1, 3, 5, 10 turn scenarios
3. **Performance Impact**: Measure latency/memory with longer contexts
4. **Failure Modes**: Test behavior at length limits

This analysis shows that your multi-turn training is hitting fundamental length mismatches between components. The recommended changes should resolve these issues and enable effective multi-turn training.