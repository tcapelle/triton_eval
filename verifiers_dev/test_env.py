from triton_rewards_modular import TritonEnvWithExecution
from verifiers.envs import SingleTurnEnv

from datasets import load_dataset
import openai

import weave

weave.init("grpo-cuda/verifiers-dev")

openai_client = openai.OpenAI(
    base_url=f"http://127.0.0.1:8000/v1",
    api_key="NoKey",
)

train_dataset = load_dataset("tcapelle/boostrap_oai_pt_think", split="train")

triton_env = TritonEnvWithExecution(dataset=train_dataset)

# triton_env = SingleTurnEnv(dataset=train_dataset, message_type="chat")

# Get first example from IterableDataset
first_example = next(iter(train_dataset))
prompt = first_example["prompt"][:-1]

print(prompt)

# Single rollout
completion, state = triton_env.rollout(
    client=openai_client,
    model="Qwen/Qwen3-4B",
    prompt=prompt,
    answer=None,
    sampling_args={
        "temperature": 0.6,
        "max_tokens": 8000,
    },
)

print(completion)
print("\n===============")
print(state)

# Score rollouts - Convert IterableDataset to proper format
# Get first 10 examples from the dataset
sample_data = []
dataset_iter = iter(train_dataset)
for i in range(10):
    try:
        sample_data.append(next(dataset_iter))
    except StopIteration:
        break

inputs = {
    "prompt": [item["prompt"][:-1] for item in sample_data],
    "answer": [item.get("answer") for item in sample_data],  # Use get() in case answer doesn't exist
    "task": ["triton_perf"] * len(sample_data)
}

results = triton_env.generate(
    client=openai_client,
    model="Qwen/Qwen3-4B",
    inputs=inputs,
    sampling_args={
        "temperature": 0.6,
        "max_tokens": 8000,
    },
)

print(results)