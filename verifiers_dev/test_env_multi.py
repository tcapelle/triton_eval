from triton_rewards_modular import get_multi_turn_env

from rich.pretty import pprint
from datasets import load_dataset
import openai

import weave



model_name = "qwen-32b-vf_ep2-v2/checkpoint-500"
dataset_name = "tcapelle/boostrap_oai_pt_think_ep2_v2"

vllm_url = "http://cw-verifiers-vllm-service:8000/v1"
triton_url = "http://cw-verifiers-rewards-service-grpo:9347"




weave.init("grpo-cuda/verifiers-dev")

openai_client = openai.OpenAI(
    base_url=vllm_url,
    api_key="NoKey",
)

train_dataset = load_dataset(dataset_name, split="train")

def remove_last_assistant_message(row, column_name="prompt"):
    if row[column_name][-1]["role"] == "assistant":
        row[column_name] = row[column_name][:-1]
    return row

train_dataset = train_dataset.map(remove_last_assistant_message)
train_dataset = train_dataset.map(lambda row: {"answer": row.get("triton_code", "")})

# Map dataset columns into info key for triton_execution_reward
def map_to_info(row):
    info = {
        "tests": row.get("tests", ""),
        "pt_stdout": row.get("pt_stdout", ""),
        "entrypoint": row.get("entrypoint", ""),
        "benchmark_mean_time_ms": row.get("benchmark_mean_time_ms"),
        "benchmark_memory_peak_mb": row.get("benchmark_memory_peak_mb")
    }
    return {"info": info}

train_dataset = train_dataset.map(map_to_info)

triton_env = get_multi_turn_env(train_dataset, triton_server_url=triton_url)

# triton_env = SingleTurnEnv(dataset=train_dataset, message_type="chat")

# Get first example from IterableDataset
for sample in train_dataset:
    if sample["info"]["tests"] != "":

        # Single rollout
        completion, state = triton_env.rollout(
            client=openai_client,
            model=model_name,
            prompt=sample["prompt"],
            answer=sample["answer"],
            info=sample["info"],
            sampling_args={
                "temperature": 0.6,
                "max_tokens": 8000,
            },
        )

# # Score rollouts - Convert IterableDataset to proper format
# # Get first 10 examples from the dataset
# sample_data = []
# dataset_iter = 
# for i, sample in enumerate(train_dataset)):
#     try:
#         sample_data.append(sample)
#     except StopIteration:
#         break

# # inputs = {
# #     "prompt": [item["prompt"] for item in sample_data],
# #     "answer": [item.get("answer") for item in sample_data],  # Use get() in case answer doesn't exist
# #     "task": ["triton_perf"] * len(sample_data)
# # }

# results = triton_env.generate(
#     client=openai_client,
#     model=model_name,
#     inputs=sample_data,
#     sampling_args={
#         "temperature": 0.6,
#         "max_tokens": 8000,
#     },
# )

# print(results)