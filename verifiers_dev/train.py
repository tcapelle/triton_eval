# CUDA_VISIBLE_DEVICES=0,1 vf-vllm --model 'Qwen/Qwen3-4B' --tensor-parallel-size 2

import os
# Set environment variables to help with NCCL issues
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_TIMEOUT"] = "3600"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from verifiers.envs import SingleTurnEnv
from verifiers.trainers import GRPOTrainer, grpo_defaults
from verifiers.parsers import XMLParser
from verifiers.rubrics import Rubric
from verifiers.utils.model_utils import get_model_and_tokenizer

import wandb
import weave

from accelerate import Accelerator

from triton_rewards_modular import get_triton_env

accelerator = Accelerator()

if accelerator.is_main_process:
    wandb.init(entity="grpo-cuda", project="verifiers")
    weave.init("grpo-cuda/verifiers")

# Load model and tokenizer

model, tokenizer = get_model_and_tokenizer("/model-checkpoints/sft-qwen3-4b-boot")
tokenizer.pad_token = tokenizer.eos_token


train_dataset = load_dataset("tcapelle/boostrap_oai_pt_think", split="train")
train_dataset = train_dataset.map(lambda row: {"prompt": row["prompt"][:-1]}) # drop last assistant message

# Map dataset columns into info key for triton_execution_reward
def map_to_info(row):
    info = {
        "tests": row.get("tests", ""),
        "stdout": row.get("stdout", ""),
        "entrypoint": row.get("entrypoint", ""),
        "benchmark_mean_time_ms": row.get("benchmark_mean_time_ms"),
        "benchmark_memory_peak_mb": row.get("benchmark_memory_peak_mb")
    }
    return {"info": info}

train_dataset = train_dataset.map(map_to_info)

triton_env = get_triton_env(dataset=train_dataset)

# Training configuration
training_args=grpo_defaults(
    run_name="qwen-4b",
    per_device_train_batch_size = 4,
    gradient_accumulation_steps = 2,
    num_generations = 8,
    num_train_epochs = 1,
    max_prompt_length = 12000,  # Remove length limit
    max_completion_length = 12000,
    beta = 0.0,
    temperature = 0.6,
)
# Create trainer
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=triton_env,
    args=training_args,
    eval_dataset=None
)

# Train
trainer.train()

# Save final model
trainer.save_model("/model-checkpoints/qwen3-4b0-verifiers")