import datasets
import transformers
from transformers import set_seed, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from verifiers.envs import SingleTurnEnv
from verifiers.trainers import GRPOTrainer, grpo_defaults, GRPOConfig
from verifiers.parsers import XMLParser
from verifiers.rubrics import Rubric
from verifiers.utils.model_utils import get_model_and_tokenizer
from trl import ModelConfig, TrlParser

import wandb
import weave

import torch.distributed as dist

from config import GRPOScriptArgs
from triton_rewards_modular import get_triton_env, get_multi_turn_env

def is_main_process():
    return dist.is_initialized() and dist.get_rank() == 0

def train(script_args: GRPOScriptArgs, training_args: GRPOConfig, model_args: ModelConfig):
    set_seed(training_args.seed)

    training_args.output_dir = f"{training_args.output_dir}/{script_args.wandb_name}"

    if is_main_process():
        print(f"Script parameters:\n{script_args}\n--------------------------------")
        print(f"Training parameters:\n{training_args}\n--------------------------------")
        print(f"Model parameters:\n{model_args}\n--------------------------------")

        wandb.init(entity=script_args.wandb_entity, project=script_args.wandb_project, name=script_args.wandb_name)
        weave.init(f"{script_args.wandb_entity}/{script_args.wandb_project}")

    # Load model and tokenizer

    model, tokenizer = get_model_and_tokenizer(model_args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token


    train_dataset = load_dataset(script_args.dataset_name, split=script_args.split)
    if script_args.eval_dataset_name is not None:
        eval_dataset = load_dataset(script_args.eval_dataset_name, split=script_args.eval_split)

    def remove_last_assistant_message(row, column_name=script_args.field_messages):
        if row[column_name][-1]["role"] == "assistant":
            row[column_name] = row[column_name][:-1]
        return row

    train_dataset = train_dataset.map(remove_last_assistant_message)
    train_dataset = train_dataset.map(lambda row: {"answer": row.get("triton_code", "")})
    train_dataset = train_dataset.filter(lambda row: row.get("tests", "") != "")

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
    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(map_to_info)
    if script_args.multi_turn:
        triton_env = get_multi_turn_env(train_dataset=train_dataset, triton_server_url=script_args.triton_server_url, eval_dataset=eval_dataset)
    else:
        triton_env = get_triton_env(train_dataset=train_dataset, triton_server_url=script_args.triton_server_url, eval_dataset=eval_dataset)

    if is_main_process:
        print(f"Loading Triton environment with server url {script_args.triton_server_url}")

    # Create trainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        env=triton_env,
        args=training_args,
    )

    # Train
    trainer.train()

    # Save final model
    trainer.save_model(training_args.output_dir)
    if trainer.accelerator.is_main_process:
        print(f"Model saved to {training_args.output_dir}")
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArgs, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    train(script_args, training_args, model_args)