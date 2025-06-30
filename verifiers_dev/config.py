from dataclasses import dataclass

@dataclass
class GRPOScriptArgs:
    dataset_name: str
    split: str = "train"
    eval_dataset_name: str | None = None
    eval_split: str = "train"
    field_messages: str = "prompt"

    wandb_entity: str = "grpo-cuda"
    wandb_project: str = "verifiers"
    wandb_name: str = "qwen-4b"

    triton_server_url: str = "http://127.0.0.1:9347"