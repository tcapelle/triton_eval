from dataclasses import dataclass

@dataclass
class GRPOScriptArgs:
    dataset_name: str
    split: str = "train"
    eval_dataset_name: str | None = None
    eval_split: str = "train"
    field_messages: str = "prompt"
    
    # environment stuff
    max_turns: int = 3
    triton_server_url: str = "http://127.0.0.1:9347"
    triton_run_endpoint: str = "/run_triton"
    triton_benchmark: bool = True
    triton_benchmark_runs: int = 10

    # wandb stuff
    wandb_entity: str = "grpo-cuda"
    wandb_project: str = "verifiers"
    wandb_name: str = "qwen-4b"