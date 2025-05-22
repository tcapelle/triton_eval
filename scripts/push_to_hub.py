import os
from dataclasses import dataclass
import simple_parsing as sp
from huggingface_hub import upload_folder, HfApi

import wandb

api = HfApi(token=os.environ["HF_API_TOKEN"])

@dataclass
class ScriptArgs:
    artifact: str = "grpo-cuda/axolotl-sft/model-sft-qwen3-14b-v2-instruct:latest"
    model_name: str = "tcapelle/axolotl-sft-qwen3-14b-instruct"


args: ScriptArgs = sp.parse(ScriptArgs)

run = wandb.init()
artifact = run.use_artifact(args.artifact, type="model")
artifact_dir = artifact.download()

print(f"Downloaded artifact to {artifact_dir}")
# api.create_repo(args.model_name, repo_type="model")

upload_folder(
    repo_id=args.model_name,
    folder_path=artifact_dir,
    commit_message="Uploaded from W&B",
    repo_type="model",
)
