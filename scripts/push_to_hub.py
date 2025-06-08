import os
from dataclasses import dataclass
import simple_parsing as sp
from huggingface_hub import upload_folder, HfApi

import wandb


@dataclass
class ScriptArgs:
    artifact: str = "grpo-cuda/axolotl-sft/model-sft-qwen3-14b-v2-instruct:latest"
    model_name: str = "tcapelle/axolotl-sft-qwen3-14b-instruct"
    folder: str = None

args: ScriptArgs = sp.parse(ScriptArgs)

if args.folder:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(args.folder, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.folder)
    model.save_pretrained(args.folder)
    tokenizer.save_pretrained(args.folder)
    model.push_to_hub(args.model_name)
    tokenizer.push_to_hub(args.model_name)
else:
    api = HfApi(token=os.environ["HF_API_TOKEN"])

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
