from datasets import load_dataset

dataset = load_dataset("tcapelle/annotated_dataset_o3_sample", split="train")

# select columns final_triton_code", "final_pytorch_code"
dataset = dataset.select_columns(["final_triton_code", "final_pytorch_code"])

# save to json
dataset.push_to_hub("tcapelle/annotated_dataset_o3_train_pytorch_triton")

