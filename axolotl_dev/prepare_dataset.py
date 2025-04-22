from datasets import load_dataset
from dataclasses import dataclass
import simple_parsing as sp

from tools import extract_tests

@dataclass
class Args:
    dataset_name: str = "tcapelle/dataset_wiht_pt_errors"#"GPUMODE/Inductor_Created_Data_Permissive"
    code_column: str = "pytorch_code_fixed"


SYSTEM_PROMPT = """
You are an expert in Triton programming, capable of writing corresponding Triton kernels and wrapper functions based on functional descriptions and function parameters. 

# Instructions
- Ensure that the wrapper function fully corresponds to the provided function information.
- Generate a detailed plan on how to convert and optimize the Pytorch code to a Triton kernel before writing the code.
- The reasoning process MUST BE enclosed within <think> and </think> tags."
- Reply with the thinking process and a single blob of code surrounded with ```python and ```.
"""

USER_PROMPT = """Convert the following PyTorch code to a Triton kernel.
Pytorch code:
```python
{pytorch_code}```

The function should have the same name as the PyTorch function. 

Don't forget to format your answer as:
<think>
thinking process
</think>
```python
code
```"""

def split_at_tests(code: str, entrypoint: str) -> tuple[str, str]:
    test_name = f"test_{entrypoint}"
    pattern = f"def {test_name}"
    match_index = code.find(pattern)
    if match_index == -1:
        return code, ""
    code_without_tests = code[:match_index]
    tests = code[match_index:]
    return code_without_tests, tests

def get_dataset(dataset_name, split="train", code_column="pytorch_code"):
    # Load the dataset - this is expected to be preprocessed by prepare_dataset.py
    data = load_dataset(dataset_name)[split]
    print(f"Loaded {len(data)} examples from {dataset_name} split {split}")
    data = data.filter(lambda x: x["pytorch_code_fixed"] is not None)
    print(f"Filtered to {len(data)} examples with non-None pytorch_code_fixed")
    data = data.filter(lambda x: x["entrypoint"] is not None)
    print(f"Filtered to {len(data)} examples with non-None entrypoint")

    def format_example(example):
        # Format the prompt with the preprocessed code
        pytorch_code = example[code_column]
        pytorch_code, tests = split_at_tests(pytorch_code, example["entrypoint"])

        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT.format(pytorch_code=pytorch_code)},
                {"role": "assistant", "content": "<think>"},
            ],
            "tests": tests
        }

    # Format each example in the dataset
    formatted_data = data.map(format_example)

    formatted_data = formatted_data.filter(lambda x: x["tests"] is not None)
    print(f"Filtered to {len(formatted_data)} examples with non-None tests")

    return formatted_data

if __name__ == "__main__":
    args = sp.parse(Args)
    dataset = get_dataset(args.dataset_name, "train", args.code_column)
    dataset.save_to_disk("train_dataset")