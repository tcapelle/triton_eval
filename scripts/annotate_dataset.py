"Reverse the jit triton kernel to pytorch code"

import openai
from rich.console import Console
from rich.pretty import pprint
from datasets import load_dataset, load_from_disk
from pydantic import BaseModel, Field

from prompts import generate_pytorch_prompt, KernelInfo, description_prompt, PytorchCodeWithTestCases

console = Console()



console.rule("[bold blue]Prompt[/bold blue]")


# console.rule("[bold blue]Generating Descriptions[/bold blue]")

client = openai.OpenAI()

## Add some description to the dataset
######################################
# dataset_name = "GPUMODE/categorized_triton_data_permissive"
# dataset = load_dataset(dataset_name, split="train")

# def generate_description(row):
#     code = row["input"]
#     response = client.beta.chat.completions.parse(
#         model="gpt-4o",
#         messages=[{"role": "user", "content": description_prompt.format(code=code)}],
#         response_format=KernelInfo,
#     )
#     return {"description": response.choices[0].message.parsed.description}



# dataset = dataset.map(generate_description, num_proc=40)
# dataset.save_to_disk("annotated_dataset")

## Generate Pytorch Code
######################################

# dataset = load_from_disk("annotated_dataset")

# def generate_pytorch_code(row):
#     code = row["input"]
#     description = row["description"]
#     response = client.beta.chat.completions.parse(
#         model="o3-mini",
#         messages=[{"role": "user", "content": generate_pytorch_prompt.format(code=code, description=description)}],
#         response_format=PytorchCodeWithTestCases,
#     )
#     return {"pytorch_code_with_test_cases": response.choices[0].message.parsed.pytorch_code_with_test_cases}


# dataset = dataset.map(generate_pytorch_code, num_proc=40)
# dataset.save_to_disk("annotated_dataset_pt")
# dataset.push_to_hub("tcapelle/annotated_dataset_o3")

## Run code
######################################

dataset = load_from_disk("annotated_dataset_pt")

from tools import run_python_code

def run_code(row):
    pytorch_code_with_test_cases = row["pytorch_code_with_test_cases"]
    out = run_python_code(pytorch_code_with_test_cases)
    print(f"=============== Running code ==========================")
    print(f"status: {out['status_code']}")
    if out['status_code'] != 0:
        print(f"error: {out['output']}")
    return {"test_cpu_passing": out["status_code"] == 0, "test_cpu_output": out["output"]}

# Only run first 10 examples
# dataset = dataset.select(range(10))
dataset = dataset.map(run_code, num_proc=12)
dataset.save_to_disk("annotated_dataset_pt_tested")
dataset.push_to_hub("tcapelle/annotated_dataset_o3")

