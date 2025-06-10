import asyncio
import random
import openai
import weave
from pathlib import Path
from dataclasses import dataclass
from pydantic import BaseModel, Field
from datasets import load_dataset
import simple_parsing as sp

client = openai.AsyncOpenAI()

triton_cookbook = Path("./data/triton_cookbook.md").read_text()

@dataclass
class Args:
    model: str = "gpt-4.1"
    weave_project: str = "grpo-cuda/boostrap_triton2"
    output_ds_name: str = "tcapelle/boostrap_triton"
    init: bool = False
    N: int = 1
    rows: int = 100

args = sp.parse(Args)

boostrap_data = "./data/simple_samples.jsonl"

if args.init:
    ds = load_dataset("json", data_files=boostrap_data)["train"]
    print("="*100)
    print(ds[0]["pt_code"])
    print("-"*100)
    print(ds[0]["triton_code"])
    print("="*100)
else:
    ds = load_dataset(args.output_ds_name)["train"]

print(ds)


class PyTorchTritonRow(BaseModel):
    conversion_reasoning: str = Field(description="The reasoning step by step on how the conversion to triton should be done for this specific function")
    pt_code: str = Field(description="The PyTorch code for the function")
    triton_code: str = Field(description="The Triton code for the function")
    pt_entrypoint: str = Field(description="The entrypoint of the function in Pytorch")
    triton_entrypoint: str = Field(description="The entrypoint of the function in Triton")

system_prompt = f"""We are generating a PyTorch/Triton pairs dataset. We want functions that have exactly the same funcionalities.

Please generate a new row for our dataset. Focus on clarity and simplicity, Triton can be very complex, the idea is generating pairs that are easy to understand and that can be used to learn Triton.

Here it's a best practice on writing Triton kernels: {triton_cookbook}

Return the reasoning step by step on how the conversion to triton should be done for this specific function.
"""

user_prompt = """Our dataset is comprised now of the following rows:
{formatted_rows}

Be creative and think about dataset diversity, let's not make every row of the dataset row identical. Don't just re-implement top level torch operations like torch.sum, torch.floor, etc.. those have Triton primitives already.
Generate a new sample pair of PyTorch and Triton code that we can add to the dataset.

Keep the format of naming your triton entrypoint the same as the PyTorch entrypoint with _triton suffix.
"""

weave.init(args.weave_project)


def join_past_rows(ds, rows_to_sample):
    "sample `rows_to_sample` randomly from ds"
    rows = random.sample(ds.to_list(), rows_to_sample)
    formatted_rows = "\n".join([f"== {i} == \n{row['pt_code']}\n---\n{row['triton_code']}" for i, row in enumerate(rows)])
    return formatted_rows

async def generate_one_more_row(ds):
    formatted_rows = join_past_rows(ds, args.rows)
    # print(formatted_rows)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt.format(formatted_rows=formatted_rows)}
    ]

    response = await client.responses.parse(
        model=args.model,
        input=messages,
        text_format=PyTorchTritonRow,
    )
    extra_row = response.output_parsed
    print("="*100)
    print(extra_row.pt_code)
    print("-"*100)
    print(extra_row.conversion_reasoning)
    print("-"*100)
    print(extra_row.triton_code)
    print("="*100)
    return extra_row

async def generate_all_rows():
    tasks = [generate_one_more_row(ds) for _ in range(args.N)]
    return await asyncio.gather(*tasks)

extra_rows = asyncio.run(generate_all_rows())

for row in extra_rows:
    ds = ds.add_item(row.model_dump())


ds.push_to_hub(args.output_ds_name)