import asyncio
import weave
import openai
from dataclasses import dataclass
from datasets import load_dataset, Dataset
import simple_parsing as sp
from triton_eval.utils import map
from prompts import sft_system_prompt, sft_user_prompt, TorchTritonReasoning

client = openai.AsyncOpenAI()

@dataclass
class Args:
    ds_name: str = "tcapelle/boostrap_triton_ran"
    model: str = "gpt-4.1"
    num_proc: int = 20
    weave_project: str = "grpo-cuda/llm-tricks"
    output_ds_name: str = "tcapelle/boostrap_triton_ran"
    push: bool = False
    debug: bool = False

args = sp.parse(Args)

ds = load_dataset(args.ds_name)["train"]


async def format_row(row):
    messages = [
        {"role": "system", "content": sft_system_prompt},
        {"role": "user", "content": sft_user_prompt.format(pt_code=row["pt_code"], triton_code=row["triton_code"])}
    ]
    response = await client.responses.parse(
        model=args.model,
        input=messages,
        text_format=TorchTritonReasoning,
    )
    reasoning = response.output_parsed.reasoning
    row["reasoning"] = reasoning
    if args.debug:
        print(reasoning)
        print("-"*100)
    return {"reasoning": reasoning}



# weave.init(args.weave_project)

if args.debug:
    ds = ds.select(range(10))

out_ds = asyncio.run(map(ds, format_row, num_proc=args.num_proc))
out_ds = Dataset.from_list(out_ds)
out_ds.save_to_disk(args.output_ds_name.replace("/", "_"))

if args.push:
    out_ds.push_to_hub(args.output_ds_name)