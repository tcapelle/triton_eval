from dataclasses import dataclass
from datasets import load_dataset, load_from_disk
import simple_parsing as sp

from prompts import eval_system_prompt, eval_user_prompt


@dataclass
class Args:
    ds_name: str = "tcapelle/boostrap_triton_ran"
    debug: bool = False
    pt_col: str = "pt_code"
    triton_col: str = "triton_code"
    entrypoint_col: str = "pt_entrypoint"
    reasoning_col: str = None
    code_token: str = "triton"
    reasoning_token: str = "reasoning"

args = sp.parse(Args)

try:
    ds = load_dataset(args.ds_name)["train"]
except:
    ds = load_from_disk(args.ds_name)

print(ds)

def format_example(example):
    pt_code = example[args.pt_col]
    entrypoint = example[args.entrypoint_col]

    messages = [
        {"role": "system", "content": eval_system_prompt.format(code_token=args.code_token, reasoning_token=args.reasoning_token)},
        {"role": "user", "content": eval_user_prompt.format(pt_code=pt_code, entrypoint=entrypoint, code_token=args.code_token, reasoning_token=args.reasoning_token)},
    ]
    if args.reasoning_col:
        reasoning = example[args.reasoning_col]
        triton_code = example[args.triton_col]
        output = f"<reasoning>\n{reasoning}\n</reasoning>\n\n3. Triton Code:\n<triton>\n{triton_code}\n</triton>"
        messages.append({"role": "assistant", "content": output})


    return {
        "messages": messages,
        "entrypoint": entrypoint,
    }

if not args.debug:
   formatted_ds = ds.map(format_example)
   formatted_ds.push_to_hub(args.ds_name + "_sft")