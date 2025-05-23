from datasets import load_from_disk
from dataclasses import dataclass
import simple_parsing as sp

@dataclass
class ScriptArgs:
    ds_folder: str = sp.field(positional=True, help="Name of the HuggingFace dataset to load")
    output_name: str = sp.field(default="tcapelle/boostrap_triton_ran", help="Name of the HuggingFace dataset to push")

args = sp.parse(ScriptArgs)

ds = load_from_disk(args.ds_folder)

ds.push_to_hub(args.output_name)